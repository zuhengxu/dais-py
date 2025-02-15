import jax.numpy as np
import jax
from . import variationaldist as vd
from . import momdist as md
from .nn import amortize_eps_nn
from jax.flatten_util import ravel_pytree
import functools


def initialize(
    dim,
    vdparams=None,
    nbridges=0,
    lfsteps=1,
    eps=0.0,
    eta=0.5,
    mdparams=None,
    ngridb=32,
    mgridref_y=None,
    trainable=["eps", "eta"],
    epsmode = "amortize",
    epsdim = 1,
    epsbound = 0.25,
):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    if "vd" in trainable:
        params_train["vd"] = vdparams
        if vdparams is None:
            params_train["vd"] = vd.initialize(dim)
    else:
        params_notrain["vd"] = vdparams
        if vdparams is None:
            params_notrain["vd"] = vd.initialize(dim)

    if "eps" in trainable:
        if epsmode == "amortize":
            init_fun_eps, apply_fun_eps = amortize_eps_nn(epsdim, epsbound)
            params_train["eps"] = init_fun_eps(jax.random.PRNGKey(1), (-1, 1))[1]
        else:
            apply_fun_eps = None
            params_train["eps"] = eps
    else:
        apply_fun_eps = None
        params_notrain["eps"] = eps

    if "eta" in trainable:
        params_train["eta"] = eta
    else:
        params_notrain["eta"] = eta

    if "md" in trainable:
        params_train["md"] = mdparams
        if mdparams is None:
            params_train["md"] = md.initialize(dim)
    else:
        params_notrain["md"] = mdparams
        if mdparams is None:
            params_notrain["md"] = md.initialize(dim)

    # Everything related to betas
    if mgridref_y is not None:
        ngridb = mgridref_y.shape[0] - 1
    else:
        if nbridges < ngridb:
            ngridb = nbridges
        mgridref_y = np.ones(ngridb + 1) * 1.0
    params_notrain["gridref_x"] = np.linspace(0, 1, ngridb + 2)
    params_notrain["target_x"] = np.linspace(0, 1, nbridges + 2)[1:-1]
    if "mgridref_y" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, nbridges, lfsteps, apply_fun_eps)
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob):
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, nbridges, lfsteps, apply_fun_eps = params_fixed

    if nbridges >= 1:
        # setup betas by transforming the gridref
        gridref_y = np.cumsum(params["mgridref_y"]) / np.sum(params["mgridref_y"])
        gridref_y = np.concatenate([np.array([0.0]), gridref_y])
        betas = np.interp(params["target_x"], params["gridref_x"], gridref_y)

    rng_key_gen = jax.random.PRNGKey(seed)

    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    z = vd.sample_rep(rng_key, params["vd"])
    w = -vd.log_prob(params["vd"], z)

    # Evolve UHA and update weight
    delta_H = np.array([0.0])
    if nbridges >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        if apply_fun_eps is not None:
            z, w_mom, delta_H = evolve_amortize(
                z, betas, params, rng_key, params_fixed, log_prob
            ) 
        else:
            z, w_mom, delta_H = evolve(
                z, betas, params, rng_key, params_fixed, log_prob
            )
        w += w_mom

    # Update weight with final model evaluation
    w = w + log_prob(z)
    delta_H = np.max(np.abs(delta_H))
    # delta_H = np.mean(np.abs(delta_H))
    return -1.0 * w, (z, delta_H)




# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
    ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob
    )
    logz_est = jax.scipy.special.logsumexp(-ratios) - np.log(ratios.shape[0])
    return ratios.mean(), (ratios.mean(), logz_est, z)





def evolve(z, betas, params, rng_key_gen, params_fixed, log_prob):
    def U(z, beta):
        return -1.0 * (beta * log_prob(z) + (1.0 - beta) * vd.log_prob(params["vd"], z))

    def evolve_bridges(aux, i):
        z, rho_prev, w, rng_key_gen = aux
        beta = betas[i]
        # Re-sample momentum
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho = md.sample(rng_key, params["eta"], rho_prev, params["md"])
        # Simulate dynamics
        z_new, rho_new, delta_H = leapfrog(z, rho, beta)
        # Update weight
        w = w + md.log_prob(rho_new, params["md"]) - md.log_prob(rho, params["md"])
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        # return aux, z2_new
        return aux, delta_H

    def leapfrog(z, rho, beta):
        def K(rho):
            return -1.0 * md.log_prob(rho, params["md"])

        def full_leap(aux, i):
            z, rho = aux
            rho = rho - params["eps"] * jax.grad(U, 0)(z, beta)
            z = z + params["eps"] * jax.grad(K, 0)(rho)
            aux = (z, rho)
            return aux, None

        # Half step for momentum
        U_init, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - params["eps"] * U_grad / 2.0
        # Full step for z
        K_init, K_grad = jax.value_and_grad(K, 0)(rho)
        z = z + params["eps"] * K_grad

        # # Alternate full steps
        if lfsteps > 1:
            aux = (z, rho)
            aux = jax.lax.scan(full_leap, aux, np.arange(lfsteps - 1))[0]
            z, rho = aux

        # Half step for momentum
        U_final, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - params["eps"] * U_grad / 2.0
        K_final = K(rho)

        delta_H = U_init + K_init - U_final - K_final

        return z, rho, delta_H

    nbridges = params_fixed[1]
    lfsteps = params_fixed[2]
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = md.sample(rng_key, params["eta"], None, params["md"])
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, rho, 0, rng_key_gen)
    aux, delta_H = jax.lax.scan(evolve_bridges, aux, np.arange(nbridges))
    z, _, w, _ = aux
    return z, w, delta_H


def evolve_amortize(z, betas, params, rng_key_gen, params_fixed, log_prob):
    nbridges = params_fixed[1]
    lfsteps = params_fixed[2]
    apply_fun_eps = params_fixed[3]

    def U(z, beta):
        return -1.0 * (beta * log_prob(z) + (1.0 - beta) * vd.log_prob(params["vd"], z))

    def evolve_bridges(aux, i):
        z, rho_prev, w, rng_key_gen = aux
        beta = betas[i]
        # Re-sample momentum
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho = md.sample(rng_key, params["eta"], rho_prev, params["md"])
        # Simulate dynamics
        z_new, rho_new, delta_H = leapfrog(z, rho, beta)
        # Update weight
        w = w + md.log_prob(rho_new, params["md"]) - md.log_prob(rho, params["md"])
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        # return aux, z2_new
        return aux, delta_H

    def leapfrog(z, rho, beta):
        def K(rho):
            return -1.0 * md.log_prob(rho, params["md"])

        def full_leap(aux, i):
            z, rho = aux
            rho = rho - apply_fun_eps(params["eps"], np.array([beta])) * jax.grad(U, 0)(
                z, beta
            )
            z = z + apply_fun_eps(params["eps"], np.array([beta])) * jax.grad(K, 0)(rho)
            aux = (z, rho)
            return aux, None

        # Half step for momentum
        U_init, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - apply_fun_eps(params["eps"], np.array([beta])) * U_grad / 2.0
        # Full step for z
        K_init, K_grad = jax.value_and_grad(K, 0)(rho)
        z = z + apply_fun_eps(params["eps"], np.array([beta])) * K_grad

        # # Alternate full steps
        if lfsteps > 1:
            aux = (z, rho)
            aux = jax.lax.scan(full_leap, aux, np.arange(lfsteps - 1))[0]
            z, rho = aux

        # Half step for momentum
        U_final, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - apply_fun_eps(params["eps"], np.array([beta])) * U_grad / 2.0
        K_final = K(rho)

        delta_H = U_init + K_init - U_final - K_final
        

        return z, rho, delta_H

    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = md.sample(rng_key, params["eta"], None, params["md"])
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, rho, 0, rng_key_gen)
    aux, delta_H = jax.lax.scan(evolve_bridges, aux, np.arange(nbridges))
    z, _, w, _ = aux
    return z, w, delta_H



def _n_calls_per_iter(nbridges, lfsteps, batchsize):
    lpdf_calls = (2 * nbridges + 1)* batchsize
    grad_calls = (nbridges *(lfsteps + 1))* batchsize
    return lpdf_calls, grad_calls

def n_calls_per_iter(params_fixed, batchsize):
    nbridges = params_fixed[1]
    lfsteps = params_fixed[2]
    return _n_calls_per_iter(nbridges, lfsteps, batchsize)
