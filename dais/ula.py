import jax.numpy as np
import jax
import numpyro.distributions as npdist
from jax.flatten_util import ravel_pytree
import functools
from . import variationaldist as vd
from .nn import amortize_eps_nn


def initialize(
    dim,
    vdparams=None,
    nbridges=0,
    eps=0.01,
    gamma=10.0,
    eta=0.5,
    trainable=["eps"],
    epsdim=1,
    epsbound=0.5,
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
        init_fun_eps, apply_fun_eps = amortize_eps_nn(epsdim, epsbound)
        params_train["eps"] = init_fun_eps(jax.random.PRNGKey(1), (-1, 1))[1]
    else:
        apply_fun_eps = None
        params_notrain["eps"] = eps
        print("No stepsize network needed by the method.")

    if "gamma" in trainable:
        params_train["gamma"] = gamma
    else:
        params_notrain["gamma"] = gamma

    if "eta" in trainable:
        params_train["eta"] = eta
    else:
        params_notrain["eta"] = eta

    # Everything related to betas
    mgridref_y = np.ones(nbridges)
    if "mgridref_y" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    time_correct_bw = True
    params_fixed = (dim, nbridges, time_correct_bw, apply_fun_eps)
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob):
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, nbridges, _, _ = params_fixed
    # time_correct_bw = params_fixed[2]

    if nbridges >= 1:
        # setup betas by transforming the gridref
        gridref_y = np.cumsum(params["mgridref_y"]) / np.sum(params["mgridref_y"])
        betas = np.concatenate([np.array([0.0]), gridref_y])

    rng_key_gen = jax.random.PRNGKey(seed)

    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    z = vd.sample_rep(rng_key, params["vd"])

    # match G1 defined in the end of Sec 4.1 of adaptive smc paper
    w = np.array(0.0)

    # Evolve ULA and compute weights
    if nbridges >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z, w_upd, _ = evolve_ula_amortize(
            z,
            betas,
            params,
            rng_key,
            params_fixed,
            log_prob,
            sample_kernel,
            log_prob_kernel,
        )
        w += w_upd

    # Update weight with final model evaluation
    w = w + log_prob(z)
    return -1.0 * w, (z, _)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
    ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob
    )
    logz_est = jax.scipy.special.logsumexp(-ratios) - np.log(ratios.shape[0])
    return ratios.mean(), (ratios.mean(), logz_est, z)


# For transition kernel
def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def evolve_ula_amortize(
    z,
    betas,
    params,
    rng_key_gen,
    params_fixed,
    log_prob_model,
    sample_kernel,
    log_prob_kernel,
):
    dim, nbridges, time_correct_bw, apply_fun_eps = params_fixed

    def U(z, beta):
        return -1.0 * (
            beta * log_prob_model(z) + (1.0 - beta) * vd.log_prob(params["vd"], z)
        )

    def first_evolve(aux):
        z, w, rng_key_gen = aux
        beta = betas[1]

        # Forward kernel
        fk_mean = z - apply_fun_eps(params["eps"], np.array([beta])) * jax.grad(U, 0)(
            z, beta
        )  # - because it is gradient of U = -log \pi
        scale = np.sqrt(2 * apply_fun_eps(params["eps"], np.array([beta])))

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # forward kernel
        # match G1 defined in the end of Sec 4.1 of adaptive smc paper
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)

        # Update weight and return
        w -= fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    def evolve(aux, i):
        z, w, rng_key_gen = aux
        beta = betas[i]
        beta_prev = betas[i - 1]

        # Forward kernel
        fk_mean = z - apply_fun_eps(params["eps"], np.array([beta])) * jax.grad(U, 0)(
            z, beta
        )  # - because it is gradient of U = -log \pi
        scale = np.sqrt(2 * apply_fun_eps(params["eps"], np.array([beta])))

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        bk_mean = z_new - apply_fun_eps(
            params["eps"], np.array([beta_prev])
        ) * jax.grad(U, 0)(
            z_new, beta
        )  # recover Thin et al. but with time corrected bw kernel
        scale_prev = np.sqrt(
            2 * apply_fun_eps(params["eps"], np.array([beta_prev]))
        )

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z, bk_mean, scale_prev)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)

    aux, _ = first_evolve(aux)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges)[2:])

    z, w, _ = aux
    return z, w, None


def n_calls_per_iter(params_fixed, batchsize):
    nbridges = params_fixed[1]
    nlpdf_call = batchsize
    ngrad_call = (2 * nbridges - 1) * batchsize
    return nlpdf_call, ngrad_call
