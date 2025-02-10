import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "dais/"))

import jax.numpy as np
import jax
import boundingmachine as bm
import opt
from model_handler import load_model

log_prob_model, dim = load_model('brownian')
rng_key_gen = jax.random.PRNGKey(1)

trainable = ("vd", "eps", "eta", "mgridref_y")
# trainable = ("eps", "eta", "mgridref_y")
params_flat, unflatten, params_fixed = bm.initialize(
    dim=dim,
    nbridges=16,
    eta=0.0,
    eps=0.00001,
    lfsteps=1,
    vdparams=None,
    trainable=trainable,
)
grad_and_loss = jax.jit(
    jax.grad(bm.compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4)
)


iters_base=1500
batchsize = 32
lr = 0.001

logzs, losses, diverged, params_flat, tracker = opt.run_repl(
    batchsize,
    lr,
    iters_base,
    params_flat,
    unflatten,
    params_fixed,
    log_prob_model,
    grad_and_loss,
    trainable,
    rng_key_gen,
)


# import plotting library
import matplotlib.pyplot as plt
plt.plot(logzs)
plt.show()


final_elbo = -np.mean(np.array(losses[-500:]))
print("Done training, got ELBO %.2f." % final_elbo)

# tracker['elbo_init'] = elbo_init
tracker["elbo_final"] = final_elbo
tracker["logzs"] = logzs

print(diverged)
print(tracker)

# params_train, params_notrain = unflatten(params_flat)
# params = {**params_train, **params_notrain}
