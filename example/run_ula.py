import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))

import jax.numpy as np
import jax
from dais import ula
from dais import opt
from model_handler import load_model

import matplotlib.pyplot as plt

log_prob_model, dim = load_model("brownian")
rng_key_gen = jax.random.PRNGKey(1)
# trainable = ("vd", "eps", "eta", "mgridref_y")
trainable = ("eps", "eta", "mgridref_y")
params_flat, unflatten, params_fixed = ula.initialize(
    dim=dim,
    nbridges=16,
    eta=0.0,
    eps=0.00001,
    vdparams=None,
    trainable=trainable,
    mode="DAIS_ULA_TC",
    epsdim=dim,
    epsbound=0.25,
)
grad_and_loss = jax.jit(
    jax.grad(ula.compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4)
)
callback_eval = jax.jit(ula.compute_bound, static_argnums=(2, 3, 4))

niters = 1000
batchsize = 32
eval_bs = 128
lr = 0.001

logzs_eval, logzs, losses, diverged, params_flat, tracker = opt.run_with_track(
    log_prob_model,
    grad_and_loss,
    batchsize,
    lr,
    niters,
    params_flat,
    unflatten,
    params_fixed,
    trainable,
    callback_eval,
    eval_bs,
    rng_key_gen,
)


plt.plot(logzs_eval)
plt.plot(logzs)
save_path = "figure/ula_bm_eval.png"
plt.savefig(save_path)
