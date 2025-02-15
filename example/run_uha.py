import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
sys.path.insert(1, os.path.join(sys.path[0], "../"))

import jax.numpy as np
import jax
from dais import uha
from dais import opt
from model_handler import load_model
import pandas as pd


def main(
    id=1,
    target_name="brownian",
    niters=500,
    batchsize=32,
    eval_bs=1024,
    lr=0.001,
    nbridge=64,
    epsbound=0.25,
    res_dir="results",
):
    fname = f"{res_dir}/{target_name}/uha/nbridge_{nbridge}_lr_{lr}_bs_{batchsize}_id_{id}.csv"
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    if os.path.exists(fname):
        print(f"File {fname} already exists, skipping run")
        return

    log_prob_model, dim = load_model(target_name)
    rng_key_gen = jax.random.PRNGKey(id)

    # trainable = ("vd", "eps", "eta", "mgridref_y")
    trainable = ("eps", "eta", "mgridref_y")  # no vd param training
    params_flat, unflatten, params_fixed = uha.initialize(
        dim=dim,
        nbridges=nbridge,
        eta=0.0,
        eps=0.00001,
        lfsteps=1,
        vdparams=None,
        trainable=trainable,
        epsmode="amortize",
        epsdim=dim,
        epsbound=epsbound,
    )
    grad_and_loss = jax.jit(
        jax.grad(uha.compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4)
    )
    callback_eval = jax.jit(uha.compute_bound, static_argnums=(2, 3, 4))
    savings, logzs, losses, diverged, params_flat, _ = opt.run_with_track(
        log_prob_model,
        grad_and_loss,
        batchsize,
        lr,
        niters,
        params_flat,
        unflatten,
        params_fixed,
        trainable,
        uha.n_calls_per_iter,
        callback_eval,
        eval_bs,
        rng_key_gen,
    )

    df = pd.DataFrame(savings)
    df["lr"] = lr
    df["nbridge"] = nbridge
    df["batchsize"] = batchsize
    df["niters"] = niters
    df["method"] = "uha"
    df["id"] = id

    df.to_csv(fname, index=False)
    return df


if __name__ == "__main__":
    import argparse

    args_parser = argparse.ArgumentParser(description="Process arguments")
    args_parser.add_argument(
        "--target", type=str, default="brownian", help="Target to fit"
    )
    args_parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Number of samples to estimate gradient at each step.",
    )
    args_parser.add_argument(
        "--nbridges", type=int, default=16, help="Number of bridging densities."
    )
    args_parser.add_argument(
        "--lfsteps", type=int, default=1, help="Leapfrog steps, for UHA."
    )
    args_parser.add_argument(
        "--niters", type=int, default=5000, help="Number of optimization iterations."
    )
    args_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args_parser.add_argument(
        "--epsbound", type=float, default=0.25, help="Bound for stepsize."
    )
    args_parser.add_argument(
        "--id", type=int, default=-1, help="Unique ID for each run."
    )
    args = args_parser.parse_args()

    main(
        id=args.id,
        target_name=args.target,
        niters=args.niters,
        batchsize=args.bs,
        lr=args.lr,
        nbridge=args.nbridges,
        epsbound=args.epsbound,
        res_dir="results",
    )
