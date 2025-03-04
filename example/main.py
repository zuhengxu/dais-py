import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))

from run_uha import run_uha
from run_ula import run_ula
import submitit

def main(
    method="uha",
    id=id,
    target_name="brownian",
    optimize_vd=False,
    niters=5000,
    batchsize=32,
    lr=0.001,
    nbridge=64,
    epsbound_ula=0.1,
    epsbound_uha=0.25,
    res_dir="results",
):
    if method == "uha":
        run_uha(
            id=id,
            target_name=target_name,
            optimize_vd=optimize_vd,
            niters=niters,
            batchsize=batchsize,
            lr=lr,
            nbridge=nbridge,
            epsbound=epsbound_uha,
            res_dir=res_dir,
        )
    elif method == "ula":
        run_ula(
            id=id,
            target_name=target_name,
            optimize_vd=optimize_vd,
            niters=niters,
            batchsize=batchsize,
            lr=lr,
            nbridge=nbridge,
            epsbound=epsbound_ula,
            res_dir=res_dir,
        )
    else:
        raise ValueError(f"Method {method} not recognized")


def submit_jobs():
    # Executor is part of submitit, it automatically manages job submissions
    executor = submitit.AutoExecutor(
        folder="../../logs/"
    )  # logs and checkpoints will be stored here

    # Set up Slurm parameters
    executor.update_parameters(
        slurm_array_parallelism=128,
        mem_gb=16,  # Memory in GB
        gpus_per_node=1,  # Number of GPUs
        cpus_per_task=2,  # Number of CPUs
        nodes=1,  # Number of nodes
        tasks_per_node=1,  # Tasks per node
        slurm_time=60*2,  # Time in minutes
        slurm_job_name="run_cox",  # Job name
        slurm_account="st-tdjc-1-gpu",
        local_setup=["export JAX_PLATFORMS=''"],
    )

    #####################
    # job arrays
    #####################
    print("Creating job arrays")
    # Define the hyperparameters to sweep through
    methods = ["uha", "ula"]
    ids = range(1, 33)
    # targets = ["brownian", "neal", "log_sonar"]
    targets = ["coxpine"]
    vds = [False]
    lrs = [0.01, 0.001, 0.0001]
    ntemps = [32, 64]
    ebs_uha = [0.25]
    ebs_ula = [0.1]
    
    
    jobs = []
    args = [
        (method, target, vd, lr, nt, eb_uha, eb_ula, id)
        for method in methods
        for target in targets
        for vd in vds
        for lr in lrs
        for nt in ntemps
        for eb_uha in ebs_uha
        for eb_ula in ebs_ula
        for id in ids
    ]
    
    with executor.batch():
        for arg in args:
            method, target, vd, lr, nt, eb_uha, eb_ula, id = arg
            job = executor.submit(
                main, 
                method=method,
                id=id,
                target_name=target,
                optimize_vd=vd,
                niters=5000,
                batchsize=32,
                lr=lr,
                nbridge=nt,
                epsbound_ula=eb_ula,
                epsbound_uha=eb_uha,
                res_dir = "results",
            )
            jobs.append(job)

if __name__ == "__main__":
    submit_jobs() 