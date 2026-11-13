# Multi-node CPU experiment

The report includes Atlas job 1040644 and Vanda job 1547560. Both use 1/2/4 nodes from one allocation, 6 MPI ranks x 2 threads per node, block decomposition and TCP transport. Each benchmark uses 512x512x64 cells and 200 steps, one warmup and three repeats per layout/exchange. The baseline is the corresponding one-node MPI run.

Before timing, each layout passes serial/MPI field comparisons on a 65x49x9 case. The driver records placement, bindings, modules, CPU models, configurations and checksums. Diagnostics check mass balance and positivity throughout.

## Atlas

Complete the root README build/validation steps first. Then:

```bash
job_id=$(qsub scripts/atlas_multinode.pbs)
printf '%s\n' "$job_id" > .multinode-job-id
qstat -f "$job_id"
tail -n 60 -F "logs/atlas-multinode-${job_id}.log"
```

The script reserves four nodes with 12 CPUs per node and scatter placement. It uses the existing validated `build/atlas-cpu` executables and `.venv-atlas`. On the tested Atlas hosts, selecting `eth0` resolved a TCP connection timeout; the script records this choice.

## Vanda

From a fresh extracted submission, verify `sha256sum -c SHA256SUMS` before changing files. On the login node, load the modules and prepare dependencies:

```bash
source scripts/vanda_env.sh
module load Python/3.12.3-GCCcore-13.3.0
bash scripts/fetch_kokkos.sh
python3 -m venv .venv-multinode
.venv-multinode/bin/python -m pip install numpy==2.2.6
git -C external/kokkos rev-parse HEAD > KOKKOS_REVISION.txt
job_id=$(qsub -q auto -N dispersion-4node \
  -l select=4:ncpus=12:mpiprocs=6:ompthreads=2:mem=16gb \
  scripts/vanda_multinode.pbs)
printf '%s\n' "$job_id" > .vanda-multinode-job-id
qstat -f "$job_id"
tail -n 60 -F "logs/vanda-multinode-${job_id}.log"
```

Git must be available for the pinned Kokkos download. The batch script checks source hashes, builds job-specific executables and runs CTest before the experiment. Vanda routes the request through `auto`; allocations and queue policies may differ for other accounts. Replace the PBS project name with your own allocation when necessary.

For either cluster, a two-node request supports a 1/2-node experiment. Batch jobs continue after SSH disconnection; Ctrl+C on `tail` only stops viewing the log.

## Check and retrieve

Require the final SUCCESS message and `results/CLUSTER-multinode-JOBID/SUCCESS.json`. For a completed job, `qstat -xf JOBID` shows historical PBS status. Copy the result directory and job log back, then run:

```bash
.venv/bin/python scripts/review_multinode_run.py results/CLUSTER-multinode-JOBID
.venv/bin/python scripts/plot_multinode_review.py results/CLUSTER-multinode-JOBID --cluster CLUSTER
```

The review recomputes field comparisons, timing medians, speedups and efficiencies, and checks recorded bindings. Preserve timing ranges and separate cluster/exchange baselines. The launch wrapper finds `prted` from the loaded modules and exports its runtime environment through `PRTE_MCA_prte_launch_agent` before MPI starts.
