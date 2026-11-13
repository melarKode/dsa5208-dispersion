#!/usr/bin/env python3
"""Tabulates lagrangian_parallel scaling runs (one output directory per run).

Usage: scaling_summary.py [--weak] RUN_DIR [RUN_DIR ...] > scaling.csv

For each run it reads run_summary.txt, timings.csv and rank_counts.csv and
prints one CSV row: layout, timers (max over ranks), the share of the time
loop spent waiting (load imbalance) and migrating, load imbalance (max / mean
active particles per rank; maximum, mean and final over all outputs), and
speedup / efficiency against the run with the fewest cores
among those given (same case and particle count). With --weak, particle
counts may differ between runs (fixed work per core): runs are grouped by case
only and efficiency = loop time of the smallest run / loop time of this run.
Standard library only.
"""

import csv
import sys
from pathlib import Path


def read_kv(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def summarise(run_dir):
    run_dir = Path(run_dir)
    s = read_kv(run_dir / "run_summary.txt")
    with open(str(run_dir / "timings.csv"), newline="") as f:
        timings = list(csv.DictReader(f))
    with open(str(run_dir / "rank_counts.csv"), newline="") as f:
        counts = list(csv.DictReader(f))
    # Load imbalance = max / mean active particles per rank, at every output.
    imbalances = []
    for index in sorted({int(r["snapshot_index"]) for r in counts}):
        active = [int(r["active_particles"]) for r in counts if int(r["snapshot_index"]) == index]
        mean_active = sum(active) / len(active)
        if mean_active > 0:
            imbalances.append(max(active) / mean_active)
    ranks, threads = int(s["mpi_ranks"]), int(s["threads_per_rank"])
    stepping = [float(t["stepping_seconds"]) for t in timings]
    return dict(
        run=run_dir.name, case=s["case_name"], particles=int(s["particle_count"]), steps=int(s["num_steps"]),
        ranks=ranks, threads=threads, cores=ranks * threads, dims=s["mpi_dims"],
        total_s=float(s["total_seconds_max"]), setup_s=float(s["setup_seconds_max"]),
        stepping_s=float(s["stepping_seconds_max"]), migration_s=float(s["migration_seconds_max"]),
        output_s=float(s["output_seconds_max"]),
        stepping_min_s=min(stepping),
        wait_s=float(s["wait_seconds_max"]),
        loop_s=float(s["loop_seconds_max"]),
        migrated=int(s["migrated_particles"]),
        imbalance_final=imbalances[-1] if imbalances else float("nan"),
        imbalance_max=max(imbalances) if imbalances else float("nan"),
        imbalance_mean=sum(imbalances) / len(imbalances) if imbalances else float("nan"),
        particle_steps_per_s=int(s["particle_count"]) * int(s["num_steps"]) / max(1e-12, float(s["loop_seconds_max"])),
    )


def main():
    args = sys.argv[1:]
    weak = bool(args) and args[0] == "--weak"
    if weak:
        args = args[1:]
    if not args:
        sys.exit(__doc__)
    rows = [summarise(d) for d in args]
    # Speedup of the time loop (stepping + wait + migration) relative to the
    # smallest core count of the same case (and, for strong scaling, the same
    # particle count).
    for row in rows:
        group = [r for r in rows if r["case"] == row["case"] and (weak or r["particles"] == row["particles"])]
        base = min(group, key=lambda r: (r["cores"], r["ranks"]))
        ratio = base["loop_s"] / row["loop_s"]
        row["speedup"] = ratio * (row["particles"] / base["particles"]) if weak else ratio
        row["efficiency"] = ratio if weak else ratio * base["cores"] / row["cores"]
        row["migration_share"] = row["migration_s"] / row["loop_s"] if row["loop_s"] > 0 else 0.0
        row["wait_share"] = row["wait_s"] / row["loop_s"] if row["loop_s"] > 0 else 0.0
    columns = ["case", "particles", "cores", "ranks", "threads", "dims", "loop_s", "stepping_s", "stepping_min_s",
               "wait_s", "wait_share", "migration_s", "migration_share", "output_s", "setup_s", "total_s", "speedup",
               "efficiency", "imbalance_max", "imbalance_mean", "imbalance_final", "migrated", "particle_steps_per_s",
               "run"]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["case"], r["particles"], r["cores"], r["ranks"])):
        writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
