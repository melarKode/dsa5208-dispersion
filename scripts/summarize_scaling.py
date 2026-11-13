"""Summarize repeated fixed-problem timings; speedup uses a measured serial reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from results import config, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    groups = {}
    for path in args.root.rglob("timing.csv"):
        cfg = config(path.parent / "effective.cfg")
        numerical = {k: v for k, v in cfg.items() if k not in
                     {"output_dir", "decomposition", "exchange", "px", "py", "case_name", "particle_count", "seed"}}
        fingerprint = hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()[:16]
        row = rows(path)[0]
        key = (fingerprint, row["backend"], row["ranks"], row["threads"], row["decomposition"], row["exchange"], row["steps"])
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No timing files found")
    baseline = {}
    for key, records in groups.items():
        if key[1] == "serial":
            baseline.setdefault((key[0], key[6]), []).extend(float(r["solve_seconds"]) for r in records)
    output = args.output or args.root / "scaling.csv"
    fields = ["problem", "backend", "ranks", "threads", "decomposition", "exchange", "steps", "repeats",
              "median_solve_seconds", "min_solve_seconds", "max_solve_seconds", "median_elapsed_seconds",
              "speedup_vs_serial", "cpu_core_efficiency", "max_rank_host_rss_mib"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for key, records in sorted(groups.items()):
            problem, backend, ranks, threads, decomposition, exchange, steps = key
            times = [float(r["solve_seconds"]) for r in records]
            median = statistics.median(times)
            reference = baseline.get((problem, steps))
            speedup = statistics.median(reference) / median if reference and median > 0 else None
            cpu = backend.lower() in ("serial", "openmp", "threads")
            efficiency = speedup / (int(ranks) * int(threads)) if cpu and speedup is not None else None
            writer.writerow(dict(problem=problem, backend=backend, ranks=ranks, threads=threads,
                                 decomposition=decomposition, exchange=exchange, steps=steps, repeats=len(times),
                                 median_solve_seconds=median, min_solve_seconds=min(times), max_solve_seconds=max(times),
                                 median_elapsed_seconds=statistics.median(float(r["elapsed_seconds"]) for r in records),
                                 speedup_vs_serial=speedup, cpu_core_efficiency=efficiency,
                                 max_rank_host_rss_mib=max(float(r["max_rank_host_rss_mib"]) for r in records)))
    print(output)


if __name__ == "__main__":
    main()
