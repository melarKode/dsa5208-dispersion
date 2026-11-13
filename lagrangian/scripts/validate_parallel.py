#!/usr/bin/env python3
"""Validates the MPI+Kokkos solver (lagrangian_parallel) against the serial reference.

Standard library only (Python >= 3.6), so it runs on a cluster login or compute node.

For every case the serial solver runs once, then the parallel solver runs with
1, 2 and 4 ranks (default x-y rank grid, plus 4 ranks as 4x1x1 slabs) and with
several OpenMP thread counts. Each parallel run is compared with the serial run:

  particles      every global id present exactly once at every snapshot, and the
                 full state (position, unwrapped position, image counters via the
                 unwrapped position, mass, active flag) bit-identical to serial.
                 Justification for zero tolerance: both executables link the same
                 compiled per-particle code (lps::step_particle), and a particle's
                 update depends only on its own state, id and step, never on rank,
                 thread or storage order.
  diagnostics    reductions are summed in a different order, so moments may differ
                 by round-off: |a - b| <= 1e-10 * max(1, |a|) (recursive summation
                 bound ~ N * 1.1e-16 = 2e-11 for N = 2e5, with margin); masses are
                 compensated sums, |a - b| <= 1e-13; integer columns and times exact;
                 nan must match nan.
  concentration  every cell; each cell's mass comes from one rank only, so cells
                 must agree to round-off: |a - b| <= 1e-15 + 1e-12 |a|.
  accounting     final active + escaped = initial count; per-rank stored counts
                 sum to the initial count at every output; escaped, wrap and
                 off-home-image counts equal serial.

Case-specific requirements (multi-subdomain jumps, empty ranks, release-time
escapes, total escape, walls straddling a rank boundary) are checked too.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"

MOMENT_TOLERANCE = 1e-10
MASS_TOLERANCE = 1e-13
CELL_ABS_TOLERANCE = 1e-15
CELL_REL_TOLERANCE = 1e-12

# (label, ranks, --mpi-dims or None for the solver's default)
RANK_LAYOUTS = [("1", 1, None), ("2", 2, None), ("4", 4, None), ("4-slabs", 4, "4,1,1")]


# ----------------------------------------------------------------------------- small helpers

def read_kv(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_rows(path):
    with open(str(path), newline="") as f:
        return list(csv.DictReader(f))


def as_float(text):
    return float("nan") if text == "nan" else float(text)


class Report:
    def __init__(self):
        self.lines = []
        self.failures = 0

    def say(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def check(self, name, passed, detail=""):
        if not passed:
            self.failures += 1
        self.say("    [{}] {}{}".format("PASS" if passed else "FAIL", name, ": " + detail if detail else ""))
        return passed


# ----------------------------------------------------------------------------- the prepared urban case

def write_urban_case(directory):
    """A small prepared case: a wall straddling a rank boundary plus a building
    block, closed y/z walls and an open x outlet, with a non-zero origin."""
    nx, ny, nz = 24, 16, 8
    directory.mkdir(parents=True)
    (directory / "case.cfg").write_text(
        "nx={}\nny={}\nnz={}\ndx=1\ndy=1\ndz=1\norigin_x=-5\norigin_y=2\norigin_z=0\n".format(nx, ny, nz))
    solid = bytearray(nx * ny * nz)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                wall = i in (11, 12) and j < 12            # cells 11|12 sit on both sides of the 4-slab boundary at i=12
                block = 16 <= i < 19 and 4 <= j < 9 and k < 5
                if wall or block:
                    solid[i + nx * (j + ny * k)] = 1
    (directory / "solid.bin").write_bytes(bytes(solid))
    for name, value in (("u", 3.0), ("v", 0.5), ("w", 0.0)):
        data = [0.0 if solid[c] else value for c in range(nx * ny * nz)]
        (directory / (name + ".bin")).write_bytes(struct.pack("<{}d".format(len(data)), *data))
    manifest = ""
    for name in ("case.cfg", "u.bin", "v.bin", "w.bin", "solid.bin"):
        manifest += hashlib.sha256((directory / name).read_bytes()).hexdigest() + "  " + name + "\n"
    (directory / "input.sha256").write_text(manifest)
    return solid, (nx, ny, nz)


def case_list(work):
    urban_dir = work / "p6-urban-case"
    solid, shape = write_urban_case(urban_dir)
    urban_cfg = work / "p6_urban_wall.cfg"
    urban_cfg.write_text(
        # Source just upstream of the wall, beside the gap at j >= 12: part of the
        # plume reflects off the wall, part slips through the gap and leaves at x.
        "case_name=p6_urban_wall\ncase_dir={}\nsource_x=3\nsource_y=13\nsource_z=4\nsource_sigma=1.5\n"
        "initial_mass=1\ndiffusivity=0.5\nparticle_count=10000\nseed=206\ndt=0.05\nend_time=8\n"
        "output_interval=2\nboundary_x=open\nboundary_y=closed\nboundary_z=closed\nwrite_binary=true\n"
        "write_particles=true\n".format(urban_dir.resolve()))
    return [
        dict(name="p1_periodic_multihop", cfg=CASES / "parallel" / "p1_periodic_multihop.cfg", sets=[]),
        dict(name="p2_open_escape", cfg=CASES / "parallel" / "p2_open_escape.cfg", sets=[]),
        dict(name="p3_empty_ranks", cfg=CASES / "parallel" / "p3_empty_ranks.cfg", sets=[]),
        dict(name="p4_all_escape", cfg=CASES / "parallel" / "p4_all_escape.cfg", sets=[]),
        dict(name="p5_analytic_v4", cfg=CASES / "validation" / "v4_advection_diffusion.cfg",
             sets=["case_name=p5_analytic_v4", "write_particles=true", "write_concentration=false",
                   "output_interval=5.0"]),
        dict(name="p6_urban_wall", cfg=urban_cfg, sets=[], solid=solid, shape=shape),
    ]


# ----------------------------------------------------------------------------- comparisons

def compare_particles(serial_dir, run_dir, snapshots, n_total):
    """Returns (ids_ok, mismatched_particles, max_abs_difference) over all snapshots."""
    ids_ok, mismatched, max_diff = True, 0, 0.0
    for snap in snapshots:
        name = "particles_{:06d}.csv".format(int(snap["snapshot_index"]))
        reference = {row["id"]: row for row in read_rows(serial_dir / name)}
        rows = read_rows(run_dir / name)
        ids = [row["id"] for row in rows]
        ids_ok &= len(rows) == n_total and len(set(ids)) == n_total and set(ids) == set(reference)
        for row in rows:
            ref = reference.get(row["id"])
            if ref is None:
                continue
            if row != ref:
                mismatched += 1
                for key in ("x", "y", "z", "x_unwrapped", "y_unwrapped", "z_unwrapped", "mass"):
                    max_diff = max(max_diff, abs(float(row[key]) - float(ref[key])))
    return ids_ok, mismatched, max_diff


def compare_diagnostics(serial_rows, run_rows):
    exact_ok, worst_moment, worst_mass = len(serial_rows) == len(run_rows), 0.0, 0.0
    for a, b in zip(serial_rows, run_rows):
        exact_ok &= a["time"] == b["time"] and a["active_particles"] == b["active_particles"]
        for key in ("mass_inside", "mass_escaped", "mass_residual"):
            worst_mass = max(worst_mass, abs(float(a[key]) - float(b[key])))
        for key in ("mean_x", "mean_y", "mean_z", "var_x", "var_y", "var_z"):
            x, y = as_float(a[key]), as_float(b[key])
            if math.isnan(x) or math.isnan(y):
                exact_ok &= math.isnan(x) and math.isnan(y)
            else:
                worst_moment = max(worst_moment, abs(x - y) / max(1.0, abs(x)))
    return exact_ok, worst_moment, worst_mass


def read_cells(path, binary):
    if binary:
        data = path.read_bytes()
        return list(struct.unpack("<{}d".format(len(data) // 8), data))
    return [(r["cell_id"], r["i"], r["j"], r["k"], float(r["mass"]), float(r["concentration"]))
            for r in read_rows(path)]


def compare_concentration(serial_dir, run_dir, snapshots, binary):
    layout_ok, worst, identical, cells = True, 0.0, 0, 0
    for snap in snapshots:
        name = "concentration_{:06d}.{}".format(int(snap["snapshot_index"]), "bin" if binary else "csv")
        a, b = read_cells(serial_dir / name, binary), read_cells(run_dir / name, binary)
        layout_ok &= len(a) == len(b)
        for x, y in zip(a, b):
            if not binary:
                layout_ok &= x[:4] == y[:4]
                x, y = x[4], y[4]
            cells += 1
            identical += x == y
            excess = abs(x - y) - (CELL_ABS_TOLERANCE + CELL_REL_TOLERANCE * abs(x))
            worst = max(worst, excess)
    return layout_ok, worst <= 0.0, identical, cells


# ----------------------------------------------------------------------------- driver

def run(command, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(str(log), "w") as out:
        result = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default=str(ROOT / "build" / "lagrangian_solver"))
    parser.add_argument("--parallel", default=str(ROOT / "build" / "lagrangian_parallel"))
    parser.add_argument("--mpirun", default="mpirun -n {ranks}",
                        help="launcher template; {ranks} and {threads} are substituted")
    parser.add_argument("--threads", default="1,2,4", help="OpenMP threads per rank to test")
    parser.add_argument("--out", default=str(ROOT / "parallel_validation_output"))
    parser.add_argument("--cases", default="", help="comma-separated subset of case names")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit("output directory {} must be empty or absent".format(out))
    out.mkdir(parents=True, exist_ok=True)
    threads = [int(t) for t in args.threads.split(",")]
    report = Report()
    summary = []
    report.say("parallel validation: {} vs {}".format(args.parallel, args.serial))
    report.say("launcher: {}   threads per rank: {}".format(args.mpirun, threads))

    cases = case_list(out)
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["name"] in wanted]

    for case in cases:
        name = case["name"]
        report.say("\n{}  ({})".format(name, case["cfg"].name))
        serial_dir = out / name / "serial"
        base = [str(case["cfg"])]
        for item in case["sets"]:
            base += ["--set", item]
        env = dict(os.environ)
        if run([args.serial] + base + ["--output", str(serial_dir)], env, out / name / "serial.log") != 0:
            report.check("serial reference run", False, "see " + str(out / name / "serial.log"))
            continue
        cfg = read_kv(serial_dir / "effective_config.txt")
        n_total = int(cfg["particle_count"])
        binary = cfg.get("write_binary") == "true"
        has_concentration = cfg.get("write_concentration") == "true"
        snapshots = read_rows(serial_dir / "snapshots.csv")
        serial_diag = read_rows(serial_dir / "diagnostics.csv")
        serial_summary = read_kv(serial_dir / "run_summary.txt")

        for label, ranks, dims in RANK_LAYOUTS:
            for t in threads:
                run_dir = out / name / "ranks{}_threads{}".format(label, t)
                command = shlex.split(args.mpirun.format(ranks=ranks, threads=t)) + [args.parallel] + base
                command += ["--output", str(run_dir), "--kokkos-num-threads={}".format(t)]
                if dims:
                    command += ["--mpi-dims", dims]
                env = dict(os.environ, OMP_NUM_THREADS=str(t))
                env.setdefault("OMP_PROC_BIND", "false")
                title = "ranks={} ({}) threads={}".format(ranks, dims or "default", t)
                report.say("  " + title)
                if run(command, env, out / name / "ranks{}_threads{}.log".format(label, t)) != 0:
                    report.check("parallel run", False, "see log next to " + str(run_dir))
                    continue
                summary_kv = read_kv(run_dir / "run_summary.txt")
                ok = report.check("layout and threads applied",
                                  summary_kv["mpi_ranks"] == str(ranks) and summary_kv["threads_per_rank"] == str(t)
                                  and (dims is None or summary_kv["mpi_dims"] == dims.replace(",", "x")),
                                  "{} ranks as {}, {} threads".format(summary_kv["mpi_ranks"], summary_kv["mpi_dims"],
                                                                      summary_kv["threads_per_rank"]))
                ok &= report.check("snapshot times identical",
                                   (run_dir / "snapshots.csv").read_text() == (serial_dir / "snapshots.csv").read_text())

                ids_ok, mismatched, max_diff = compare_particles(serial_dir, run_dir, snapshots, n_total)
                ok &= report.check("every id 0..N-1 exactly once at every snapshot", ids_ok)
                ok &= report.check("particle states bit-identical to serial (by global id)", mismatched == 0,
                                   "{} differing, max |diff| {:.3g} m".format(mismatched, max_diff))

                exact_ok, worst_moment, worst_mass = compare_diagnostics(serial_diag,
                                                                         read_rows(run_dir / "diagnostics.csv"))
                ok &= report.check("diagnostics within reduction round-off",
                                   exact_ok and worst_moment <= MOMENT_TOLERANCE and worst_mass <= MASS_TOLERANCE,
                                   "moments {:.2g} rel, mass {:.2g}".format(worst_moment, worst_mass))
                if has_concentration:
                    layout_ok, values_ok, identical, cells = compare_concentration(serial_dir, run_dir, snapshots,
                                                                                   binary)
                    ok &= report.check("concentration snapshots match ({})".format("binary" if binary else "csv"),
                                       layout_ok and values_ok,
                                       "{}/{} cells bit-identical".format(identical, cells))

                counts_match = all(summary_kv[k] == serial_summary[k] for k in
                                   ("escaped_particles", "periodic_wraps", "final_active_particles",
                                    "particles_off_home_image"))
                balance = int(summary_kv["final_active_particles"]) + int(summary_kv["escaped_particles"]) == n_total
                ok &= report.check("active + escaped = initial count; counts equal serial", counts_match and balance,
                                   "{} active + {} escaped, {} wraps".format(summary_kv["final_active_particles"],
                                                                           summary_kv["escaped_particles"],
                                                                           summary_kv["periodic_wraps"]))
                rank_rows = read_rows(run_dir / "rank_counts.csv")
                per_snapshot = {}
                for r in rank_rows:
                    per_snapshot.setdefault(r["snapshot_index"], []).append(r)
                stored_ok = all(len(rows) == ranks and sum(int(r["stored_particles"]) for r in rows) == n_total
                                for rows in per_snapshot.values())
                ok &= report.check("per-rank stored counts sum to N at every output", stored_ok)

                # Case-specific requirements.
                non_neighbour = int(summary_kv["non_neighbour_migrations"])
                if name == "p1_periodic_multihop":
                    if dims == "4,1,1":
                        ok &= report.check("particles jumped over whole subdomains in one step", non_neighbour > 0,
                                           "{} non-neighbour migrations".format(non_neighbour))
                    ok &= report.check("periodic crossings occurred", int(summary_kv["periodic_wraps"]) > 0)
                if name == "p2_open_escape":
                    ok &= report.check("escapes at release and during the run",
                                       float(read_rows(run_dir / "diagnostics.csv")[0]["mass_escaped"]) > 0
                                       and int(summary_kv["escaped_particles"]) > 0)
                if name == "p3_empty_ranks" and ranks > 1:
                    empty_ok = all(any(int(r["stored_particles"]) == 0 for r in rows) for rows in per_snapshot.values())
                    ok &= report.check("some ranks hold no particles at every output", empty_ok)
                if name == "p4_all_escape":
                    last = read_rows(run_dir / "diagnostics.csv")[-1]
                    ok &= report.check("everything escaped, moments are nan",
                                       last["active_particles"] == "0" and last["mean_x"] == "nan")
                if name == "p6_urban_wall":
                    nx, ny, nz = case["shape"]
                    final = read_rows(run_dir / "particles_{:06d}.csv".format(len(snapshots) - 1))
                    inside_solid = 0
                    for row in final:
                        if row["active"] != "1":
                            continue
                        i = int(math.floor(float(row["x"]) + 5)); j = int(math.floor(float(row["y"]) - 2))
                        k = int(math.floor(float(row["z"])))
                        inside_solid += case["solid"][i + nx * (j + ny * k)]
                    ok &= report.check("no active particle inside a building; some escaped at the outlet",
                                       inside_solid == 0 and int(summary_kv["escaped_particles"]) > 0,
                                       "{} escaped".format(summary_kv["escaped_particles"]))

                summary.append(dict(case=name, ranks=ranks, dims=summary_kv["mpi_dims"], threads=t, passed=bool(ok),
                                    particles_differing=mismatched, max_moment_rel_diff=worst_moment,
                                    migrations=int(summary_kv["migrated_particles"]),
                                    non_neighbour_migrations=non_neighbour,
                                    stepping_s=float(summary_kv["stepping_seconds_max"]),
                                    migration_s=float(summary_kv["migration_seconds_max"]),
                                    output_s=float(summary_kv["output_seconds_max"]),
                                    total_s=float(summary_kv["total_seconds_max"])))

    passed = sum(1 for s in summary if s["passed"])
    report.say("\n{} of {} parallel runs passed every check".format(passed, len(summary)))
    report.say("ALL CHECKS PASSED" if report.failures == 0 else "{} CHECK(S) FAILED".format(report.failures))
    (out / "parallel_validation_report.txt").write_text("\n".join(report.lines) + "\n")
    (out / "parallel_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
