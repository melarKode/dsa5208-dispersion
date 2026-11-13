#!/usr/bin/env python3
"""Runs the particle-solver validation cases and checks them against analytical solutions.

Standard library only (no numpy), so it runs on a bare cluster login node.

Deterministic cases (v1 stationary, v2 translation, v5 periodic wrap) are checked
per particle, by id, against x(t) = x(0) + u t.

Statistical cases (v3 diffusion, v4 advection + diffusion) are run with several
seeds. With a Gaussian release and constant u, K the particle cloud is exactly
Gaussian with mean x0 + u t and variance s0^2 + 2 K t, so the only error is
sampling error. Each estimate is turned into a z-score using its standard error:
    SE(mean) = sqrt(var / N),   SE(var) = var * sqrt(2 / (N - 1))
and must satisfy |z| < Z_LIMIT, both per seed and for the seed-averaged estimate.

v6 checks escaped-mass accounting through the provisional open boundary, and the
example case checks the concentration snapshot contract and compares the binned
mass with the cell-integrated analytical Gaussian.
"""

import argparse
import csv
import filecmp
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases" / "validation"
Z_LIMIT = 4.5  # ~1e-5 two-sided per comparison; a few hundred comparisons per case


# ----------------------------------------------------------------------------- helpers

def read_kv(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def run_case(exe, cfg, out_dir, overrides=()):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Run from the project root with relative paths so effective_config.txt
    # records portable paths rather than this machine's absolute ones.
    rel = lambda p: os.path.relpath(p, ROOT)
    cmd = [str(exe), rel(cfg), "--output", rel(out_dir)]
    for item in overrides:
        cmd += ["--set", item]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
    return read_kv(out_dir / "effective_config.txt")


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


class Report:
    def __init__(self):
        self.lines = []
        self.failures = 0

    def say(self, text=""):
        print(text)
        self.lines.append(text)

    def check(self, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        if not passed:
            self.failures += 1
        self.say(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


AXES = ("x", "y", "z")


def velocity(cfg):
    return [float(cfg["u"]), float(cfg["v"]), float(cfg["w"])]


def source(cfg):
    return [float(cfg["source_x"]), float(cfg["source_y"]), float(cfg["source_z"])]


def check_mass(report, rows, n_particles, initial_mass, allow_escape=False):
    worst = max(abs(r["mass_residual"]) for r in rows)
    report.check("mass residual |initial - inside - escaped| < 1e-12", worst < 1e-12, f"max {worst:.2e}")
    if not allow_escape:
        ok = all(r["active_particles"] == n_particles and r["mass_escaped"] == 0.0 for r in rows)
        report.check("no particles lost, mass_escaped = 0", ok)
        ok = all(abs(r["mass_inside"] - initial_mass) < 1e-12 for r in rows)
        report.check("mass_inside = initial_mass", ok)


# ----------------------------------------------------------------------------- deterministic cases

def deterministic_case(report, exe, name, out_root, expect_wraps):
    cfg_path = CASES / f"{name}.cfg"
    out = out_root / name
    cfg = run_case(exe, cfg_path, out)
    report.say(f"\n{name}  ({cfg_path.relative_to(ROOT)})")
    u = velocity(cfg)
    n = int(cfg["particle_count"])
    lengths = [int(cfg[k]) * float(cfg[d]) for k, d in (("nx", "dx"), ("ny", "dy"), ("nz", "dz"))]
    origins = [float(cfg.get(f"origin_{a}", 0.0)) for a in AXES]

    snapshots = read_csv(out / "snapshots.csv")
    initial = read_csv(out / "particles_000000.csv")
    worst_disp, worst_box, ids_ok = 0.0, True, True
    for snap in snapshots:
        t = snap["time"]
        current = read_csv(out / f"particles_{int(snap['snapshot_index']):06d}.csv")
        ids_ok &= len(current) == n and all(int(a["id"]) == int(b["id"]) for a, b in zip(initial, current))
        for a, b in zip(initial, current):
            for d, axis in enumerate(AXES):
                disp = b[f"{axis}_unwrapped"] - a[f"{axis}_unwrapped"]
                worst_disp = max(worst_disp, abs(disp - u[d] * t))
                worst_box &= origins[d] <= b[axis] < origins[d] + lengths[d]
    report.check("every particle id present exactly once at every snapshot", ids_ok)
    report.check("per-particle displacement = u t (unwrapped)", worst_disp < 1e-9, f"max error {worst_disp:.2e} m")
    report.check("wrapped positions inside [origin, origin + L)", worst_box)

    rows = read_csv(out / "diagnostics.csv")
    check_mass(report, rows, n, float(cfg["initial_mass"]))
    worst_mean = max(abs(r[f"mean_{a}"] - rows[0][f"mean_{a}"] - u[d] * r["time"])
                     for r in rows for d, a in enumerate(AXES))
    worst_var = max(abs(r[f"var_{a}"] / rows[0][f"var_{a}"] - 1.0) for r in rows for a in AXES)
    report.check("diagnostics mean shifts by exactly u t", worst_mean < 1e-9, f"max error {worst_mean:.2e} m")
    report.check("diagnostics variance unchanged", worst_var < 1e-9, f"max relative change {worst_var:.2e}")

    summary = read_kv(out / "run_summary.txt")
    wraps = int(summary["periodic_wraps"])
    if expect_wraps:
        report.check("cloud actually crossed periodic faces", wraps > 0, f"{wraps} wraps")
        conc_rows = read_csv(out / f"concentration_{len(snapshots) - 1:06d}.csv")
        total = math.fsum(r["mass"] for r in conc_rows)
        report.check("concentration snapshot after wrapping still holds all mass", abs(total - 1.0) < 1e-12,
                     f"sum {total!r}")
    else:
        report.check("no periodic wrapping (boundary effects absent)", wraps == 0, f"{wraps} wraps")


# ----------------------------------------------------------------------------- statistical cases

def statistical_case(report, exe, name, out_root, seeds):
    cfg_path = CASES / f"{name}.cfg"
    report.say(f"\n{name}  ({cfg_path.relative_to(ROOT)}), {len(seeds)} seeds")
    per_seed = []
    cfg = None
    total_wraps = 0
    for seed in seeds:
        out = out_root / name / f"seed_{seed}"
        cfg = run_case(exe, cfg_path, out, [f"seed={seed}", "write_concentration=false"])
        per_seed.append(read_csv(out / "diagnostics.csv"))
        total_wraps += int(read_kv(out / "run_summary.txt")["periodic_wraps"])

    n = int(cfg["particle_count"])
    u, x0 = velocity(cfg), source(cfg)
    s0, k = float(cfg["source_sigma"]), float(cfg["diffusivity"])
    check_mass(report, [row for rows in per_seed for row in rows], n, float(cfg["initial_mass"]))
    report.check("no periodic wrapping in any seed (boundary effects absent)", total_wraps == 0,
                 f"{total_wraps} wraps")

    report.say("     time  exact var   seed-avg var (x, y, z)            max|z| seed  max|z| pooled  sd ratio")
    worst_seed_z, worst_pooled_z, all_z = 0.0, 0.0, []
    sd_ratios = []
    for i, row0 in enumerate(per_seed[0]):
        t = row0["time"]
        var = s0 ** 2 + 2.0 * k * t
        expected_var = var * (n - 1) / n  # population variance estimator is biased by var / N
        se_mean, se_var = math.sqrt(var / n), var * math.sqrt(2.0 / (n - 1))
        row_seed_z, row_pooled_z, avg_vars = 0.0, 0.0, []
        for d, axis in enumerate(AXES):
            exact_mean = x0[d] + u[d] * t
            for stat, exact, se in (("mean", exact_mean, se_mean), ("var", expected_var, se_var)):
                estimates = [rows[i][f"{stat}_{axis}"] for rows in per_seed]
                z = [(e - exact) / se for e in estimates]
                all_z += z
                row_seed_z = max(row_seed_z, max(abs(v) for v in z))
                avg = sum(estimates) / len(estimates)
                row_pooled_z = max(row_pooled_z, abs(avg - exact) / (se / math.sqrt(len(estimates))))
                if len(estimates) > 1:
                    sd = math.sqrt(sum((e - avg) ** 2 for e in estimates) / (len(estimates) - 1))
                    sd_ratios.append(sd / se)
                if stat == "var":
                    avg_vars.append(avg)
        worst_seed_z = max(worst_seed_z, row_seed_z)
        worst_pooled_z = max(worst_pooled_z, row_pooled_z)
        ratio = sum(sd_ratios[-6:]) / 6 if len(seeds) > 1 else float("nan")
        report.say(f"   {t:6.2f}  {var:9.5f}   {avg_vars[0]:9.5f} {avg_vars[1]:9.5f} {avg_vars[2]:9.5f}"
                   f"     {row_seed_z:6.2f}       {row_pooled_z:6.2f}      {ratio:5.2f}")

    rms_z = math.sqrt(sum(v * v for v in all_z) / len(all_z))
    report.check(f"every per-seed |z| < {Z_LIMIT} (mean and variance, all axes and times)",
                 worst_seed_z < Z_LIMIT, f"max {worst_seed_z:.2f} over {len(all_z)} estimates")
    report.check(f"seed-averaged |z| < {Z_LIMIT}", worst_pooled_z < Z_LIMIT, f"max {worst_pooled_z:.2f}")
    if len(seeds) > 1:
        report.say(f"  info: RMS z = {rms_z:.2f} (expect ~1); seed-to-seed sd / theoretical SE = "
                   f"{sum(sd_ratios) / len(sd_ratios):.2f} (expect ~1)")


def reproducibility_check(report, exe, out_root):
    report.say("\nreproducibility (v4, same seed twice)")
    cfg_path = CASES / "v4_advection_diffusion.cfg"
    a, b = out_root / "repro" / "a", out_root / "repro" / "b"
    run_case(exe, cfg_path, a, ["write_concentration=false"])
    run_case(exe, cfg_path, b, ["write_concentration=false"])
    same = filecmp.cmp(a / "diagnostics.csv", b / "diagnostics.csv", shallow=False)
    report.check("diagnostics.csv byte-identical for identical seed", same)


# ----------------------------------------------------------------------------- open boundary

def open_outflow_case(report, exe, out_root, seeds):
    name = "v6_open_outflow"
    cfg_path = CASES / f"{name}.cfg"
    report.say(f"\n{name}  ({cfg_path.relative_to(ROOT)}), {len(seeds)} seeds")
    worst_z, worst_res = 0.0, 0.0
    report.say("     time  exact escaped   seed-avg escaped")
    table = {}
    for seed in seeds:
        out = out_root / name / f"seed_{seed}"
        cfg = run_case(exe, cfg_path, out, [f"seed={seed}"])
        n = int(cfg["particle_count"])
        m0 = float(cfg["initial_mass"])
        upper = float(cfg.get("origin_x", 0.0)) + int(cfg["nx"]) * float(cfg["dx"])
        x0, s0, u = float(cfg["source_x"]), float(cfg["source_sigma"]), float(cfg["u"])
        for r in read_csv(out / "diagnostics.csv"):
            p = 1.0 - phi((upper - x0 - u * r["time"]) / s0)
            se = m0 * math.sqrt(max(p * (1 - p), 1.0 / n) / n)
            worst_z = max(worst_z, abs(r["mass_escaped"] - m0 * p) / se)
            worst_res = max(worst_res, abs(r["mass_residual"]))
            consistent = abs(r["active_particles"] * m0 / n - r["mass_inside"]) < 1e-12
            if not consistent:
                worst_res = max(worst_res, 1.0)
            table.setdefault(r["time"], (m0 * p, []))[1].append(r["mass_escaped"])
    for t, (exact, values) in sorted(table.items()):
        report.say(f"   {t:6.2f}  {exact:12.6f}   {sum(values) / len(values):12.6f}")
    report.check("mass residual < 1e-12 and mass_inside = active * m_p", worst_res < 1e-12, f"max {worst_res:.2e}")
    report.check(f"escaped mass matches 1 - Phi((L - x0 - u t)/s0), |z| < {Z_LIMIT}", worst_z < Z_LIMIT,
                 f"max {worst_z:.2f}")


# ----------------------------------------------------------------------------- snapshot contract

def poisson_mad(lam):
    """Mean absolute deviation E|X - lam| of a Poisson(lam) count."""
    if lam <= 0.0:
        return 0.0
    k = math.floor(lam)
    return 2.0 * math.exp((k + 1) * math.log(lam) - lam - math.lgamma(k + 1))


def example_case(report, exe, out_dir):
    cfg_path = ROOT / "cases" / "example_gaussian_uniform.cfg"
    report.say(f"\nexample sample output ({cfg_path.relative_to(ROOT)}) -> {os.path.relpath(out_dir, ROOT)}")
    cfg = run_case(exe, cfg_path, out_dir)
    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    dx, dy, dz = float(cfg["dx"]), float(cfg["dy"]), float(cfg["dz"])
    origin = [float(cfg["origin_x"]), float(cfg["origin_y"]), float(cfg["origin_z"])]
    n, m0 = int(cfg["particle_count"]), float(cfg["initial_mass"])
    u, x0 = velocity(cfg), source(cfg)
    s0, k = float(cfg["source_sigma"]), float(cfg["diffusivity"])
    widths = (dx, dy, dz)
    counts = (nx, ny, nz)

    diag = read_csv(out_dir / "diagnostics.csv")
    snaps = read_csv(out_dir / "snapshots.csv")
    report.check("one diagnostics row per snapshot", len(diag) == len(snaps), f"{len(snaps)} snapshots")
    contract_ok, mass_ok, ratios = True, True, []
    for snap, drow in zip(snaps, diag):
        t = snap["time"]
        rows = read_csv(out_dir / f"concentration_{int(snap['snapshot_index']):06d}.csv")
        contract_ok &= len(rows) == nx * ny * nz
        for r in rows:
            i, j, kk = int(r["i"]), int(r["j"]), int(r["k"])
            contract_ok &= int(r["cell_id"]) == i + nx * (j + ny * kk)
            contract_ok &= abs(r["concentration"] - r["mass"] / (dx * dy * dz)) <= 1e-15 * max(1.0, r["concentration"])
        total = math.fsum(r["mass"] for r in rows)
        mass_ok &= abs(total - drow["mass_inside"]) < 1e-12

        # Cell-integrated analytical Gaussian vs binned particle mass.
        sd = math.sqrt(s0 ** 2 + 2.0 * k * t)
        frac = []
        for d in range(3):
            centre = x0[d] + u[d] * t
            edges = [phi((origin[d] + c * widths[d] - centre) / sd) for c in range(counts[d] + 1)]
            frac.append([edges[c + 1] - edges[c] for c in range(counts[d])])
        l1, expected_l1 = 0.0, 0.0
        for r in rows:
            p = frac[0][int(r["i"])] * frac[1][int(r["j"])] * frac[2][int(r["k"])]
            l1 += abs(r["mass"] - m0 * p)
            expected_l1 += m0 * poisson_mad(n * p) / n
        ratios.append(l1 / expected_l1)
        report.say(f"   t={t:4.2f}  L1(binned - analytic) = {l1:.5f}, expected from sampling noise {expected_l1:.5f}"
                   f"  (ratio {l1 / expected_l1:.3f})")
    report.check("concentration files: all cells present, cell_id = i + nx(j + ny k), c = m / (dx dy dz)",
                 contract_ok)
    report.check("snapshot mass sums to mass_inside (|diff| < 1e-12)", mass_ok)
    report.check("binned mass agrees with analytic Gaussian to within sampling noise (L1 ratio in [0.85, 1.15])",
                 all(0.85 <= r <= 1.15 for r in ratios))


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", default=str(ROOT / "build" / "lagrangian_solver"))
    parser.add_argument("--seeds", type=int, default=10, help="seeds per statistical case")
    parser.add_argument("--out", default=str(ROOT / "validation_output"))
    parser.add_argument("--sample-out", default=str(ROOT / "examples" / "sample_output" / "gaussian_uniform"))
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # Distinct seed ranges per case so the cases are statistically independent.
    def seeds(base, count):
        return [base + s for s in range(count)]

    report = Report()
    report.say(f"validation of {os.path.relpath(exe, ROOT)}")

    deterministic_case(report, exe, "v1_stationary", out_root, expect_wraps=False)
    deterministic_case(report, exe, "v2_translation", out_root, expect_wraps=False)
    statistical_case(report, exe, "v3_diffusion", out_root, seeds(3000, args.seeds))
    statistical_case(report, exe, "v4_advection_diffusion", out_root, seeds(4000, args.seeds))
    deterministic_case(report, exe, "v5_periodic_wrap", out_root, expect_wraps=True)
    open_outflow_case(report, exe, out_root, seeds(6000, max(1, args.seeds // 2)))
    reproducibility_check(report, exe, out_root)
    example_case(report, exe, Path(args.sample_out).resolve())

    report.say(f"\n{'ALL CHECKS PASSED' if report.failures == 0 else f'{report.failures} CHECK(S) FAILED'}")
    (out_root / "validation_report.txt").write_text("\n".join(report.lines) + "\n")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
