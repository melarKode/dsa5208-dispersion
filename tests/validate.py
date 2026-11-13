"""Run analytical, conservation, I/O, and serial/MPI equivalence checks."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results import field, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", type=Path, default=ROOT / "build/serial/dispersion_serial")
    p.add_argument("--parallel", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-ranks", type=int, default=6)
    p.add_argument("--pbs", action="store_true", help="Use allocated cores and forward runtime environment")
    args = p.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="dispersion-validation-"))
    directory.mkdir(parents=True, exist_ok=True)
    serial = str(args.serial.resolve())
    parallel = str(args.parallel.resolve()) if args.parallel else None
    report = {"checks": [], "convergence": [], "directory": str(directory.resolve())}
    count = 0
    tested_ranks = set()

    def execute(parameters, ranks=0, expect_failure=False):
        nonlocal count
        count += 1
        cfg = dict(nx=13, ny=11, nz=5, dx=1, dy=1, dz=1, source_x=3,
                   source_y=5, source_z=2.5, source_sigma=.7, u=.3, v=-.1,
                   w=.05, diffusivity=.1, dt=0, end_time=.8, output_interval=.4,
                   boundary_x="periodic", boundary_y="periodic", boundary_z="periodic")
        cfg.update(parameters)
        destination = directory / f"run_{count:03d}"
        cfg["output_dir"] = str(destination.resolve())
        source = directory / f"case_{count:03d}.cfg"
        source.write_text("".join(f"{key}={value}\n" for key, value in cfg.items()))
        command = [serial, str(source)]
        if ranks:
            tested_ranks.add(ranks)
            if args.pbs:
                threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
                command = ["mpirun", "--map-by", f"slot:PE={threads}", "--bind-to", "core",
                           "-x", "LD_LIBRARY_PATH", "-x", "OMP_NUM_THREADS", "-x", "OMP_PLACES",
                           "-x", "OMP_PROC_BIND", "-x", "OMP_DYNAMIC", "-np", str(ranks), parallel, str(source)]
                if os.environ.get("LD_PRELOAD"):
                    command[1:1] = ["-x", "LD_PRELOAD"]
            else:
                command = ["mpirun", "--oversubscribe", "--bind-to", "none", "-np", str(ranks), parallel, str(source)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"run_{count:03d}.log").write_text(result.stdout + result.stderr)
        if expect_failure:
            assert result.returncode != 0, "Invalid input accepted"
            return
        if result.returncode:
            raise AssertionError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
        diagnostics = rows(destination / "diagnostics.csv")
        for row in diagnostics:
            assert abs(float(row["mass_residual"])) < 2e-11
            assert float(row["min_concentration"]) >= -1e-14
        return destination

    for name, velocities, diffusion in [("diffusion", (0, 0, 0), .1),
                                         ("advection", (.4, -.2, .1), 0),
                                         ("combined", (.4, -.2, .1), .1)]:
        errors = []
        for n in (12, 24, 48):
            h = 8 / n
            case = execute(dict(nx=n, ny=n, nz=n, dx=h, dy=h, dz=h, initial="cosine",
                                amplitude=.2, u=velocities[0], v=velocities[1], w=velocities[2],
                                diffusivity=diffusion, end_time=.7, output_interval=.7))
            _, t, actual = field(case)
            k = 2 * np.pi / 8
            x = (np.arange(n) + .5) * h
            terms = [np.cos(k * (x - speed * t)) * np.sinc(k * h / (2 * np.pi)) for speed in velocities]
            exact = (1 + .2 * np.exp(-3 * diffusion * k * k * t) *
                     terms[2][:, None, None] * terms[1][None, :, None] * terms[0][None, None, :]) / 8**3
            error = float(np.abs(actual - exact).sum() * h**3)
            errors.append(error)
            report["convergence"].append(dict(case=name, n=n, l1=error))
        assert errors[2] < errors[1] < errors[0], (name, errors)
        report["checks"].append(f"{name} analytical cell-average convergence")

    gaussian = execute(dict(nx=48, ny=48, nz=48, dx=.5, dy=.5, dz=.5,
                            source_x=10, source_y=12, source_z=12, source_sigma=1.5,
                            u=.3, v=-.1, w=.05, diffusivity=.2, end_time=1, output_interval=1))
    _, t, actual = field(gaussian)
    sigma = math.sqrt(1.5**2 + 2 * .2 * t)
    x = (np.arange(48) + .5) * .5
    axes = []
    for mean, speed in zip((10, 12, 12), (.3, -.1, .05)):
        hi = (x + .25 - mean - speed * t) / (math.sqrt(2) * sigma)
        lo = hi - .5 / (math.sqrt(2) * sigma)
        axes.append(np.array([math.erf(a) - math.erf(b) for a, b in zip(hi, lo)]))
    exact = axes[2][:, None, None] * axes[1][None, :, None] * axes[0][None, None, :]
    error = float(np.abs(actual - exact).sum() * .5**3)
    assert error < .06, error
    report["gaussian_l1"] = error
    report["checks"].append("Gaussian advection-diffusion cell-average solution")
    stationary = execute(dict(u=0, v=0, w=0, diffusivity=0))
    assert np.array_equal(field(stationary, 0)[2], field(stationary)[2])
    report["checks"].append("zero-transport identity")
    execute(dict(dt=100), expect_failure=True)
    execute(dict(nx=0), expect_failure=True)
    execute(dict(diffusivity=-1), expect_failure=True)
    execute(dict(typo_key=1), expect_failure=True)
    report["checks"].append("unsafe timestep and invalid configuration rejection")
    opened = execute(dict(boundary_x="open", boundary_y="closed", boundary_z="closed",
                          source_x=11, u=1, end_time=5, output_interval=1))
    assert float(rows(opened / "diagnostics.csv")[-1]["mass_escaped"]) > .5
    report["checks"].append("open-boundary escaped-mass accounting")

    shape = (5, 11, 13)
    z, y, x = np.indices(shape)
    u, v, w = .1 + x * .01, -.03 + y * .002, np.zeros(shape)
    solid = np.zeros(shape, dtype=np.uint8)
    solid[:, 4:7, 6] = 1
    fixture = directory / "fixture.h5"
    with h5py.File(fixture, "w") as f:
        vector = np.stack([u, v, w], axis=-1).transpose(1, 2, 0, 3)[None, ...]
        f["wind"] = vector
        f["solid"] = solid.transpose(1, 2, 0)[None, ...]
    prepared = directory / "prepared"
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_case.py"), str(fixture), str(prepared),
                    "--velocity", "wind", "--component-axis", "-1", "--case-index", "0",
                    "--axis-order", "yxz", "--solid", "solid", "--dx", "1", "--dy", "1", "--dz", "1"], check=True, capture_output=True)
    assert np.array_equal(np.fromfile(prepared / "solid.bin", dtype="u1").reshape(shape), solid)
    assert np.array_equal(np.fromfile(prepared / "u.bin", dtype="<f8").reshape(shape), np.where(solid, 0, u))
    report["checks"].append("HDF5 vector/case selection, axis conversion, binary mask round-trip")
    csv_case = execute(dict(snapshots="csv"))
    binary_case = execute({})
    assert np.array_equal(field(csv_case)[2], field(binary_case)[2])
    report["checks"].append("CSV/binary output equivalence")

    if parallel:
        for params in [{}, dict(boundary_x="open", boundary_y="closed", boundary_z="periodic"),
                       dict(wall_i=6, boundary_x="closed", boundary_y="closed", boundary_z="closed"),
                       dict(case_dir=str(prepared.resolve())),
                       dict(nx=3, ny=2, nz=1, source_x=1, source_y=1, source_z=.5)]:
            reference = execute(params)
            target = field(reference)[2]
            for decomposition, ranks in [("slab", 2), ("slab", 3), ("block", 1), ("block", 4)]:
                if params.get("nx") == 3:
                    ranks = 6 if decomposition == "block" else 3
                if ranks > args.max_ranks:
                    continue
                for exchange in ("blocking", "overlap"):
                    result = execute(dict(params, decomposition=decomposition, exchange=exchange), ranks)
                    error = float(np.abs(field(result)[2] - target).max())
                    assert error < 5e-13, (params, decomposition, ranks, exchange, error)
            report["checks"].append(f"MPI equivalence: {params or 'periodic, uneven dimensions'}")
        report["checks"].append(f"MPI ranks {sorted(tested_ranks)}; blocking/overlap; slab/block; width-one partitions")
    report["runs"] = count
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
