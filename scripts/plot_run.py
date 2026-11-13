"""Plot mass retention and vertically integrated plume snapshots."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from results import field, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--log-scale", action="store_true")
    args = p.parse_args()
    output = args.output or args.run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = rows(args.run / "diagnostics.csv")
    time = [float(r["time"]) for r in diagnostics]
    mass = [float(r["mass_inside"]) for r in diagnostics]
    escaped = [float(r["mass_escaped"]) for r in diagnostics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, mass, label="Remaining mass")
    ax.plot(time, escaped, label="Escaped mass")
    ax.set(xlabel="Time (s)", ylabel="Pollutant mass", title="Pollutant mass balance")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "mass.png", dpi=180); plt.close(fig)
    records = rows(args.run / "snapshots.csv")
    if records[0].get("format", "csv") == "none":
        return
    indices = sorted(set([0, len(records) // 2, len(records) - 1]))
    selected = [field(args.run, i) for i in indices]
    vmax = max(float((f.sum(axis=0) * float(c["dz"])).max()) for c, _, f in selected)
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), squeeze=False, constrained_layout=True)
    cmap=plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0.0))
    for ax, (cfg, t, f) in zip(axes[0], selected):
        column = f.sum(axis=0) * float(cfg["dz"])
        x0, y0 = float(cfg["origin_x"]), float(cfg["origin_y"])
        extent = [x0, x0 + int(cfg["nx"]) * float(cfg["dx"]), y0, y0 + int(cfg["ny"]) * float(cfg["dy"])]
        scale = dict(norm=LogNorm(vmin=max(vmax*1e-6, 1e-300), vmax=max(vmax, 1e-299))) if args.log_scale else dict(vmin=0, vmax=vmax)
        image = ax.imshow(np.ma.masked_less_equal(column, 0) if args.log_scale else column, origin="lower", extent=extent, cmap=cmap, **scale)
        ax.set(title=f"t = {t:g} s", xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(image, ax=axes[0].tolist(), label="Column mass density (mass/m²)" + (" · log scale" if args.log_scale else ""))
    fig.savefig(output / "plume.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
