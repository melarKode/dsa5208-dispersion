"""Compare concentration fields at matching physical times and model settings."""
import argparse
import json
from pathlib import Path
import numpy as np
from results import field


def compare(left, right, index=-1):
    a, ta, fa = field(left, index)
    b, tb, fb = field(right, index)
    keys = ["nx", "ny", "nz", "dx", "dy", "dz", "origin_x", "origin_y", "origin_z",
            "diffusivity", "initial_mass", "source_x", "source_y", "source_z", "source_sigma",
            "boundary_x", "boundary_y", "boundary_z", "velocity_mode", "u", "v", "w",
            "initial", "wall_i", "amplitude"]
    defaults = dict(initial="gaussian", wall_i="-1", amplitude="0.2", origin_x="0", origin_y="0", origin_z="0")
    def canonical(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    for key in keys:
        if key in ("u", "v", "w") and a.get("velocity_mode") == b.get("velocity_mode") == "file":
            continue
        if canonical(a.get(key, defaults.get(key))) != canonical(b.get(key, defaults.get(key))):
            raise ValueError(f"Different model setting: {key}")
    if (a.get("solver_family") == "particle") != (b.get("solver_family") == "particle"):
        particle = a if a.get("solver_family") == "particle" else b
        matched_urban = particle.get("velocity_mode") == "file" and particle.get("particle_boundary_model") == "split_advective_open_reflected_diffusion"
        if not matched_urban and any(a.get("boundary_"+d) != "periodic" for d in "xyz"):
            raise ValueError("Cross-method boundary laws are not yet matched; use periodic comparison cases")
    if a.get("velocity_mode") == "file":
        signatures = [Path(p) / "input.sha256" for p in (left, right)]
        if not all(p.exists() for p in signatures) or signatures[0].read_text() != signatures[1].read_text():
            raise ValueError("Urban comparisons require matching input.sha256 manifests")
    if not np.isclose(ta, tb, rtol=0, atol=1e-12):
        raise ValueError("Snapshot times differ")
    dv = float(a["dx"]) * float(a["dy"]) * float(a["dz"])
    difference = fa - fb
    return dict(time=ta, l1=float(np.abs(difference).sum() * dv),
                l2=float(np.sqrt(np.square(difference).sum() * dv)),
                linf=float(np.abs(difference).max()), mass_a=float(fa.sum() * dv),
                mass_b=float(fb.sum() * dv))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left"); p.add_argument("right")
    p.add_argument("--index", type=int, default=-1)
    p.add_argument("--tolerance", type=float)
    args = p.parse_args()
    report = compare(args.left, args.right, args.index)
    print(json.dumps(report, indent=2))
    if args.tolerance is not None and report["linf"] > args.tolerance:
        raise SystemExit(1)
