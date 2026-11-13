"""Audit the exact arithmetic face velocities and solid masking used by the solver."""
import argparse
import json
from pathlib import Path
import numpy as np
from results import config


def audit(path, boundaries, plane_callback=None):
    if (path / "INCOMPLETE").exists():
        raise ValueError("Case conversion is incomplete")
    cfg = config(path / "case.cfg")
    nx, ny, nz = (int(cfg[f"n{axis}"]) for axis in "xyz")
    h = [float(cfg[f"d{axis}"]) for axis in "xyz"]
    shape = (nz, ny, nx)
    mask = np.memmap(path / "solid.bin", dtype="u1", mode="r", shape=shape)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid solid mask")
    wind = [np.memmap(path / f"{axis}.bin", dtype="<f8", mode="r", shape=shape) for axis in "uvw"]
    maximum, l1, squares, count, net, fluid_volume = 0., 0., 0., 0, 0., 0.
    blocked_max = 0.
    speed_max = 0.
    max_location = None
    blocked_location = None
    def location(z, j, i):
        return dict(ijk=[int(i), int(j), int(z)], xyz=[
            float(cfg.get(f"origin_{axis}", 0)) + (index + .5) * spacing
            for axis, index, spacing in zip("xyz", [i, j, z], h)])
    for z in range(nz):
        fluid = mask[z] == 0
        div = np.zeros((ny, nx))
        speed_max = max(speed_max, float(np.sqrt(sum(v[z]**2 for v in wind))[fluid].max(initial=0)))
        for axis in range(3):
            own = np.asarray(wind[axis][z])
            if not np.isfinite(own[fluid]).all():
                raise ValueError("Nonfinite wind in air")
            for sign in (-1, 1):
                if axis < 2:
                    dimension = 1 if axis == 0 else 0
                    other = np.roll(own, -sign, axis=dimension)
                    other_fluid = np.roll(fluid, -sign, axis=dimension)
                    outside = np.zeros_like(fluid)
                    if axis == 0:
                        outside[:, -1 if sign == 1 else 0] = True
                    else:
                        outside[-1 if sign == 1 else 0, :] = True
                else:
                    zz = z + sign
                    outside = np.full_like(fluid, zz < 0 or zz >= nz)
                    other = np.asarray(wind[axis][zz % nz])
                    other_fluid = mask[zz % nz] == 0
                raw = sign * .5 * (own + other)
                crossing = fluid & ~other_fluid
                if boundaries[axis] != "periodic":
                    crossing &= ~outside
                candidate = float(np.abs(raw[crossing]).max(initial=0))
                if candidate > blocked_max:
                    j, i = np.unravel_index(np.argmax(np.where(crossing, np.abs(raw), -1)), raw.shape)
                    blocked_location = dict(**location(z, j, i), axis="xyz"[axis], sign=sign,
                                            raw_normal_speed=float(raw[j, i]),
                                            air_cell_component=float(own[j, i]))
                    blocked_max = candidate
                face = np.where(fluid & other_fluid, raw, 0.)
                if boundaries[axis] == "closed":
                    face[outside] = 0
                elif boundaries[axis] == "open":
                    face[outside] = np.where(fluid, sign * own, 0)[outside]
                div += face / h[axis]
        values = div[fluid]
        candidate = float(np.abs(values).max(initial=0))
        if candidate > maximum:
            j, i = np.unravel_index(np.argmax(np.where(fluid, np.abs(div), -1)), div.shape)
            max_location = dict(**location(z, j, i), divergence=float(div[j, i]))
            maximum = candidate
        if plane_callback is not None:
            plane_callback(z, div, fluid)
        l1 += float(np.abs(values).sum()); squares += float(np.square(values).sum())
        count += values.size; net += float(values.sum()) * np.prod(h)
        fluid_volume += values.size * np.prod(h)
    report = dict(shape_zyx=shape, boundaries_xyz=boundaries, air_cells=count,
                  solid_cells=int(mask.size - count), max_speed=speed_max,
                  max_abs_divergence=maximum, mean_abs_divergence=l1 / count if count else None,
                  rms_divergence=np.sqrt(squares / count) if count else None,
                  net_outer_volume_flux=net, fluid_volume=fluid_volume,
                  max_raw_normal_speed_at_blocked_faces=blocked_max,
                  note="No divergence projection applied. Masking wall flux can change divergence.")
    report["max_divergence_location"] = max_location
    report["max_blocked_face_location"] = blocked_location
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--boundaries", nargs=3, choices=["open", "closed", "periodic"], default=["open", "open", "closed"])
    p.add_argument("--output", type=Path)
    p.add_argument("--max-divergence", type=float)
    a = p.parse_args()
    report = audit(a.case, a.boundaries)
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if a.output:
        a.output.write_text(text + "\n")
    if a.max_divergence is not None and report["max_abs_divergence"] > a.max_divergence:
        raise SystemExit(1)
