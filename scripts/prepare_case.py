"""Convert explicitly selected HDF5 fields into the shared little-endian case format."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def selection(dataset, case_index, component_axis=None, component=None):
    indices = [slice(None)] * dataset.ndim
    if case_index is not None:
        indices[0] = case_index
    if component_axis is not None:
        axis = component_axis % dataset.ndim
        if axis == 0 and case_index is not None:
            raise ValueError("Case and component axes overlap")
        indices[axis] = component
    remaining = [a for a, value in enumerate(indices) if isinstance(value, slice)]
    if len(remaining) != 3:
        raise ValueError(f"Expected three spatial axes after selection: {dataset.name} {dataset.shape}")
    return indices, remaining


def planes(dataset, args, component=None, mask=False):
    case_index = args.case_index if not mask or dataset.ndim == 4 else None
    indices, axes = selection(dataset, case_index,
                              args.component_axis if component is not None else None, component)
    shape = {name: dataset.shape[axis] for name, axis in zip(args.axis_order, axes)}
    zaxis = axes[args.axis_order.index("z")]
    remaining_order = args.axis_order.replace("z", "")
    for z in range(shape["z"]):
        slab = indices.copy()
        slab[zaxis] = z
        values = np.asarray(dataset[tuple(slab)])
        if remaining_order == "xy":
            values = values.T
        yield values


def prepare(args):
    if sorted(args.axis_order) != ["x", "y", "z"]:
        raise ValueError("axis-order must be a permutation of xyz")
    if args.output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if not np.isfinite([args.dx, args.dy, args.dz, *args.origin]).all() or min(args.dx, args.dy, args.dz) <= 0:
        raise ValueError("Spacing must be positive")
    if not args.solid and not args.all_fluid:
        raise ValueError("Supply a verified solid-mask dataset or explicitly select --all-fluid")
    with h5py.File(args.file, "r") as h5:
        if args.velocity:
            if args.component_axis is None:
                raise ValueError("Vector velocity requires --component-axis in original dataset coordinates")
            fields = [(h5[args.velocity], i) for i in args.components]
        else:
            if not all((args.u, args.v, args.w)):
                raise ValueError("Supply --velocity or all of --u --v --w")
            fields = [(h5[name], None) for name in (args.u, args.v, args.w)]
        shape = None
        for dataset, component in fields:
            _, axes = selection(dataset, args.case_index, args.component_axis if component is not None else None, component)
            sizes = {name: dataset.shape[axis] for name, axis in zip(args.axis_order, axes)}
            candidate = tuple(sizes[name] for name in "zyx")
            if shape is not None and shape != candidate:
                raise ValueError("Velocity shapes differ")
            shape = candidate
        nz, ny, nx = shape
        args.output.mkdir(parents=True)
        (args.output / "INCOMPLETE").write_text("Conversion has not completed.\n")
        for name, (dataset, component) in zip("uvw", fields):
            with (args.output / f"{name}.bin").open("wb") as out:
                for plane in planes(dataset, args, component):
                    np.asarray(plane, dtype="<f8").tofile(out)
        if args.solid:
            with (args.output / "solid.bin").open("wb") as out:
                count = 0
                for plane in planes(h5[args.solid], args, mask=True):
                    if plane.shape != (ny, nx) or not np.isfinite(plane).all():
                        raise ValueError("Invalid solid mask shape or values")
                    if not np.isin(plane, [args.solid_value, args.air_value]).all():
                        raise ValueError("Mask contains values other than explicit air/solid values")
                    np.asarray(plane == args.solid_value, dtype=np.uint8).tofile(out)
                    count += 1
                if count != nz:
                    raise ValueError("Mask z extent differs")
        else:
            with (args.output / "solid.bin").open("wb") as out:
                for _ in range(nz):
                    np.zeros((ny, nx), dtype=np.uint8).tofile(out)
        mask = np.memmap(args.output / "solid.bin", dtype="u1", mode="r", shape=shape)
        for name in "uvw":
            wind = np.memmap(args.output / f"{name}.bin", dtype="<f8", mode="r+", shape=shape)
            for z in range(nz):
                if not np.isfinite(wind[z][mask[z] == 0]).all():
                    raise ValueError("Nonfinite wind in air")
                wind[z][mask[z] == 1] = 0
            wind.flush()
        geometry = dict(nx=nx, ny=ny, nz=nz, dx=args.dx, dy=args.dy, dz=args.dz,
                        origin_x=args.origin[0], origin_y=args.origin[1], origin_z=args.origin[2])
        (args.output / "case.cfg").write_text("".join(f"{k}={v}\n" for k, v in geometry.items()))
        (args.output / "provenance.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
        checksums = []
        for name in ("case.cfg", "u.bin", "v.bin", "w.bin", "solid.bin"):
            digest = hashlib.sha256()
            with (args.output / name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            checksums.append(f"{digest.hexdigest()}  {name}\n")
        (args.output / "input.sha256").write_text("".join(checksums))
        (args.output / "INCOMPLETE").unlink()
        print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--u"); p.add_argument("--v"); p.add_argument("--w")
    p.add_argument("--velocity")
    p.add_argument("--component-axis", type=int)
    p.add_argument("--components", nargs=3, type=int, default=[0, 1, 2])
    p.add_argument("--case-index", type=int)
    p.add_argument("--axis-order", required=True)
    p.add_argument("--solid")
    p.add_argument("--solid-value", type=float, default=1)
    p.add_argument("--air-value", type=float, default=0)
    p.add_argument("--all-fluid", action="store_true")
    for axis in "xyz":
        p.add_argument(f"--d{axis}", type=float, required=True)
    p.add_argument("--origin", nargs=3, type=float, default=[0, 0, 0])
    a = p.parse_args()
    if a.solid_value == a.air_value:
        p.error("Air and solid values must differ")
    prepare(a)
