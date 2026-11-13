"""Inspect HDF5 metadata and bounded samples; optionally stream full statistics."""
import argparse
import itertools
import json
from pathlib import Path
import h5py
import numpy as np


def blocks(shape, width=32):
    for start in itertools.product(*(range(0, n, width) for n in shape)):
        yield tuple(slice(s, min(s + width, n)) for s, n in zip(start, shape))


def inspect(path, full=False):
    report = {"file": str(path), "bytes": path.stat().st_size, "objects": {}}
    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            item = {"attributes": {k: repr(v)[:1000] for k, v in obj.attrs.items()}}
            report["objects"]["/" + name] = item
            if not isinstance(obj, h5py.Dataset):
                return
            item.update(shape=obj.shape, dtype=str(obj.dtype), chunks=obj.chunks,
                        compression=obj.compression)
            if obj.shape is None or not obj.size or not np.issubdtype(obj.dtype, np.number):
                return
            selections = blocks(obj.shape) if full else [tuple(slice(0, min(n, 8)) for n in obj.shape)]
            count = finite_count = zeros = 0
            low, high, total = np.inf, -np.inf, 0.0
            for selection in selections:
                values = np.asarray(obj[selection] if obj.ndim else obj[()])
                valid = values[np.isfinite(values)]
                count += values.size
                finite_count += valid.size
                zeros += np.count_nonzero(valid == 0)
                if valid.size:
                    low, high = min(low, float(valid.min())), max(high, float(valid.max()))
                    total += float(valid.sum(dtype=np.float64))
            item["statistics"] = dict(scope="full" if full else "leading block only",
                                      count=count, finite=finite_count, zeros=int(zeros),
                                      minimum=low if finite_count else None,
                                      maximum=high if finite_count else None,
                                      mean=total / finite_count if finite_count else None)
            if obj.ndim == 1:
                item["first_values"] = obj[:8].tolist()
                item["last_values"] = obj[-8:].tolist()
        visit("", handle)
        handle.visititems(visit)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = json.dumps(inspect(args.file, args.full), indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(result + "\n")
        print(result)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Inspection failed: {error}. Check download size and checksum first.")
