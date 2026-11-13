"""Read serial or partitioned grid outputs without assuming MPI rank ordering."""
import csv
from pathlib import Path
import numpy as np


def config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def field(directory, index=-1):
    directory = Path(directory)
    native = directory / "effective.cfg"
    cfg = config(native if native.exists() else directory / "effective_config.txt")
    if not native.exists():
        cfg["solver_family"] = "particle"
    nx, ny, nz = (int(cfg[f"n{axis}"]) for axis in "xyz")
    records = rows(directory / "snapshots.csv")
    record = records[index]
    snapshot_index = record.get("index", record.get("snapshot_index"))
    if snapshot_index is None:
        raise ValueError("Missing snapshot index")
    prefix = f"concentration_{int(snapshot_index):06d}"
    fmt = record.get("format", "csv")
    if cfg.get("write_concentration") == "false":
        raise ValueError("Run has no concentration snapshots")
    if fmt not in ("binary", "csv"):
        raise ValueError("Run has no concentration snapshots")
    result = np.empty((nz, ny, nx), dtype=np.float64)
    covered = np.zeros((ny, nx), dtype=bool)
    ranks = int(record.get("ranks", 1))
    for rank in range(ranks):
        domain_path = directory / f"domain_rank{rank}.csv"
        if domain_path.exists():
            domain = rows(domain_path)[0]
        elif ranks == 1:
            domain = dict(x0=0, y0=0, nx=nx, ny=ny, nz=nz)
        else:
            raise ValueError("Missing rank domain description")
        x0, y0, lx, ly, lz = (int(domain[k]) for k in ("x0", "y0", "nx", "ny", "nz"))
        if lz != nz or lx < 1 or ly < 1 or x0 < 0 or y0 < 0 or x0 + lx > nx or y0 + ly > ny:
            raise ValueError("Invalid partition extent")
        if covered[y0:y0 + ly, x0:x0 + lx].any():
            raise ValueError("Overlapping rank partitions")
        suffix = f"_rank{rank}" if ranks > 1 else ""
        path = directory / (prefix + suffix + (".bin" if fmt == "binary" else ".csv"))
        if fmt == "binary":
            data = np.fromfile(path, dtype="<f8").reshape(lz, ly, lx)
        else:
            table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
            order = np.argsort(table["cell_id"])
            table = table[order]
            expected = ((np.arange(nz)[:, None, None] * ny + np.arange(y0, y0 + ly)[None, :, None]) * nx + np.arange(x0, x0 + lx)[None, None, :]).ravel()
            if not np.array_equal(table["cell_id"], expected):
                raise ValueError("Missing, repeated, or invalid cell IDs")
            data = table["concentration"].reshape(lz, ly, lx)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite concentration")
        result[:, y0:y0 + ly, x0:x0 + lx] = data
        covered[y0:y0 + ly, x0:x0 + lx] = True
    if not covered.all():
        raise ValueError("Incomplete partition coverage")
    return cfg, float(record["time"]), result
