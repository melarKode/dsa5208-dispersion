"""Estimate integrated concentration from snapshots using the trapezoidal rule."""
import argparse
import json
from pathlib import Path
import numpy as np
from results import field, rows

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
if a.output.exists():
    raise SystemExit("Output exists; choose a new filename")
records = rows(a.run / "snapshots.csv")
cfg, previous_time, previous = field(a.run, 0)
exposure = np.zeros_like(previous)
for index in range(1, len(records)):
    _, time, current = field(a.run, index)
    if time <= previous_time:
        raise ValueError("Snapshot times must increase")
    exposure += .5 * (previous + current) * (time - previous_time)
    previous_time, previous = time, current
with a.output.open("wb") as stream:
    np.save(stream, exposure)
a.output.with_suffix(".json").write_text(json.dumps(dict(units="mass*s/m^3", shape_zyx=exposure.shape,
    note="Snapshot trapezoidal estimate; refine output_interval to check temporal accuracy."), indent=2))
print(a.output)
