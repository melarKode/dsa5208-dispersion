"""Verify coordinate storage axes and summarize the supplied urban mask."""
import argparse
import itertools
import json
from pathlib import Path
import h5py
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('file', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with h5py.File(a.file, 'r') as f:
    coords = [np.asarray(f[f'mesh/{c}_coords']) for c in 'xyz']
    mesh = [f[f'mesh/{c}'] for c in 'XYZ']
    shape = mesh[0].shape
    candidates = []
    for order in itertools.permutations('xyz'):
        valid = True
        for physical, c in enumerate('xyz'):
            axis = order.index(c)
            if shape[axis] != len(coords[physical]):
                valid = False
                break
            for anchor in (0, 1):
                index = [0 if anchor == 0 else n - 1 for n in shape]
                index[axis] = slice(None)
                if not np.allclose(mesh[physical][tuple(index)], coords[physical], rtol=1e-10, atol=1e-8):
                    valid = False
        if valid:
            candidates.append(''.join(order))
    if len(candidates) != 1:
        raise ValueError(f'Coordinate axis mapping ambiguous or invalid: {candidates}')
    order = candidates[0]
    report = {'axis_order': order, 'axes': {}, 'layers': [],
              'note': 'Coordinate samples are verified; treating sample positions as finite-volume cell centres is a modelling choice. Mask geometry and velocity component conventions still require source review.'}
    for c, values in zip('xyz', coords):
        step = float((values[-1] - values[0]) / (len(values) - 1))
        if step <= 0 or not np.isfinite(values).all() or not np.allclose(np.diff(values), step, rtol=1e-8, atol=1e-8):
            raise ValueError(f'Nonuniform or invalid {c} coordinates')
        report['axes'][c] = dict(count=len(values), first=float(values[0]), last=float(values[-1]), spacing=step,
                                 proposed_cell_origin=float(values[0] - step / 2))
    mask = f['building_mask/mask']
    total = 0
    for z, height in enumerate(coords[2]):
        index = [slice(None)] * 3
        index[order.index('z')] = z
        layer = np.asarray(mask[tuple(index)])
        if not np.isin(layer, [0, 1]).all():
            raise ValueError('Mask is not binary')
        solid = int(np.count_nonzero(layer))
        total += solid
        report['layers'].append(dict(z=z, coordinate=float(height), solid_cells=solid, solid_fraction=solid/layer.size))
    report['solid_cells'] = total
    report['air_cells'] = int(mask.size - total)
    report['mask_count_matches_attribute'] = total == int(f['building_mask'].attrs['building_cells'])
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
