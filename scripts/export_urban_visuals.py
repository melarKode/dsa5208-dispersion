"""Extract actual plume sections and column densities for portable plotting."""
import argparse,json
from pathlib import Path
import numpy as np
from results import config,field,rows
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path);p.add_argument('output',type=Path)
p.add_argument('--case',type=Path);p.add_argument('--times',type=float,nargs='+',default=[30,60,120,210,300,450,600])
a=p.parse_args();cfg=config(a.run/'effective.cfg');case=a.case or Path(cfg['case_dir'])
shape=tuple(int(cfg['n'+d]) for d in 'zyx')
solid=np.fromfile(case/'solid.bin',dtype='u1').reshape(shape)
coords={d:float(cfg['origin_'+d])+(np.arange(int(cfg['n'+d]))+.5)*float(cfg['d'+d]) for d in 'xyz'}
ix = int(np.argmin(np.abs(coords['x'] - float(cfg['source_x']))))
iy = int(np.argmin(np.abs(coords['y'] - 300)))
iz = int(np.argmin(np.abs(coords['z'] - float(cfg['source_z']))))
records=rows(a.run/'snapshots.csv');payload=dict(**coords,solid_xy=solid[iz],solid_yz=solid[:,:,ix],solid_xz=solid[:,iy,:],footprint=solid.any(axis=0))
metadata=dict(run=str(a.run),case=str(case),config=cfg,section_xyz=[float(coords['x'][ix]),float(coords['y'][iy]),float(coords['z'][iz])],times=[],note='Exact saved cell sections; column concentration integrated across z. No interpolation or smoothing. Coordinates are source coordinates, not terrain-relative heights.')
for j,t in enumerate(a.times):
 matches=[i for i,r in enumerate(records) if abs(float(r['time'])-t)<1e-8]
 if len(matches)!=1:raise ValueError(f'No unique saved time {t}')
 _,actual,c=field(a.run,matches[0]);assert c.shape==shape
 payload[f'xy{j}']=c[iz].copy();payload[f'yz{j}']=c[:,:,ix].copy();payload[f'xz{j}']=c[:,iy,:].copy();payload[f'column{j}']=c.sum(axis=0)*float(cfg['dz'])
 metadata['times'].append(actual)
 print(f'Extracted t={actual:g}',flush=True)
a.output.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output/'plume-sections.npz',**payload)
(a.output/'plume-sections.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'SUCCESS: saved actual plume sections to {a.output}',flush=True)
