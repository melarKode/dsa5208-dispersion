"""Choose a clear upstream release using measured local horizontal wind."""
import argparse
import json
from pathlib import Path
import numpy as np
from results import config


def select(case):
    c = config(case / 'case.cfg')
    xyz = [float(c['origin_'+d]) + (np.arange(int(c['n'+d]))+.5)*float(c['d'+d]) for d in 'xyz']
    shape = tuple(len(v) for v in xyz[::-1])
    mask = np.memmap(case/'solid.bin', dtype='u1', mode='r', shape=shape)
    wind = [np.memmap(case/(d+'.bin'), dtype='<f8', mode='r', shape=shape) for d in 'uvw']
    candidates = []
    for y in [550., 525., 500.]:
        for x in np.arange(-400., 501., 25.):
            centre = [float(x), y, 12.]
            if any(p-9 < a[0] or p+9 > a[-1] for p,a in zip(centre,xyz)):
                continue
            indices = [np.flatnonzero(abs(a-p)<=9) for p,a in zip(centre,xyz)]
            ix = np.ix_(*indices[::-1])
            if mask[ix].any():
                continue
            samples = [np.asarray(w[ix]) for w in wind]
            if not all(np.isfinite(w).all() for w in samples):
                continue
            weights = [np.exp(-.5*((a[i]-p)/3)**2) for a,i,p in zip(xyz,indices,centre)]
            weight = weights[2][:,None,None]*weights[1][None,:,None]*weights[0][None,None,:]
            mean = [float((w*weight).sum()/weight.sum()) for w in samples]
            horizontal = float(np.hypot(mean[0],mean[1]))
            p10 = float(np.percentile(np.hypot(samples[0],samples[1]),10))
            if horizontal < 3 or p10 < 2 or mean[1] > -2:
                continue
            projected_x = x + mean[0]/mean[1]*(300-y)
            if not -400 <= projected_x <= 500:
                continue
            candidates.append(dict(xyz=centre,mean_uvw=mean,horizontal_speed=horizontal,
                                   core_horizontal_p10=p10,projected_x_at_y300=float(projected_x)))
    if not candidates:
        raise ValueError('No candidate meets source criteria; no simulation submitted with an unsuitable source')
    chosen = min(candidates,key=lambda c:(c['xyz'][0]-150)**2+(c['xyz'][1]-550)**2)
    return dict(selected=chosen,eligible_candidates=len(candidates),source_sigma=3,
                criteria='Clear 3-sigma core; mean horizontal speed >=3 m/s; core 10th percentile >=2 m/s; mean v <=-2 m/s.',
                selection='Closest eligible candidate to the pilot source (150,550,12).',
                trajectory_note='Local straight-line heading towards the building area is a screening heuristic, not a predicted trajectory.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();report=select(a.case)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'source-selection.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'source.cfg').write_text(''.join(f'source_{d}={v}\n' for d,v in zip('xyz',report['selected']['xyz'])))
    print(json.dumps(report,indent=2))
