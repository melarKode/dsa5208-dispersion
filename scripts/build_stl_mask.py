"""Classify cell centres using horizontal STL contours and two independent ray directions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from inspect_building_geometry import read_stl
from plot_stl_views import section


def scan(segments, x, y):
    result=np.zeros((len(y),len(x)),dtype=bool)
    odd=[]
    for j,yy in enumerate(y):
        a,b=segments[:,0],segments[:,1]
        crossing=((a[:,1]<=yy)&(b[:,1]>yy))|((b[:,1]<=yy)&(a[:,1]>yy))
        aa,bb=a[crossing],b[crossing]
        hits=np.sort(aa[:,0]+(yy-aa[:,1])*(bb[:,0]-aa[:,0])/(bb[:,1]-aa[:,1]))
        if len(hits)%2: odd.append(j)
        result[j]=np.searchsorted(hits,x,side='right')%2==1
    return result,odd


def raster(tri,x,y,z):
    masks=[]; layers=[]
    for k,height in enumerate(z):
        lines=section(tri,height)
        mx,oddx=scan(lines,x,y)
        my,oddy=scan(lines[:,:,[1,0]],y,x)
        disagreement=int(np.count_nonzero(mx!=my.T))
        layers.append(dict(layer=k,z=float(height),segments=len(lines),solid_cells=int(mx.sum()),
                           odd_x_rows=len(oddx),odd_y_columns=len(oddy),ray_disagreements=disagreement))
        if oddx or oddy or disagreement:
            raise ValueError(f'Unreliable slice; no mask accepted: {layers[-1]}')
        masks.append(mx.astype('u1'))
    return np.stack(masks),layers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stl',type=Path);p.add_argument('inspection',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads(a.inspection.read_text())['objects']
    coords=[]
    for axis in 'xyz':
        v=meta[f'/mesh/{axis}_coords']
        coords.append(np.linspace(v['first_values'][0],v['last_values'][-1],v['shape'][0]))
    mask,layers=raster(read_stl(a.stl),*coords)
    np.save(a.output/'building-mask-zyx.npy',mask)
    report=dict(source_sha256=hashlib.sha256(a.stl.read_bytes()).hexdigest(),shape_zyx=list(mask.shape),
                method='Even-odd horizontal contour filling; x/y ray classifications must agree at every cell centre',
                status='Candidate building-only mask; excludes terrain and outside-domain classification. Overlapping solids and geometry alignment require review.',
                solid_cells=int(mask.sum()),layers=layers)
    (a.output/'mask-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
