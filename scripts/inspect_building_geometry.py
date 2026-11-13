"""Inspect STL coordinates and exact-weld topology before solid voxelization."""
import argparse
import json
from pathlib import Path
import re
import struct
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def read_stl(path):
    data=path.read_bytes()
    count=struct.unpack_from('<I',data,80)[0] if len(data)>=84 else 0
    if len(data)==84+50*count:
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        return np.frombuffer(data,dtype=dtype,offset=84,count=count)['vertices'].astype(float)
    rows=re.findall(rb'\bvertex\s+([^\r\n]+)',data)
    values=np.asarray([list(map(float,row.split())) for row in rows])
    if not rows or values.ndim!=2 or values.shape[1]!=3 or len(values)%3:
        raise ValueError(f'Unrecognized or incomplete STL: {path}')
    return values.reshape(-1,3,3)


def inspect(tri):
    if not len(tri) or not np.isfinite(tri).all():
        raise ValueError('Empty or nonfinite mesh')
    vertices,indices=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True)
    faces=indices.reshape(-1,3)
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges.sort(axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    degenerate=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0
    return dict(triangles=len(tri),vertices=len(vertices),bounds_min=vertices.min(axis=0).tolist(),
                bounds_max=vertices.max(axis=0).tolist(),boundary_edges=int(np.sum(counts==1)),
                nonmanifold_edges=int(np.sum(counts>2)),degenerate_triangles=int(degenerate.sum()),
                closed_edge_topology=bool(np.all(counts==2) and not degenerate.any()),
                note='Exact-coordinate welding; closed edges alone do not establish absence of self-intersections or correct coordinate alignment.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry',type=Path)
    p.add_argument('grid',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.grid,'r') as f:
        limits={c:[float(f[f'mesh/{c}_coords'][0]),float(f[f'mesh/{c}_coords'][-1])] for c in 'xyz'}
    report=dict(grid_coordinate_ranges=limits,meshes=[])
    files=sorted(a.geometry.glob('*.stl'))
    if not files: raise ValueError('No extracted STL files')
    for index,path in enumerate(files):
        tri=read_stl(path)
        item=dict(file=path.name,**inspect(tri))
        report['meshes'].append(item)
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        step=max(1,int(np.ceil(len(tri)/50000)))
        for ax,(d0,d1) in zip(axes,[(0,1),(0,2)]):
            ax.add_collection(PolyCollection(tri[::step,:][:,:,[d0,d1]],facecolors='#53779b',edgecolors='none',alpha=.5))
            ax.autoscale_view()
            c0,c1='xyz'[d0],'xyz'[d1]
            lo,hi=limits[c0]; bottom,top=limits[c1]
            ax.plot([lo,hi,hi,lo,lo],[bottom,bottom,top,top,bottom],color='red',label='HDF5 sample extent')
            ax.set(xlabel=c0,ylabel=c1,title=f'{c0}–{c1} projection; every {step}th triangle')
            ax.legend()
        axes[0].set_aspect('equal',adjustable='datalim')
        fig.suptitle(path.name)
        fig.savefig(a.output/f'stl-projection-{index}.png',dpi=160)
        plt.close(fig)
    (a.output/'geometry-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
