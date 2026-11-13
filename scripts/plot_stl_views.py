"""Render STL projections and exact plane intersections without repairing the mesh."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from inspect_building_geometry import read_stl


def section(tri, height):
    chosen=tri[(tri[:,:,2].min(axis=1)<height)&(tri[:,:,2].max(axis=1)>height)]
    points=np.full((len(chosen),3,2),np.nan)
    for edge,(a,b) in enumerate([(0,1),(1,2),(2,0)]):
        za,zb=chosen[:,a,2],chosen[:,b,2]
        crosses=((za<=height)&(zb>height))|((zb<=height)&(za>height))
        fraction=(height-za[crosses])/(zb[crosses]-za[crosses])
        points[crosses,edge]=chosen[crosses,a,:2]+fraction[:,None]*(chosen[crosses,b,:2]-chosen[crosses,a,:2])
    valid=np.isfinite(points[:,:,0])
    keep=valid.sum(axis=1)==2
    return points[keep][valid[keep]].reshape(-1,2,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stl',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    t=read_stl(a.stl)
    lo=t.min(axis=(0,1)); hi=t.max(axis=(0,1))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    norm=plt.Normalize(lo[2],hi[2]); cmap=plt.get_cmap('viridis')
    # Paint low triangles first so roofs remain visible in the top projection.
    order=np.argsort(t[:,:,2].mean(axis=1))
    fig,ax=plt.subplots(figsize=(12,10),constrained_layout=True)
    poly=PolyCollection(t[order,:,:2],array=t[order,:,2].mean(axis=1),cmap=cmap,norm=norm,edgecolors='none',rasterized=True)
    ax.add_collection(poly); ax.autoscale_view(); ax.set_aspect('equal')
    ax.set(title='Urban STL — top-down surface projection',xlabel='x (source coordinates)',ylabel='y (source coordinates)')
    fig.colorbar(poly,ax=ax,label='Surface z (source coordinates)',shrink=.8)
    fig.text(.5,.005,'All 910,412 source triangles • projection, not a validated solid-cell mask',ha='center',fontsize=10)
    fig.savefig(a.output/'stl-top-plan.png',dpi=170); plt.close(fig)
    print('Top plan saved',flush=True)
    fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
    for ax,height in zip(axes.flat,[0.5,2.5,10.5,20.5]):
        seg=section(t,height)
        ax.add_collection(LineCollection(seg,colors='#155e75',linewidths=.45))
        ax.set(xlim=(lo[0]-15,hi[0]+15),ylim=(lo[1]-15,hi[1]+15),aspect='equal',xlabel='x',ylabel='y',title=f'Horizontal cut at z = {height:g} | {len(seg):,} segments')
    fig.suptitle('Floor-plan-style slices — actual triangle/plane intersections\nUnfilled outlines; z is not verified height above local ground',fontsize=15)
    fig.savefig(a.output/'stl-horizontal-sections.png',dpi=170); plt.close(fig)
    print('Sections saved',flush=True)
    q=np.round(t/2)*2
    keep=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)>0
    q=q[keep]
    fig=plt.figure(figsize=(15,8),layout='constrained')
    ax=fig.add_subplot(111,projection='3d')
    colors=cmap(norm(q[:,:,2].mean(axis=1)))
    ax.add_collection3d(Poly3DCollection(q,facecolors=colors,edgecolors='none',zsort='average',rasterized=True))
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2]-2,hi[2]+2),xlabel='x',ylabel='y',zlabel='z')
    ax.set_box_aspect((hi[0]-lo[0],hi[1]-lo[1],6*(hi[2]-lo[2])))
    ax.view_init(elev=35,azim=-65)
    ax.set_title('Urban geometry in 3D — vertical scale exaggerated 6×\nDisplay mesh quantized to 2 coordinate units; original STL unchanged',fontsize=15)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.55,pad=.08,label='Surface z')
    fig.savefig(a.output/'stl-3d.png',dpi=140); plt.close(fig)
    print('3D saved',flush=True)
    fig,axes=plt.subplots(2,1,figsize=(14,6),constrained_layout=True)
    for ax,d,label in zip(axes,[0,1],['x','y']):
        ax.add_collection(PolyCollection(t[:,:,[d,2]],facecolors='#245b78',edgecolors='none',rasterized=True))
        ax.set(xlim=(lo[d]-10,hi[d]+10),ylim=(lo[2]-1,hi[2]+1),xlabel=f'{label} (source coordinates)',ylabel='z',title=f'{label}–z elevation: all surfaces projected, horizontal depth collapsed')
    fig.suptitle('Side elevations — unequal horizontal and vertical scales',fontsize=15)
    fig.savefig(a.output/'stl-elevations.png',dpi=160); plt.close(fig)
    print('Elevations saved',flush=True)


if __name__=='__main__': main()
