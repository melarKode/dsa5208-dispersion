"""Plot raw urban slices with supplied solids masked; xyz storage is required."""
import argparse
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('file', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
with h5py.File(a.file, 'r') as f:
    x, y, z = [f[f'mesh/{c}_coords'][:] for c in 'xyz']
    mask = f['building_mask/mask']
    wind = f['cases/4_SanGeronimo_inletOutlet_vegetation_u2_alpha315/velocity']
    levels = sorted(set(int(np.argmin(abs(z-height))) for height in [2,20,40]))
    fig, axes = plt.subplots(1, len(levels), figsize=(5*len(levels),5), squeeze=False, constrained_layout=True)
    for ax,k in zip(axes[0],levels):
        solid = mask[:,:,k].T.astype(bool)
        ax.pcolormesh(x,y,solid,cmap='gray_r',vmin=0,vmax=1,shading='nearest',rasterized=True)
        ax.set(title=f'Supplied solid mask | z={z[k]:.2f}',xlabel='x (source coordinates)',ylabel='y (source coordinates)',aspect='equal')
    fig.savefig(a.output/'urban-mask-horizontal.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(levels), figsize=(5*len(levels),5), squeeze=False, constrained_layout=True)
    for ax,k in zip(axes[0],levels):
        v = wind[:,:,k,:].transpose(1,0,2)
        solid = mask[:,:,k].T.astype(bool)
        speed = np.ma.array(np.linalg.norm(v,axis=-1),mask=solid)
        ax.set_facecolor('#30343b')
        im=ax.pcolormesh(x,y,speed,shading='nearest',cmap='viridis',rasterized=True)
        stride=max(1,len(x)//25)
        u=np.ma.array(v[:,:,0],mask=solid)[::stride,::stride]
        vv=np.ma.array(v[:,:,1],mask=solid)[::stride,::stride]
        ax.quiver(x[::stride],y[::stride],u,vv,color='white',angles='xy')
        ax.set(title=f'Wind | z={z[k]:.2f}',xlabel='x (source coordinates)',ylabel='y (source coordinates)',aspect='equal')
        fig.colorbar(im,ax=ax,label='3D speed (source velocity units)',shrink=.75)
    fig.suptitle('Raw wind; arrows show horizontal components; supplied solids are grey\nComponent mapping (0,1,2)=(u,v,w) is provisional; panels use individual colour scales')
    fig.savefig(a.output/'urban-wind-horizontal.png',dpi=160)
    plt.close(fig)
    j=len(y)//2
    v=wind[:,j,:,:].transpose(1,0,2)
    solid=mask[:,j,:].T.astype(bool)
    fig,axes=plt.subplots(2,1,figsize=(13,6),constrained_layout=True)
    axes[0].pcolormesh(x,z,solid,cmap='gray_r',vmin=0,vmax=1,shading='nearest')
    axes[0].set_title(f'Supplied solid mask, x–z slice at y={y[j]:.2f}')
    axes[1].set_facecolor('#30343b')
    im=axes[1].pcolormesh(x,z,np.ma.array(np.linalg.norm(v,axis=-1),mask=solid),cmap='viridis',shading='nearest')
    axes[1].set_title('Raw wind speed; vertical dimension exaggerated for visibility')
    fig.colorbar(im,ax=axes[1],label='Speed (source velocity units)')
    for ax in axes: ax.set(xlabel='x (source coordinates)',ylabel='z (source coordinates)')
    fig.savefig(a.output/'urban-vertical-section.png',dpi=160)
    plt.close(fig)
print(a.output)
