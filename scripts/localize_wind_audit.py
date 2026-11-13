"""Locate wind-audit extremes and plot wind/mask overlays at the worst layers."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_case import audit
from results import config

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=config(a.case/'case.cfg')
nx,ny,nz=[int(c[f'n{d}']) for d in 'xyz']
x,y,z=[float(c[f'origin_{d}'])+(np.arange(n)+.5)*float(c[f'd{d}']) for d,n in zip('xyz',[nx,ny,nz])]
wind=[np.memmap(a.case/f'{d}.bin',dtype='<f8',mode='r',shape=(nz,ny,nx)) for d in 'uvw']
mask=np.memmap(a.case/'solid.bin',dtype='u1',mode='r',shape=(nz,ny,nx))
inner=(x[None,:]>=-700)&(x[None,:]<=800)&(y[:,None]>=-650)&(y[:,None]<=650)
regional=dict(count=0,sum_abs=0.,sum_squares=0.,maximum=0.)
layers=[];worst=[]
def collect(k,div,fluid):
    values=div[fluid]
    maximum=float(np.abs(values).max(initial=0))
    layers.append(dict(k=k,z=float(z[k]),max_abs=maximum,rms=float(np.sqrt(np.mean(values**2))) if len(values) else None))
    if z[k]>=2.4:
        v=div[fluid & inner]
        regional['count']+=len(v);regional['sum_abs']+=float(np.abs(v).sum())
        regional['sum_squares']+=float(np.square(v).sum());regional['maximum']=max(regional['maximum'],float(np.abs(v).max(initial=0)))
    worst.append((maximum,k,div.copy()))
    worst.sort(key=lambda q:q[0],reverse=True)
    del worst[3:]
report=audit(a.case,['open','open','closed'],collect)
report['layers']=layers
n=regional['count']
report['interior_region']=dict(bounds_xyz=[[-700,800],[-650,650],[2.4,float(z[-1])]],air_cells=n,
 mean_abs_divergence=regional['sum_abs']/n if n else None,rms_divergence=np.sqrt(regional['sum_squares']/n) if n else None,
 max_abs_divergence=regional['maximum'],note='Subset of the original audit, not a newly cropped-domain simulation; excludes circular exterior approximately, not a verified CFD-validity mask.')
for _,k,div in worst:
    fluid=mask[k]==0
    mag=np.sqrt(sum(v[k]**2 for v in wind))
    jj,ii=np.unravel_index(np.argmax(np.where(fluid,np.abs(div),-1)),div.shape)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    radius=70
    ix=slice(max(0,ii-radius),min(nx,ii+radius+1));iy=slice(max(0,jj-radius),min(ny,jj+radius+1))
    for ax,zoom in zip(axes,[False,True,True]):
        xx,yy=(x[ix],y[iy]) if zoom else (x,y)
        s=(iy,ix) if zoom else (slice(None),slice(None))
        field=div[s] if ax is axes[2] else mag[s]
        if ax is axes[2]:
            lim=max(float(np.abs(field).max()),1e-12)
            image=ax.pcolormesh(xx,yy,np.ma.array(field,mask=~fluid[s]),cmap='RdBu_r',vmin=-lim,vmax=lim,shading='nearest')
        else:
            image=ax.pcolormesh(xx,yy,np.ma.array(field,mask=~fluid[s]),cmap='viridis',shading='nearest')
        ax.set_facecolor('#30343b')
        if np.any(mask[k][s]) and not np.all(mask[k][s]): ax.contour(xx,yy,mask[k][s],levels=[.5],colors='red',linewidths=.5)
        if ax is not axes[2]:
            stride=max(1,len(xx)//25)
            ax.quiver(xx[::stride],yy[::stride],np.ma.array(wind[0][k][s],mask=~fluid[s])[::stride,::stride],np.ma.array(wind[1][k][s],mask=~fluid[s])[::stride,::stride],color='white',angles='xy')
        ax.plot(x[ii],y[jj],'x',color='magenta',markersize=8)
        ax.set(aspect='equal',xlabel='x',ylabel='y')
        fig.colorbar(image,ax=ax,shrink=.7,label='Divergence (source units)' if ax is axes[2] else 'Speed (source units)')
    axes[0].set_title('Whole layer: speed + STL mask')
    axes[1].set_title('Worst-cell neighbourhood: speed')
    axes[2].set_title('Worst-cell neighbourhood: divergence')
    fig.suptitle(f'z={z[k]:.3f}; red = building boundary; magenta = largest |divergence|; provisional component mapping')
    fig.savefig(a.output/f'wind-overlay-layer-{k:02d}.png',dpi=150);plt.close(fig)
(a.output/'localized-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
