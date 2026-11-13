"""Render conservative grid/particle aggregation with shared colour scales."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('result',type=Path);p.add_argument('--output',type=Path);p.add_argument('--factor',type=int,default=8);a=p.parse_args()
 out=a.output or a.result/'figures';out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((a.result/'aggregation.json').read_text());cfg=meta['config'];data=np.load(a.result/'column-maps.npz');times=meta['times'];f=a.factor
 if f not in meta['factors'] or f==1:raise ValueError('Select one of the saved coarse bin factors')
 with (a.result/'metrics.csv').open() as stream:records=list(csv.DictReader(stream))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 edges={d:float(cfg['origin_'+d])+np.r_[np.arange(0,int(cfg['n'+d]),f),int(cfg['n'+d])]*float(cfg['d'+d]) for d in 'xy'}
 area=np.diff(edges['y'])[:,None]*np.diff(edges['x'])[None,:]
 grids=[data[f'grid_t{i}_f{f}']/area for i in range(len(times))];parts=[data[f'particles_t{i}_f{f}']/area for i in range(len(times))]
 vmax=max(float(v.max()) for v in grids+parts);norm=LogNorm(vmin=vmax*1e-5,vmax=vmax)
 delta=max(float(abs(b-g).max()) for g,b in zip(grids,parts));dnorm=SymLogNorm(linthresh=max(delta*1e-3,1e-30),vmin=-max(delta,1e-30),vmax=max(delta,1e-30),base=10)
 cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#eef2f6');cmap.set_under('#eef2f6')
 fig,axes=plt.subplots(len(times),3,figsize=(12,4*len(times)),layout='constrained',squeeze=False)
 for i,t in enumerate(times):
  for j,value in enumerate([grids[i],parts[i],parts[i]-grids[i]]):
   ax=axes[i,j];im=ax.pcolormesh(edges['x'],edges['y'],np.ma.masked_less_equal(value,0) if j<2 else value,cmap=cmap if j<2 else 'RdBu_r',norm=norm if j<2 else dnorm,shading='flat',rasterized=True)
   ax.contour(data['x'],data['y'],data['footprint'],levels=[.5],colors='#64748b',linewidths=.3)
   ax.set(aspect='equal',xlabel='x (m)',ylabel='y (m)',title=['Finite volume','Particles: 3-seed mean','Particles minus grid'][j]+f' | t={t:g} s')
   if j==0:concentration=im
   if j==2:difference=im
 fig.colorbar(concentration,ax=axes[:,:2].ravel().tolist(),shrink=.8,label='Column mass density (mass / m²); common log scale')
 fig.colorbar(difference,ax=axes[:,2].tolist(),shrink=.8,label='Signed difference (mass / m²); symmetric log scale')
 fig.suptitle(f'Same spatial bins: {f*float(cfg["dx"]):.1f} × {f*float(cfg["dy"]):.1f} m; integrated over height',fontsize=15)
 fig.supxlabel('Grey outlines: projected solid footprint. Exact partial edge areas retained. No smoothing or per-panel renormalization.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(out/('aggregate_maps.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(2,len(times),figsize=(12,7),layout='constrained',squeeze=False)
 for row,metric in enumerate(['normalized_l1','survivor_total_variation']):
  for col,t in enumerate(times):
   ax=axes[row,col]
   for name,color,style in [('seed12345','#9ca3af',':'),('seed12346','#9ca3af',':'),('seed12347','#9ca3af',':'),('ensemble-mean','black','-'),('double-N','#2563eb','--'),('half-dt','#d97706','--')]:
    r=sorted([v for v in records if v['mode']=='3d' and v['method']==name and float(v['time'])==t],key=lambda v:int(v['factor']))
    ax.plot([int(v['factor']) for v in r],[float(v[metric]) for v in r],style,marker='o',ms=3,color=color,label=name)
   ax.set(xscale='log',ylim=(0,1.02),xlabel='Cells per bin per axis',ylabel='Normalized mass L1' if row==0 else 'Survivor-shape total variation',title=f't = {t:g} s');ax.set_xticks(meta['factors'],[str(v) for v in meta['factors']]);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('3D aggregation: mass discrepancy and conditional spatial shape',fontsize=15)
 fig.supxlabel('Nested bins retain partial edges. Grouping cells reduces or preserves mass L1 by construction.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(out/('aggregate_metrics.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig);print('Saved spatial comparison figures:',out)

if __name__=='__main__':main()
