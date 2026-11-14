"""Plot exported simulation sections with building masks and original STL geometry."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from inspect_building_geometry import read_stl


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,default=Path('report/figures'));p.add_argument('--source-selection',type=Path,default=Path('results/atlas-urban-main-1032157.venus01/review/source-selection.json'));p.add_argument('--stl',type=Path,default=Path('data/geometry/00-buildings.stl'));p.add_argument('--stl-only',action='store_true');p.add_argument('--sections-only',action='store_true');a=p.parse_args()
 m=json.loads((a.bundle/'plume-sections.json').read_text());d=np.load(a.bundle/'plume-sections.npz');a.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 cfg=m['config'];times=m['times'];x,y,z=(d[k] for k in 'xyz');sx,sy,sz=m['section_xyz']
 wind=json.loads(a.source_selection.read_text())['selected']
 assert np.allclose(wind['xyz'],[float(cfg['source_'+k]) for k in 'xyz'])
 uv=np.array(wind['mean_uvw'][:2]);speed=float(np.linalg.norm(uv));direction=uv/speed
 edges=lambda arr,delta:[arr[0]-delta/2,arr[-1]+delta/2]
 ex=edges(x,float(cfg['dx']));ey=edges(y,float(cfg['dy']));ez=edges(z,float(cfg['dz']))
 cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#f0f3f7');cmap.set_under('#f0f3f7')
 def save(fig,name):
  fig.savefig(a.output/(name+'.pdf'),bbox_inches='tight');fig.savefig(a.output/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 for kind,extent,mask,labels,cut in [('column',ex+ey,d['footprint'],('x (m)','y (m)'),'Height-integrated plume'),('xy',ex+ey,d['solid_xy'],('x (m)','y (m)'),f'Horizontal section: z = {sz:.2f} m'),('yz',ey+ez,d['solid_yz'],('y (m)','z (m)'),f'Vertical section: x = {sx:.2f} m'),('xz',ex+ez,d['solid_xz'],('x (m)','z (m)'),f'Vertical section: y = {sy:.2f} m')]:
  if a.stl_only:continue
  chosen=[i for i,t in enumerate(times) if t in [60,120,210,300,450,600]]
  vmax=max(float(d[f'{kind}{i}'].max()) for i in chosen)
  if vmax<=0:raise ValueError('No positive concentration in '+kind)
  norm=LogNorm(vmin=vmax*1e-5,vmax=vmax)
  if kind=='xz':chosen=[times.index(t) for t in [30,60,120]]
  fig,axes=plt.subplots(1 if kind=='xz' else 2,3,figsize=(13,3.5 if kind=='xz' else 8 if kind in ['column','xy'] else 5.8),layout='constrained')
  for ax,i in zip(axes.flat,chosen):
   values=d[f'{kind}{i}'];im=ax.imshow(np.ma.masked_less_equal(values,0),origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest',aspect='equal' if kind in ['column','xy'] else 'auto')
   if kind=='column':
    ax.contour(x,y,mask,levels=[.5],colors='#64748b',linewidths=.35)
   else:
    ax.imshow(np.ma.masked_equal(mask,0),origin='lower',extent=extent,cmap=ListedColormap(['#64748b']),vmin=0,vmax=1,interpolation='nearest',aspect='equal' if kind=='xy' else 'auto')
   ax.set(title=f't = {times[i]:g} s',xlabel=labels[0],ylabel=labels[1])
   if kind in ['xy','column']:
    release=np.array([float(cfg['source_x']),float(cfg['source_y'])])
    ax.plot(*release,marker='+',color='white',ms=11,mew=3,zorder=9)
    ax.plot(*release,marker='+',color='#0284c7',ms=10,mew=1.6,zorder=10)
    ax.annotate('',xy=release+145*direction,xytext=release,arrowprops=dict(arrowstyle='-|>',color='#0284c7',lw=1.6))
    if i==chosen[0]:ax.text(release[0]+45,release[1]-25,f'Local wind\n{speed:.2f} m/s',color='#0369a1',fontsize=8,va='top')
  units='mass / m²' if kind=='column' else 'mass / m³'
  fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,label=f'Concentration ({units}); shared log scale')
  fig.suptitle(cut+' — actual 600-second urban experiment',fontsize=15)
  caption='Grey outlines: projected simulated building footprint; pollutant may pass above roofs.' if kind=='column' else 'Grey: solid cells in this exact section. Vertical panels exaggerate height for readability.' if kind in ['yz','xz'] else 'Grey: solid cells in this exact section. Blue +: release location.'
  fig.supxlabel(caption+' Pale: below colour minimum or zero.'+(' Blue arrow: local mean wind at release; direction only.' if kind in ['xy','column'] else ''),fontsize=8)
  save(fig,'urban_plume_'+kind)
 if a.sections_only:
  print('Saved section plots to',a.output);return
 # Render original triangles beside a ground-projected column-density map.
 tri=read_stl(a.stl)
 keep=(tri[:,:,0].max(axis=1)>=ex[0])&(tri[:,:,0].min(axis=1)<=ex[1])&(tri[:,:,1].max(axis=1)>=ey[0])&(tri[:,:,1].min(axis=1)<=ey[1])
 tri=tri[keep]
 chosen=[times.index(t) for t in [120,210] if t in times]
 vmax=max(float(d[f'column{i}'].max()) for i in chosen);norm=LogNorm(vmin=vmax*1e-5,vmax=vmax)
 fig=plt.figure(figsize=(14,6),layout='constrained');axes=[]
 for j,i in enumerate(chosen):
  ax=fig.add_subplot(1,len(chosen),j+1,projection='3d',computed_zorder=False);axes.append(ax)
  xx,yy=np.meshgrid(x[::3],y[::3]);vv=d[f'column{i}'][::3,::3]
  colors=cmap(norm(np.maximum(vv,vmax*1e-7)))
  ax.plot_surface(xx,yy,np.full_like(xx,float(tri[:,:,2].min())-1),facecolors=colors,rcount=len(yy),ccount=xx.shape[1],shade=False,linewidth=0,rasterized=True,zorder=1)
  ax.add_collection3d(Poly3DCollection(tri,facecolors='#64748b',edgecolors='none',rasterized=True,zorder=3))
  ax.set(xlim=ex,ylim=ey,zlim=(float(tri[:,:,2].min())-1,ez[1]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f't = {times[i]:g} s')
  ax.grid(False);ax.set_zticks([0,20,40])
  ax.set_box_aspect((ex[1]-ex[0],ey[1]-ey[0],6*(ez[1]-ez[0])));ax.view_init(elev=40,azim=-65)
 fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.6,label='Column concentration (mass / m²)')
 fig.suptitle('Plume projected beneath the original STL buildings',fontsize=15)
 fig.supxlabel('Height-integrated concentration projected onto the ground; display samples every third cell. Building height exaggerated 6×.',fontsize=9)
 save(fig,'urban_plume_stl')
 print('Saved plume plots to',a.output)

if __name__=='__main__':main()
