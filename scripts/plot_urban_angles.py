"""Three camera views of the same measured plume and original building geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from inspect_building_geometry import read_stl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--view', type=int, choices=[1,2,3])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
bundle = root / 'results/atlas-plume-views-1033416.venus01'
meta = json.loads((bundle / 'plume-sections.json').read_text())
data = np.load(bundle / 'plume-sections.npz')
cfg = meta['config']
x, y = data['x'], data['y']
index = meta['times'].index(210)
values = data[f'column{index}']
tri = read_stl(root / 'data/geometry/00-buildings.stl')
ex = [x[0]-float(cfg['dx'])/2, x[-1]+float(cfg['dx'])/2]
ey = [y[0]-float(cfg['dy'])/2, y[-1]+float(cfg['dy'])/2]
keep = (tri[:,:,0].max(1)>=ex[0]) & (tri[:,:,0].min(1)<=ex[1]) & (tri[:,:,1].max(1)>=ey[0]) & (tri[:,:,1].min(1)<=ey[1])
tri = tri[keep]
floor = float(tri[:,:,2].min())-1
vmax = float(values.max())
norm = LogNorm(vmin=vmax*1e-5, vmax=vmax)
cmap = plt.get_cmap('inferno').copy()
cmap.set_under('#f0f3f7')
xx, yy = np.meshgrid(x[::3], y[::3])
colors = cmap(norm(np.maximum(values[::3,::3], vmax*1e-7)))
plt.rcParams.update({'font.size':10, 'savefig.facecolor':'white'})
views = [('Opposite oblique view', 40, 115), ('Low oblique view', 22, 25), ('Near-overhead view', 80, -90)]
for number, (title, elev, azim) in enumerate(views, 1):
    if args.view is not None and args.view != number:
        continue
    fig = plt.figure(figsize=(9,5.3), layout='constrained')
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.plot_surface(xx, yy, np.full_like(xx, floor), facecolors=colors,
                    rcount=xx.shape[0], ccount=xx.shape[1], shade=False,
                    linewidth=0, rasterized=True, zorder=1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors='#64748b', edgecolors='none', rasterized=True, zorder=3))
    ax.set(xlim=ex, ylim=ey, zlim=(floor,45), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((ex[1]-ex[0],ey[1]-ey[0],6*(45-floor)))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_zticks([0,20,40])
    if elev >= 80:
        ax.set_zticks([])
        ax.set_zlabel('')
    ax.set_title(f'{title} | t = 210 s', fontsize=13)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, shrink=.65,
                 pad=.08, label='Column concentration (mass / m²)')
    fig.supxlabel(f'Camera elevation {elev}°, azimuth {azim}°. Height exaggerated 6×; colour is a ground projection.', fontsize=9)
    out = root / f'report/figures/urban_angle_{number}'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Saved',out.name,flush=True)
