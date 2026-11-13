"""Run a matched uniform-wind comparison across independent particle seeds."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare import compare
from results import field


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-exe',type=Path,default=Path('build/serial/dispersion_serial'))
    p.add_argument('--particle-exe',type=Path,default=Path('lagrangian/build/lagrangian_solver'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args()
    if a.seeds<2: p.error('At least two seeds are required')
    a.output.mkdir(parents=True,exist_ok=False)
    grid=a.output/'grid'
    subprocess.run([str(a.grid_exe.resolve()),'configs/gaussian.cfg','dt=0.05',f'output_dir={grid}'],check=True)
    cfg,t,fg=field(grid); dv=np.prod([float(cfg['d'+d]) for d in 'xyz'])
    mean=np.zeros_like(fg);reports=[]
    for seed in range(12345,12345+a.seeds):
        out=a.output/f'particle-{seed}'
        subprocess.run([str(a.particle_exe.resolve()),'lagrangian/cases/example_gaussian_uniform.cfg',
                        '--output',str(out),'--set',f'seed={seed}'],check=True)
        reports.append(dict(seed=seed,**compare(grid,out)))
        _,_,fp=field(out);mean+=fp/a.seeds
    delta=fg-mean
    report=dict(case='uniform periodic Gaussian',time=t,particles_per_seed=100000,seeds=a.seeds,
                per_seed=reports,mean_per_seed_l1=float(np.mean([r['l1'] for r in reports])),
                ensemble_mean_l1=float(np.abs(delta).sum()*dv),
                ensemble_mean_mass=float(mean.sum()*dv),
                note='Differences include finite-volume discretization and particle sampling error; this is not an exact-equivalence test.')
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    dz=float(cfg['dz']);columns=[fg.sum(axis=0)*dz,mean.sum(axis=0)*dz]
    vmax=max(v.max() for v in columns)
    x0=float(cfg['origin_x']);y0=float(cfg['origin_y'])
    extent=[x0,x0+int(cfg['nx'])*float(cfg['dx']),y0,y0+int(cfg['ny'])*float(cfg['dy'])]
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,data,title in zip(axes[:2],columns,['Finite volume',f'Particles: mean of {a.seeds} seeds']):
        im=ax.imshow(data,origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='magma')
        ax.set(title=title,xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(im,ax=axes[:2].tolist(),label='Column mass density')
    difference=columns[0]-columns[1];limit=max(float(abs(difference).max()),1e-20)
    im=axes[2].imshow(difference,origin='lower',extent=extent,vmin=-limit,vmax=limit,cmap='RdBu_r')
    axes[2].set(title='Finite volume − particles',xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(im,ax=axes[2],label='Column mass density difference')
    fig.suptitle(f'Matched Gaussian case at t={t:g} s')
    fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
