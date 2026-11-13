"""Run a particle ensemble and sensitivity checks against a completed urban grid run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from results import config, rows
from compare import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-run',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--particles',type=int,default=100000)
    p.add_argument('--times',nargs='+',type=float,default=[120,300,600])
    a=p.parse_args()
    if a.particles<1:p.error('Positive particle count required')
    c=config(a.grid_run/'effective.cfg');case=Path(c['case_dir'])
    if (case/'INCOMPLETE').exists():raise ValueError('Incomplete input case')
    manifest=(case/'input.sha256').read_text()
    if manifest!=(a.grid_run/'input.sha256').read_text():raise ValueError('Grid/prepared input manifests differ')
    seen=set()
    for line in manifest.splitlines():
        digest,name=line.split()
        if name not in {'case.cfg','u.bin','v.bin','w.bin','solid.bin'} or name in seen:raise ValueError('Unexpected manifest')
        seen.add(name);h=hashlib.sha256()
        with (case/name).open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError('Input checksum mismatch: '+name)
    if len(seen)!=5:raise ValueError('Incomplete manifest')
    if c['initial']!='gaussian':raise ValueError('Expected Gaussian grid run')
    grid=rows(a.grid_run/'diagnostics.csv');times=np.array([float(r['time']) for r in grid])
    if times[-1]!=float(c['end_time']):raise ValueError('Grid run incomplete')
    for t in a.times:
        if not np.isclose(times,t,rtol=0,atol=1e-10).any():raise ValueError('Requested comparison time unavailable')
    a.output.mkdir(parents=True,exist_ok=False);review=a.output/'review';review.mkdir()
    keys=['source_x','source_y','source_z','source_sigma','initial_mass','diffusivity','end_time','output_interval','boundary_x','boundary_y','boundary_z']
    common={k:c[k] for k in keys};common.update(case_dir=str(case.resolve()),write_binary='true',write_concentration='true')
    experiments=[('seed12345',12345,a.particles,.1),('seed12346',12346,a.particles,.1),('seed12347',12347,a.particles,.1),('half-dt',12345,a.particles,.05),('double-N',12345,2*a.particles,.1)]
    series=[];reports=[]
    for name,seed,n,dt in experiments:
        cfg=common|dict(seed=seed,particle_count=n,dt=dt,case_name=name)
        path=a.output/(name+'.cfg');path.write_text(''.join(f'{k}={v}\n' for k,v in cfg.items()))
        out=a.output/name
        subprocess.run([str(a.exe.resolve()),str(path),'--output',str(out)],check=True)
        records=rows(out/'diagnostics.csv');t=np.array([float(r['time']) for r in records])
        if len(t)!=len(times) or not np.allclose(t,times,rtol=0,atol=1e-10):raise ValueError('Output times differ')
        for r in records:
            vals=[float(r[k]) for k in ['mass_inside','mass_escaped','mass_residual']]
            if not np.isfinite(vals).all() or abs(vals[2])>1e-9 or min(vals[:2])<0:raise ValueError('Mass accounting failed')
        mass=np.array([float(r['mass_inside']) for r in records]);series.append(mass)
        fields=[compare(a.grid_run,out,int(np.flatnonzero(np.isclose(times,t))[0])) for t in a.times]
        reports.append(dict(name=name,seed=seed,particle_count=n,dt=dt,fields=fields))
        for filename in ['diagnostics.csv','run_summary.txt','effective_config.txt']:
            (review/(name+'-'+filename)).write_bytes((out/filename).read_bytes())
    values=np.array(series);mean=values[:3].mean(axis=0);sd=values[:3].std(axis=0,ddof=1)
    report=dict(runs=reports,seed_count=3,seed_spread='Sample standard deviation, not a confidence interval',
                final_mean_mass=float(mean[-1]),final_seed_sd=float(sd[-1]),
                max_half_dt_retention_change=float(abs(values[3]-values[0]).max()),
                max_double_N_retention_change=float(abs(values[4]-values[0]).max()),
                note='Different dt uses different Brownian paths; sensitivity differences include sampling noise. Building-corner reflection and split integration remain discretizations requiring refinement.')
    (review/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot(times,[float(r['mass_inside']) for r in grid],label='Finite volume',color='black')
    ax.plot(times,mean,label='Particles: 3-seed mean',color='#2563eb')
    ax.fill_between(times,np.maximum(0,mean-sd),mean+sd,alpha=.2,color='#2563eb',label='±1 seed SD')
    ax.plot(times,values[3],'--',label='Particles: half dt (one seed)')
    ax.plot(times,values[4],':',label='Particles: double count (one seed)')
    ax.set(xlabel='Time (s)',ylabel='Mass remaining',title='Urban grid–particle comparison');ax.legend();ax.grid(alpha=.2)
    fig.savefig(review/'retention-comparison.png',dpi=180);plt.close(fig)
    (review/'grid-diagnostics.csv').write_bytes((a.grid_run/'diagnostics.csv').read_bytes())
    (review/'input.sha256').write_text(manifest)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
