"""Conservatively aggregate existing grid and particle snapshots into shared bins."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from results import config,field,rows
from compare import compare


def block_sum(values, factor):
    if factor < 1 or int(factor) != factor:
        raise ValueError('Bin factor must be a positive integer')
    result=np.asarray(values,dtype=np.float64)
    for axis in range(result.ndim):
        result=np.add.reduceat(result,np.arange(0,result.shape[axis],factor),axis=axis)
    return result


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Expected matching finite nonnegative bin masses')
    ma=float(a.sum());mb=float(b.sum());l1=float(np.abs(a-b).sum())
    return dict(mass_grid=ma,mass_particles=mb,l1_mass=l1,
                normalized_l1=l1/(ma+mb) if ma+mb else 0.,
                survivor_total_variation=float(.5*np.abs(a/ma-b/mb).sum()) if ma>0 and mb>0 else None)


def index_at(run,time):
    found=[i for i,r in enumerate(rows(run/'snapshots.csv')) if abs(float(r['time'])-time)<1e-10]
    if len(found)!=1:raise ValueError(f'No unique snapshot at {time}: {run}')
    return found[0]


def check_mass(run,time,mass):
    found=[r for r in rows(run/'diagnostics.csv') if abs(float(r['time'])-time)<1e-10]
    if len(found)!=1 or not np.isclose(mass,float(found[0]['mass_inside']),rtol=1e-10,atol=1e-12):
        raise ValueError(f'Field/diagnostic mass mismatch: {run}, t={time}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-run',type=Path,required=True);p.add_argument('--particle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--times',type=float,nargs='+',default=[120,300,600])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=['seed12345','seed12346','seed12347','half-dt','double-N'];factors=[1,4,8,16,32]
    cfg=config(a.grid_run/'effective.cfg');dv=np.prod([float(cfg['d'+k]) for k in 'xyz'])
    solid=np.fromfile(Path(cfg['case_dir'])/'solid.bin',dtype='u1').reshape(tuple(int(cfg['n'+k]) for k in 'zyx'))
    payload={'footprint':solid.any(axis=0)};records=[];checks=[]
    for k in 'xy':
        payload[k]=float(cfg['origin_'+k])+(np.arange(int(cfg['n'+k]))+.5)*float(cfg['d'+k])
    for ti,time in enumerate(a.times):
        gi=index_at(a.grid_run,time);_,_,g=field(a.grid_run,gi);g*=dv;check_mass(a.grid_run,time,float(g.sum()))
        gb={f:block_sum(g,f) for f in factors};gc={f:block_sum(g.sum(axis=0),f) for f in factors}
        ensemble={f:np.zeros_like(gb[f]) for f in factors};columns={f:[] for f in factors}
        for name in names:
            run=a.particle_root/name;pi=index_at(run,time)
            if pi!=gi:raise ValueError('Snapshot index conventions differ; align before comparison')
            # Reuse the established model, boundary-law, input-checksum and time guards.
            native=compare(a.grid_run,run,gi)
            _,_,particle=field(run,pi);particle*=dv;check_mass(run,time,float(particle.sum()))
            pb={f:block_sum(particle,f) for f in factors};pc={f:block_sum(particle.sum(axis=0),f) for f in factors}
            for mode,left,right in [('3d',gb,pb),('column',gc,pc)]:
                previous=float('inf')
                for f in factors:
                    r=metrics(left[f],right[f]);assert r['l1_mass']<=previous+1e-12;previous=r['l1_mass']
                    assert np.isclose(right[f].sum(),particle.sum(),rtol=1e-12,atol=1e-13)
                    records.append(dict(time=time,method=name,mode=mode,factor=f,**r))
            assert np.isclose(metrics(gb[1],pb[1])['l1_mass'],native['l1'],rtol=1e-10,atol=1e-12)
            if name.startswith('seed'):
                for f in factors:ensemble[f]+=pb[f]/3;columns[f].append(pc[f])
            print(f'Aggregated t={time:g}: {name}',flush=True)
        for f in factors:
            mean=np.mean(columns[f],axis=0);sd=np.std(columns[f],axis=0,ddof=1)
            for mode,left,right in [('3d',gb[f],ensemble[f]),('column',gc[f],mean)]:
                records.append(dict(time=time,method='ensemble-mean',mode=mode,factor=f,**metrics(left,right)))
            if f>1:
                payload[f'grid_t{ti}_f{f}']=gc[f];payload[f'particles_t{ti}_f{f}']=mean;payload[f'seed_sd_t{ti}_f{f}']=sd
        checks.append(dict(time=time,grid_mass=float(g.sum()),native_norm_reproduced=True,partial_edge_bins_retained=True,mass_conserved_all_factors=True,nested_l1_nonincreasing=True))
    with (a.output/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(a.output/'column-maps.npz',**payload)
    report=dict(grid_run=str(a.grid_run),particle_root=str(a.particle_root),config=cfg,times=a.times,factors=factors,seeds=names[:3],checks=checks,
        definitions={'3d':'Sum concentration times native cell volume in f by f by f cells; retain partial edge bins.',
        'column':'Sum mass over all z, then f by f horizontal cells; maps divide by exact bin footprint area.',
        'normalized_l1':'sum(abs(grid_bin_mass - particle_bin_mass)) / (grid_mass + particle_mass)',
        'survivor_total_variation':'0.5 * sum(abs(grid_bin_mass/grid_mass - particle_bin_mass/particle_mass))',
        'ensemble-mean':'Equal-weight mean of three separate 100000-particle seed fields; not the mean of their errors.',
        'seed_sd':'Sample standard deviation across three seed maps; not a confidence interval.'},
        limitations='Coarsening cannot increase L1 for nested bins. A lower error alone is not evidence of physical validity or numerical convergence. Column maps additionally integrate out height. The half-dt and double-N runs each contain one seed.')
    (a.output/'aggregation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUCCESS: spatial aggregation complete:',a.output,flush=True)

if __name__=='__main__':main()
