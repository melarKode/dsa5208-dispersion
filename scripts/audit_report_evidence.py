"""Recompute the report's quantitative claims from bundled cluster evidence."""
import csv,hashlib,json,math,subprocess,sys,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def data(path):return json.loads((ROOT/path).read_text())
def table(path):
    with (ROOT/path).open() as f:return list(csv.DictReader(f))
def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-12):raise ValueError((a,b))
def require(ok,label):
    if not ok:raise ValueError(label)


def main():
    out={}
    v=data('results/atlas-cpu-1031236.venus01/validation/validation.json');require(v['runs']==59,'Atlas run count');out['grid_validation_runs']=v['runs']
    out['convergence']=v['convergence']
    base=Path('results/atlas-scaling-1031246.venus01')
    serial=np.median([float(table(base/f'serial-{i}/timing.csv')[0]['solve_seconds']) for i in [1,2,3]])
    scaling=table(base/'scaling.csv')
    for row in scaling:
        if row['backend']!='OpenMP':continue
        name=f"r{row['ranks']}-t{row['threads']}-{row['decomposition']}-{row['exchange']}"
        times=[float(table(base/f'{name}-{i}/timing.csv')[0]['solve_seconds']) for i in [1,2,3]]
        close(np.median(times),row['median_solve_seconds']);close(serial/np.median(times),row['speedup_vs_serial'])
        close(float(row['speedup_vs_serial'])/(int(row['ranks'])*int(row['threads'])),row['cpu_core_efficiency'])
    best=min((v for v in scaling if v['backend']=='OpenMP'),key=lambda r:float(r['median_solve_seconds']))
    out['grid_scaling']={'serial_median_seconds':float(serial),'best':best}
    out['grid_scaling']['best_halo_median_seconds']=statistics.median(float(table(base/f'r6-t2-slab-blocking-{i}/timing.csv')[0]['halo_seconds']) for i in [1,2,3])
    pb=Path('results/tridiv-update-20261104/vanda-scaling-REVIEW')
    strong=table(pb/'strong_scaling.csv');weak=table(pb/'weak_scaling.csv')
    baseline=next(r for r in strong if r['cores']=='1');weak_base=next(r for r in weak if r['cores']=='1')
    for row in strong:
        require(abs(float(row['speedup'])-float(baseline['loop_s'])/float(row['loop_s']))<1e-5,'Particle speedup')
    for row in weak:
        require(abs(float(row['efficiency'])-float(weak_base['loop_s'])/float(row['loop_s']))<1e-5,'Weak efficiency')
    for filename in ['strong_scaling.csv','weak_scaling.csv','plume_imbalance.csv']:
        for row in table(pb/filename):
            rank_times=table(pb/'runs'/row['run']/'timings.csv')
            require(abs(max(float(r['loop_seconds']) for r in rank_times)-float(row['loop_s']))<1e-5,'Raw Vanda rank timer')
    out['particle_scaling']={'best':min(strong,key=lambda r:float(r['loop_s'])),'weak_16_core':[r for r in weak if r['cores']=='16'],'plume':table(pb/'plume_imbalance.csv')}
    for path in ['results/integrated-parallel-verified/parallel_validation.json','results/tridiv-update-20261104/vanda-validation-REVIEW/parallel_validation.json']:
        runs=data(path);require(len(runs)==72 and all(r['passed'] for r in runs),'72 MPI cases');require(all(r['particles_differing']==0 for r in runs),'Per-ID equality')
    ref=data('results/particle-refactor-verification/verification.json');require(len(ref)==10 and all(not r['different'] for r in ref),'Serial refactor')
    urban=Path('results/atlas-urban-particles-1032182.venus01/review')
    grid=table(urban/'grid-diagnostics.csv');t=np.array([float(r['time']) for r in grid]);g=np.array([float(r['mass_inside']) for r in grid])
    curves=[]
    for name in ['seed12345','seed12346','seed12347','half-dt','double-N']:
        rows=table(urban/(name+'-diagnostics.csv'));require([float(r['time']) for r in rows]==list(t),'Urban times')
        curves.append(np.array([float(r['mass_inside']) for r in rows]))
    curves=np.array(curves);mean=curves[:3].mean(0);sd=curves[:3].std(0,ddof=1)
    comp=data(urban/'comparison.json');close(mean[-1],comp['final_mean_mass']);close(sd[-1],comp['final_seed_sd'])
    out['retention']={'grid_final':g[-1],'particle_mean_final':mean[-1],'seed_sd_final':sd[-1],'max_difference_pp':float(100*abs(g-mean).max()),'max_difference_time':float(t[abs(g-mean).argmax()]),'half_dt_max_pp':float(100*abs(curves[3]-curves[0]).max()),'double_n_max_pp':float(100*abs(curves[4]-curves[0]).max()),'clearance_50_seconds':float(np.interp(.5,1-g,t)),'clearance_90_seconds':float(np.interp(.9,1-g,t)),'grid_max_mass_residual':max(abs(float(r['mass_residual'])) for r in grid)}
    agg=Path('results/atlas-aggregate-1036035.venus01');rows=table(agg/'metrics.csv');require(len(rows)==180,'Aggregation rows')
    maps=np.load(ROOT/agg/'column-maps.npz');selected=[]
    for ti,time in enumerate([120,300,600]):
        for f in [4,8,16,32]:
            row=next(r for r in rows if r['method']=='ensemble-mean' and r['mode']=='column' and int(r['factor'])==f and float(r['time'])==time)
            a,b=maps[f'grid_t{ti}_f{f}'],maps[f'particles_t{ti}_f{f}'];close(a.sum(),row['mass_grid']);close(b.sum(),row['mass_particles']);close(abs(a-b).sum()/(a.sum()+b.sum()),row['normalized_l1']);close(.5*abs(a/a.sum()-b/b.sum()).sum(),row['survivor_total_variation'])
        for run in comp['runs']:
            native=next(x for x in run['fields'] if x['time']==time)
            row=next(r for r in rows if r['method']==run['name'] and r['mode']=='3d' and r['factor']=='1' and float(r['time'])==time);close(row['l1_mass'],native['l1'])
        selected.extend([r for r in rows if r['method']=='ensemble-mean' and float(r['time'])==time])
    out['aggregation']=selected
    replay=Path('results/atlas-urban-particle-mpi-1033401.venus01/review');eq=data(replay/'equivalence.json')
    require(eq['passed'] and eq['diagnostics_match'] and len(eq['snapshots'])==21,'Urban MPI replay')
    require(all(r[k]==0 for r in eq['snapshots'] for k in ['l1','l2','linf']),'Zero recorded field differences')
    timings=table(replay/'timings.csv');out['urban_mpi']={k:max(float(r[k]) for r in timings) for k in ['total_seconds','loop_seconds','output_seconds','wait_seconds','stepping_seconds','migration_seconds']}
    audit=data('results/atlas-urban-pilot-1032143.venus01/review/cropped-wind-audit.json');require(audit['air_cells']+audit['solid_cells']==641*556*22,'Urban cell counts');out['urban_geometry_audit']=audit
    mask=data('data/geometry/mask-report.json');require(hashlib.sha256((ROOT/'data/geometry/00-buildings.stl').read_bytes()).hexdigest()==mask['source_sha256'],'Bundled STL provenance')
    out['multinode']={}
    for cluster, job in [('atlas','1040644.venus01'),('vanda','1547560.stdct-mgmt-02')]:
        base=ROOT/f'results/{cluster}-multinode-{job.split(".")[0]}-review/results/{cluster}-multinode-{job}'
        subprocess.run([sys.executable,str(ROOT/'scripts/review_multinode_run.py'),str(base)],check=True,stdout=subprocess.DEVNULL)
        out['multinode'][cluster]=json.loads((base/'independent-review.json').read_text())
        out['multinode'][cluster]['blocking_phase_medians']=[dict(nodes=n,**{k:statistics.median(float(table((base/f'measured-n{n}-blocking-r{i}'/'timing.csv').relative_to(ROOT))[0][k]) for i in [1,2,3]) for k in ['update_seconds','halo_seconds']}) for n in [1,2,4]]
    serial_summary=dict(line.split('=',1) for line in (ROOT/urban/'seed12345-run_summary.txt').read_text().splitlines() if '=' in line)
    out['urban_serial']={k:float(serial_summary[k]) for k in ['wall_seconds','output_seconds']}
    out['passed']=True
    dest=ROOT/'report/evidence-audit.json';dest.write_text(json.dumps(out,indent=2)+'\n');print('PASS: report numerical evidence, raw scaling medians, native/aggregated comparisons and STL provenance.');print(dest)

if __name__=='__main__':main()
