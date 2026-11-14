"""Rebuild report figures from recorded cluster summaries; no simulation is run."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/figures';OUT.mkdir(parents=True,exist_ok=True)
def table(path):
 with (ROOT/path).open() as f:return list(csv.DictReader(f))
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
v=json.loads((ROOT/'results/atlas-cpu-1031236.venus01/validation/validation.json').read_text())
fig,ax=plt.subplots(figsize=(6,3.4))
for name in ['diffusion','advection','combined']:
 r=[x for x in v['convergence'] if x['case']==name];n=np.array([x['n'] for x in r]);err=np.array([x['l1'] for x in r]);ax.loglog(n,err,'o-',label=name)
ax.set(xlabel='Cells per refined axis',ylabel='Volume-weighted L1 error');ax.legend();ax.grid(alpha=.2,which='both');save(fig,'convergence')
s=table(Path('results/atlas-scaling-1031246.venus01/scaling.csv'));parallel=[r for r in s if r['backend']=='OpenMP']
fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for threads in [1,2]:
 r=sorted([r for r in parallel if int(r['threads'])==threads and r['decomposition']=='slab' and r['exchange']=='blocking'],key=lambda r:int(r['ranks']))
 axes[0].plot([int(x['ranks'])*threads for x in r],[float(x['speedup_vs_serial']) for x in r],'o-',label=f'{threads} thread(s)/rank')
axes[0].plot([1,12],[1,12],'k:',label='Ideal');axes[0].set(xlabel='CPU cores',ylabel='Speedup vs serial',title='Atlas grid strong scaling');axes[0].legend();axes[0].grid(alpha=.2)
r=sorted([r for r in parallel if int(r['ranks'])==6 and int(r['threads'])==2],key=lambda r:(r['decomposition'],r['exchange']))
axes[1].bar(range(len(r)),[float(x['median_solve_seconds']) for x in r],color='#4676b8');axes[1].set_xticks(range(len(r)),[x['decomposition']+'\n'+x['exchange'] for x in r]);axes[1].set(ylabel='Median solve time (s)',title='Same 6 ranks × 2 threads');save(fig,'grid_scaling')
base=Path('results/tridiv-update-20261104/vanda-scaling-REVIEW')
strong=table(base/'strong_scaling.csv');weak=table(base/'weak_scaling.csv');plume=table(base/'plume_imbalance.csv')
fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for label,predicate in [('Threads',lambda r:int(r['ranks'])==1),('MPI',lambda r:int(r['threads'])==1),('Hybrid',lambda r:int(r['threads'])==4 and int(r['ranks'])>1)]:
 r=[r for r in strong if predicate(r)];axes[0].plot([int(x['cores']) for x in r],[float(x['speedup']) for x in r],'o-',color={'Threads':'#1f77b4','MPI':'#ff7f0e','Hybrid':'#2ca02c'}[label],label=label)
 r=[r for r in weak if predicate(r)]
 if len(r)>1:axes[1].plot([int(x['cores']) for x in r],[float(x['efficiency'])*100 for x in r],'o-',color={'Threads':'#1f77b4','MPI':'#ff7f0e','Hybrid':'#2ca02c'}[label],label=label)
axes[0].plot([1,16],[1,16],'k:');axes[0].set(xlabel='CPU cores',ylabel='Speedup vs 1-rank/1-thread',title='Vanda particles: fixed 10 million')
axes[1].set(xlabel='CPU cores',ylabel='Weak-scaling efficiency (%)',title='1 million particles per core',ylim=(0,105))
for ax in axes:ax.legend();ax.grid(alpha=.2)
save(fig,'particle_scaling')
fig,axes=plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
labels=[f"{r['ranks']} rank"+("s" if int(r["ranks"])>1 else "")+f" ×\n{r['threads']} thread"+("s" if int(r["threads"])>1 else "") for r in plume]
axes[0].bar(labels,[float(r['loop_s']) for r in plume],color='#4676b8');axes[0].set(ylabel='Loop time (s)',title='Compact plume: same 16 cores')
axes[1].bar(labels,[float(r['imbalance_max']) for r in plume],color='#d57b35');axes[1].set(ylabel='Maximum / mean active load',title='Worst recorded imbalance')
save(fig,'imbalance')
p=Path('results/atlas-urban-particles-1032182.venus01/review')
g=table(p/'grid-diagnostics.csv');t=np.array([float(r['time']) for r in g]);mass=np.array([float(r['mass_inside']) for r in g])
seeds=[table(p/f'seed{x}-diagnostics.csv') for x in [12345,12346,12347]]
m=np.array([[float(r['mass_inside']) for r in data] for data in seeds]);mean=m.mean(axis=0);sd=m.std(axis=0,ddof=1)
half=np.array([float(r['mass_inside']) for r in table(p/'half-dt-diagnostics.csv')]);double=np.array([float(r['mass_inside']) for r in table(p/'double-N-diagnostics.csv')])
fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
axes[0].plot(t,mass,'k-',label='Grid');axes[0].plot(t,mean,label='Particles: mean of 3 seeds');axes[0].set(xlabel='Time (s)',ylabel='Fraction of initial mass inside');axes[0].legend()
axes[1].errorbar(t,100*(mean-mass),yerr=100*sd,capsize=2,lw=1.3,label='Particle mean − grid (±1 seed SD)');axes[1].plot(t,100*(half-m[0]),'--',label='Half dt − baseline seed');axes[1].plot(t,100*(double-m[0]),':',label='Double N − baseline seed');axes[1].axhline(0,color='gray',lw=.6);axes[1].set(xlabel='Time (s)',ylabel='Difference (percentage points)');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
save(fig,'retention')
r=json.loads((ROOT/p/'comparison.json').read_text())['runs']
fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
for run,style,color,marker in [(r[3],'--','#e17c05','x'),(r[4],':','#2ca02c','^'),(r[0],'-','#1f77b4','o')]:
 f=run['fields'];times=[x['time'] for x in f];label={'seed12345':'100k, dt=0.1','half-dt':'100k, dt=0.05','double-N':'200k, dt=0.1'}[run['name']]
 axes[0].plot(times,[x['l1'] for x in f],linestyle=style,color=color,marker=marker,ms=8 if marker=='o' else 5,mfc='none',mew=1.3,label=label)
 axes[1].plot(times,[x['l1']/(x['mass_a']+x['mass_b']) for x in f],linestyle=style,color=color,marker=marker,ms=8 if marker=='o' else 5,mfc='none',mew=1.3,label=label)
axes[0].set(xlabel='Time (s)',ylabel='Raw concentration L1 (mass units)');axes[1].set(xlabel='Time (s)',ylabel='L1 / (grid mass + particle mass)',ylim=(0,1.05))
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
save(fig,'field_difference')
fig,axes=plt.subplots(1,3,figsize=(10,3.2),constrained_layout=True)
for ax,d in zip(axes,'xyz'):
 ax.plot(t,[float(x['mean_'+d]) for x in g],'k-',label='Grid')
 ax.plot(t,np.mean([[float(x['mean_'+d]) for x in data] for data in seeds],axis=0),label='Particle mean')
 ax.set(xlabel='Time (s)',ylabel=f'Mean {d} (m)');ax.grid(alpha=.2)
axes[0].legend();save(fig,'centroids')
metrics=dict(grid_best=min(parallel,key=lambda r:float(r['median_solve_seconds'])),particle_best=max(strong,key=lambda r:float(r['speedup'])),final_grid=float(mass[-1]),final_particles=float(mean[-1]),final_particle_sd=float(sd[-1]),max_retention_difference_pp=float(abs(mean-mass).max()*100),max_retention_difference_time=float(t[np.argmax(abs(mean-mass))]),max_half_dt_change_pp=float(abs(half-m[0]).max()*100),max_double_N_change_pp=float(abs(double-m[0]).max()*100))
(ROOT/'report/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
