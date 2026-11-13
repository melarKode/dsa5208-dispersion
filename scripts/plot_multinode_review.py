"""Plot within-cluster CPU scaling, retaining timing spread and distinct baselines."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from results import rows
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--cluster',required=True)
p.add_argument('--output-prefix',type=Path)
a=p.parse_args()
records=rows(a.root/'scaling.csv')
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for mode,color in [('blocking','#2463a0'),('overlap','#dd7a24')]:
    data=sorted([r for r in records if r['exchange']==mode],key=lambda r:int(r['nodes']))
    n=[int(r['nodes']) for r in data]
    median=[float(r['median_seconds']) for r in data]
    lo=[m-float(r['min_seconds']) for m,r in zip(median,data)]
    hi=[float(r['max_seconds'])-m for m,r in zip(median,data)]
    axes[0].errorbar(n,median,yerr=[lo,hi],fmt='o-',capsize=4,color=color,label=mode)
    axes[1].plot(n,[float(r['speedup_vs_one_node']) for r in data],'o-',color=color,label=mode)
    axes[2].plot(n,[100*float(r['efficiency_vs_one_node']) for r in data],'o-',color=color,label=mode)
axes[1].plot([1,4],[1,4],'--',color='.65',label='ideal')
for ax,title in zip(axes,['Solve time (s): median and min–max','Speedup vs own one-node baseline','Efficiency vs one node (%)']):
    ax.set_title(title,fontsize=10); ax.set_xlabel('Nodes (12 CPU cores each)'); ax.set_xticks([1,2,4]); ax.grid(alpha=.2)
axes[0].legend(frameon=False)
fig.suptitle(f'{a.cluster}: fixed 512 × 512 × 64 grid, 200 steps, three repetitions; TCP transport',fontsize=11)
prefix=a.output_prefix or a.root/'multinode-review'
prefix.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(prefix.with_suffix('.png'),dpi=180)
fig.savefig(prefix.with_suffix('.pdf'))
print(prefix.with_suffix('.png'))
