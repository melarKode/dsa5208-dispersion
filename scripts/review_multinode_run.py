"""Recompute multi-node validation and timing summaries from downloaded evidence."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import statistics
from compare import compare
from results import rows, config
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
a=p.parse_args(); root=a.root
manifest=json.loads((root/'manifest.json').read_text())
assert not manifest['local_test']
assert json.loads((root/'SUCCESS.json').read_text())['passed']
checks=[]
for n in manifest['nodes']:
    seen=Counter(line.split()[1].split('.')[0].lower() for line in (root/f'placement-{n}.log').read_text().splitlines() if line.startswith('PLACEMENT '))
    assert seen==Counter({h.split('.')[0].lower():manifest['ranks_per_node'] for h in manifest['hosts'][:n]})
    for mode in ('blocking','overlap'):
        directory=root/f'validation-n{n}-{mode}'
        for i in range(len(rows(root/'validation-serial'/'snapshots.csv'))):
            r=compare(root/'validation-serial',directory,i)
            assert r['l1']<=1e-10 and r['linf']<=1e-11
            checks.append(r)
records=[]
numerics=[]
for directory in sorted(p for p in root.glob('measured-*') if p.is_dir()):
    row=rows(directory/'timing.csv')[0]
    cfg=config(directory/'effective.cfg')
    numerics.append({k:v for k,v in cfg.items() if k not in ('output_dir','exchange','px','py')})
    assert (int(row['nx']),int(row['ny']),int(row['nz']),int(row['steps']))==(512,512,64,200)
    assert cfg['snapshots']=='none' and row['backend']=='OpenMP'
    n=int(row['ranks'])//manifest['ranks_per_node']
    assert int(row['ranks'])==n*manifest['ranks_per_node'] and int(row['threads'])==manifest['threads']
    for d in rows(directory/'diagnostics.csv'):
        assert abs(float(d['mass_residual']))<=1e-9 and float(d['min_concentration'])>=-1e-12
    bindings={}; ranks=set()
    log=(root/(directory.name+'.log')).read_text(errors='replace')
    for host,rank,cores in re.findall(r'\[([^:\]]+):\d+\] Rank (\d+) bound to package\[\d+\]\[core:([^\]]+)\]',log):
        assert rank not in ranks
        ranks.add(rank); values=set()
        for part in cores.split(','):
            if '-' in part:
                start,end=map(int,part.split('-')); values.update(range(start,end+1))
            else:
                values.add(int(part))
        assert len(values)==manifest['threads']
        assert not bindings.setdefault(host,set()).intersection(values)
        bindings[host].update(values)
    assert len(bindings)==n and len(ranks)==int(row['ranks'])
    records.append(dict(nodes=n,exchange=row['exchange'],seconds=float(row['solve_seconds'])))
assert all(c==numerics[0] for c in numerics)
assert len(records)==len(manifest['nodes'])*6
for row in rows(root/'scaling.csv'):
    n=int(row['nodes']); mode=row['exchange']
    values=[r['seconds'] for r in records if r['nodes']==n and r['exchange']==mode]
    baseline=statistics.median(r['seconds'] for r in records if r['nodes']==1 and r['exchange']==mode)
    assert len(values)==3
    median=statistics.median(values)
    expected=dict(median_seconds=median,min_seconds=min(values),max_seconds=max(values),speedup_vs_one_node=baseline/median,efficiency_vs_one_node=baseline/median/n)
    for key,value in expected.items():
        assert math.isclose(float(row[key]),value,rel_tol=1e-12)
report=dict(passed=True,placement_checks=len(manifest['nodes']),field_comparisons=len(checks),measured_runs=len(records),maximum_validation_l1=max(r['l1'] for r in checks),maximum_validation_linf=max(r['linf'] for r in checks),transport=manifest['transport'],summary=rows(root/'scaling.csv'))
(root/'independent-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
