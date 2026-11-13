"""Check every full-case MPI snapshot against a completed serial particle run."""
import argparse,json
from pathlib import Path
import numpy as np
from results import rows
from compare import compare
p=argparse.ArgumentParser(description=__doc__);p.add_argument('serial',type=Path);p.add_argument('parallel',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
sa=rows(a.serial/'snapshots.csv');sb=rows(a.parallel/'snapshots.csv')
if len(sa)!=len(sb):raise SystemExit('Different snapshot counts')
reports=[compare(a.serial,a.parallel,i) for i in range(len(sa))]
if max(r['linf'] for r in reports)>1e-12:raise SystemExit('Concentration mismatch')
da=rows(a.serial/'diagnostics.csv');db=rows(a.parallel/'diagnostics.csv')
if len(da)!=len(db):raise SystemExit('Different diagnostic lengths')
for x,y in zip(da,db):
 for k in x:
  if not np.isclose(float(x[k]),float(y[k]),rtol=1e-10,atol=1e-12,equal_nan=True):raise SystemExit('Diagnostic mismatch: '+k)
a.output.write_text(json.dumps(dict(passed=True,snapshots=reports,diagnostics_match=True),indent=2)+'\n')
print('PASS: every concentration snapshot and diagnostic matches serial')
