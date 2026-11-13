"""Compare original and integrated serial particle physics using identical compilers."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);report=[]
 cases=list(Path('lagrangian/cases/validation').glob('*.cfg'))+[Path('lagrangian/cases/example_gaussian_uniform.cfg')]
 cases += [Path('results/urban-particle-tests')/(n+'.cfg') for n in ['wall','diffusive_open','advective_open']]
 for cfg in cases:
  folders=[]
  for label,exe in [('old',a.old),('new',a.new)]:
   out=a.output/cfg.stem/label;folders.append(out)
   subprocess.run([str(exe.resolve()),str(cfg),'--output',str(out),'--set','particle_count=10000','--set','write_particles=true'],check=True,stdout=subprocess.DEVNULL)
  names=['diagnostics.csv','snapshots.csv']+[f.name for f in folders[0].glob('concentration_*')]+[f.name for f in folders[0].glob('particles_*')]
  different=[n for n in names if (folders[0]/n).read_bytes()!=(folders[1]/n).read_bytes()]
  report.append(dict(case=cfg.stem,files=len(names),different=different))
 (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 if any(r['different'] for r in report):raise SystemExit('Refactor changed numerical output')
 print(f'PASS: {len(report)} cases, byte-identical diagnostics, snapshots and particle states')

if __name__=='__main__':main()
