"""End-to-end tests for prepared inputs, reflected diffusion and particle snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from results import field, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for name,wall,speed,diffusion,bx in [('wall',True,4,1,'closed'),('diffusive_open',False,0,2,'open'),('advective_open',False,8,0,'open')]:
        case=a.output/(name+'-case');case.mkdir()
        (case/'case.cfg').write_text('nx=12\nny=8\nnz=6\ndx=1\ndy=1\ndz=1\norigin_x=0\norigin_y=0\norigin_z=0\n')
        mask=np.zeros((6,8,12),dtype='u1')
        if wall:mask[:,:,6]=1
        mask.tofile(case/'solid.bin')
        for axis,value in [('u',speed),('v',0),('w',0)]:
            wind=np.full(mask.shape,value,dtype='<f8');wind[mask!=0]=0;wind.tofile(case/(axis+'.bin'))
        (case/'input.sha256').write_text(''.join(hashlib.sha256((case/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['case.cfg','u.bin','v.bin','w.bin','solid.bin']))
        cfg=a.output/(name+'.cfg')
        cfg.write_text(f'case_dir={case.resolve()}\nsource_x=3\nsource_y=4\nsource_z=3\nsource_sigma=0.5\ninitial_mass=1\ndiffusivity={diffusion}\nparticle_count=10000\nseed=123\ndt=0.02\nend_time=2\noutput_interval=1\nboundary_x={bx}\nboundary_y=closed\nboundary_z=closed\nwrite_binary=true\n')
        run=a.output/name
        subprocess.run([str(a.exe.resolve()),str(cfg),'--output',str(run)],check=True)
        _,time,c=field(run);d=rows(run/'diagnostics.csv')[-1]
        assert time==2 and np.isfinite(c).all() and (c>=0).all()
        assert abs(float(d['mass_residual']))<1e-12
        assert abs(c.sum()-float(d['mass_inside']))<1e-12
        assert not c[mask!=0].any()
        if name=='wall':assert not c[:,:,6:].any()
        if name!='advective_open':assert float(d['mass_escaped'])==0
        else:assert float(d['mass_escaped'])>.99
        if name=='wall':
            csvrun=a.output/'wall-csv'
            subprocess.run([str(a.exe.resolve()),str(cfg),'--output',str(csvrun),'--set','write_binary=false'],check=True)
            np.testing.assert_array_equal(field(csvrun)[2],c)
        reports.append(dict(case=name,passed=True,mass_inside=float(d['mass_inside'])))
    (a.output/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('PASS: prepared input, blocked-cell exclusion, reflective diffusion, advective escape, binary/CSV agreement')


if __name__=='__main__':main()
