import json
import os
from pathlib import Path
import subprocess
import sys
from run_multinode import write_launch_agent
root = Path('results') / ('atlas-network-' + os.environ['PBS_JOBID'])
root = root.resolve()
root.mkdir()
hosts = list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
agent = root / 'daemon.sh'
write_launch_agent(agent, Path(os.environ['EBROOTPRRTE'])/'bin/prted', os.environ)
os.environ['PRTE_MCA_prte_launch_agent'] = str(agent)
base = ['mpirun', '-np', '2', '--host', ','.join(hosts), '--map-by', 'ppr:1:node', '--bind-to', 'core', '-x', 'LD_LIBRARY_PATH', '-x', 'LD_PRELOAD', '-x', 'OMP_NUM_THREADS']
probe = root/'inspect.sh'
probe.write_text('#!/bin/bash\nhostname\nip -o -4 addr show\nip -4 route\ngetent hosts '+ ' '.join(hosts)+'\n')
probe.chmod(0o700)
subprocess.run(base+[str(probe)],check=True,timeout=90)
for interface in ('default', 'eth0'):
    cmd=base+['--mca','pml','ob1','--mca','btl','self,sm,tcp','--timeout','45']
    if interface != 'default':
        cmd += ['--mca','btl_tcp_if_include',interface]
    cmd += ['build/atlas-cpu/mpi_kokkos_smoke']
    print('PROBE',interface,flush=True)
    with (root/(interface+'.log')).open('w') as f:
        try:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=60)
            print('EXIT',result.returncode,flush=True)
        except subprocess.TimeoutExpired:
            print('TIMEOUT',flush=True)
    print((root/(interface+'.log')).read_text(errors='replace')[-5000:],flush=True)
