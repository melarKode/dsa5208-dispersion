"""Validate cross-node execution, then measure fixed-problem CPU strong scaling."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shlex
import shutil
import subprocess
import time
from compare import compare
from results import rows


def write_launch_agent(path, daemon, environment):
    if not daemon.is_file() or not os.access(daemon, os.X_OK):
        raise FileNotFoundError(f'Missing executable MPI daemon: {daemon}')
    lines = ['#!/bin/bash', 'set -e']
    for name in ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        if name in environment:
            lines.append(f'export {name}={shlex.quote(environment[name])}')
    lines.append(f'exec {shlex.quote(str(daemon))} "$@"')
    path.write_text('\n'.join(lines) + '\n')
    path.chmod(0o700)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build', type=Path, default=Path('build/atlas-cpu'))
    p.add_argument('--batch-script', type=Path, default=Path('scripts/atlas_multinode.pbs'))
    p.add_argument('--local-test', action='store_true')
    p.add_argument('--transport', choices=('tcp', 'default'), default='tcp')
    p.add_argument('--tcp-interface', default=os.environ.get('DISPERSION_TCP_INTERFACE'))
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise ValueError('Use a new output directory; never mix runs')
    ranks_per_node, threads = (2, 1) if a.local_test else (6, 2)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OMP_PLACES'] = 'cores'
    os.environ['OMP_PROC_BIND'] = 'close'
    os.environ['OMP_DYNAMIC'] = 'false'
    hosts = ['localhost'] if a.local_test else list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    if not a.local_test and len(hosts) not in (2, 4):
        raise ValueError(f'Expected 2 or 4 distinct nodes, got {hosts}')
    counts = [n for n in (1, 2, 4) if n <= len(hosts)]
    binaries = {s: (a.build / ('dispersion_' + s)).resolve() for s in ('serial', 'mpi')}
    hashes = {}
    for path in [*binaries.values(), Path(__file__), a.batch_script, *Path('src').glob('*.cpp'), *Path('include').rglob('*.hpp')]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(local_test=a.local_test, transport=a.transport, tcp_interface=a.tcp_interface, hosts=hosts, nodes=counts, ranks_per_node=ranks_per_node,
                    threads=threads, repetitions=3, hashes=hashes, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    baseline='same MPI implementation on one node, separately for each exchange',
                    scope='synthetic fixed-problem CPU scaling; not urban, GPU, or serial speedup')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def command(cmd, label):
        print(f'RUN {label}', flush=True)
        with (root / 'commands.jsonl').open('a') as f:
            f.write(json.dumps(dict(label=label, argv=[str(x) for x in cmd])) + '\n')
        log = root / (label + '.log')
        try:
            with log.open('w') as f:
                subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print('\n'.join(log.read_text(errors='replace').splitlines()[-60:]), flush=True)
            raise

    agent = root / 'launch-prted.sh'
    if not a.local_test:
        daemon_name = shutil.which('prted')
        if daemon_name is None:
            raise FileNotFoundError('Loaded MPI modules do not provide prted on PATH')
        daemon = Path(daemon_name).resolve()
        write_launch_agent(agent, daemon, os.environ)
        command([agent, '--version'], 'daemon-version')
        # PRRTE 3.0.5 registers this setting before processing --launch-agent.
        os.environ['PRTE_MCA_prte_launch_agent'] = str(agent)

    def launch(nodes, per_node=None, pe=None):
        per_node = per_node or ranks_per_node
        pe = pe or threads
        cmd = ['mpirun', '-np', str(nodes * per_node)]
        if a.transport == 'tcp':
            cmd += ['--mca', 'pml', 'ob1', '--mca', 'btl', 'self,sm,tcp']
            if a.tcp_interface:
                cmd += ['--mca', 'btl_tcp_if_include', a.tcp_interface]
        if a.local_test:
            cmd += ['--oversubscribe', '--bind-to', 'none']
        else:
            cmd += ['--host', ','.join(h + ':12' for h in hosts[:nodes]),
                    '--map-by', f'ppr:{per_node}:node:PE={pe}', '--bind-to', 'core', '--report-bindings']
        for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND', 'OMP_DYNAMIC'):
            if name in os.environ:
                cmd += ['-x', name]
        return cmd

    for n in counts:
        command(launch(n) + ['bash', '-c', 'echo PLACEMENT $(hostname)'], f'placement-{n}')
        observed = Counter(line.split()[1].split('.')[0].lower() for line in (root / f'placement-{n}.log').read_text().splitlines() if line.startswith('PLACEMENT '))
        if not a.local_test and observed != Counter({h.split('.')[0].lower(): ranks_per_node for h in hosts[:n]}):
            raise ValueError(f'Unexpected physical rank placement: {observed}')
    if not a.local_test:
        command(launch(len(hosts), 1, 1) + ['bash', '-c', 'hostname; lscpu; uname -a'], 'hardware')

    common = ['decomposition=block', 'dt=0.5', 'initial_mass=1', 'u=0.2', 'v=0.1', 'w=0',
              'diffusivity=0.1', 'boundary_x=periodic', 'boundary_y=periodic', 'boundary_z=periodic']
    small = common + ['nx=65', 'ny=49', 'nz=9', 'source_x=20', 'source_y=24', 'source_z=4',
                      'source_sigma=2', 'end_time=2', 'output_interval=1', 'snapshots=binary']
    def run(nodes, mode, label, settings, serial=False):
        directory = root / label
        cmd = launch(1, 1, 1) if serial else launch(nodes)
        command(cmd + [binaries['serial' if serial else 'mpi'], 'configs/scaling.cfg', *settings,
                       f'exchange={mode}', f'output_dir={directory}'], label)
        diagnostics = rows(directory / 'diagnostics.csv')
        for row in diagnostics:
            vals = [float(row[k]) for k in ('mass_inside', 'mass_escaped', 'mass_residual', 'min_concentration')]
            if not all(math.isfinite(x) for x in vals) or abs(vals[2]) > 1e-9 or vals[3] < -1e-12:
                raise ValueError(f'Mass/positivity failure in {label}')
        return directory

    reference = run(1, 'blocking', 'validation-serial', small, serial=True)
    checks = []
    for n in counts:
        for mode in ('blocking', 'overlap'):
            folder = run(n, mode, f'validation-n{n}-{mode}', small)
            for index in range(len(rows(reference / 'snapshots.csv'))):
                result = compare(reference, folder, index)
                if not all(math.isfinite(result[k]) for k in ('l1', 'l2', 'linf')) or result['linf'] > 1e-11 or result['l1'] > 1e-10:
                    raise ValueError(f'Field mismatch: {result}')
                checks.append(dict(nodes=n, exchange=mode, **result))
    (root / 'validation.json').write_text(json.dumps(dict(passed=True, checks=checks), indent=2) + '\n')
    print('PASS: all saved small-case fields agree with serial', flush=True)
    nx, ny, nz, end = (32, 32, 8, 2) if a.local_test else (512, 512, 64, 100)
    big = common + [f'nx={nx}', f'ny={ny}', f'nz={nz}', f'source_x={nx/4}', f'source_y={ny/2}',
                    f'source_z={nz/2}', 'source_sigma=4', f'end_time={end}', f'output_interval={end}', 'snapshots=none']
    # Warm each layout, then alternate measurement order across repetitions.
    for n in counts:
        for mode in ('blocking', 'overlap'):
            run(n, mode, f'warmup-n{n}-{mode}', big)
    measurements = []
    for repeat in range(1, 4):
        for n in (counts if repeat % 2 else counts[::-1]):
            for mode in (('blocking', 'overlap') if repeat % 2 else ('overlap', 'blocking')):
                label = f'measured-n{n}-{mode}-r{repeat}'
                folder = run(n, mode, label, big)
                row = rows(folder / 'timing.csv')[0]
                if int(row['ranks']) != n*ranks_per_node or int(row['threads']) != threads or int(row['steps']) != int(end/0.5) or row['backend'] != 'OpenMP':
                    raise ValueError(f'Unexpected timing configuration: {row}')
                if not math.isfinite(float(row['solve_seconds'])) or float(row['solve_seconds']) <= 0:
                    raise ValueError('Invalid timing')
                measurements.append(dict(row, nodes=n, repeat=repeat, directory=label))
                (root / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
    summary = []
    for mode in ('blocking', 'overlap'):
        baseline = statistics.median(float(r['solve_seconds']) for r in measurements if r['nodes'] == 1 and r['exchange'] == mode)
        for n in counts:
            values = [float(r['solve_seconds']) for r in measurements if r['nodes'] == n and r['exchange'] == mode]
            median = statistics.median(values)
            summary.append(dict(nodes=n, cores=n*ranks_per_node*threads, exchange=mode, repeats=len(values),
                                median_seconds=median, min_seconds=min(values), max_seconds=max(values),
                                speedup_vs_one_node=baseline/median, efficiency_vs_one_node=baseline/median/n,
                                relative_range=(max(values)-min(values))/median))
    with (root / 'scaling.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (root / 'SUCCESS.json').write_text(json.dumps(dict(passed=True, local_test=a.local_test, summary=summary), indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('SUCCESS: validation and all measured runs complete', flush=True)


if __name__ == '__main__':
    main()
