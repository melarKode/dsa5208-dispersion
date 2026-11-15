"""Package source, report and compact measured evidence with file checksums."""
from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
files = set()
excluded = {'__pycache__', '.git', '.DS_Store', 'build', 'build-par', 'build-serial'}

def include(path):
    p = root / path
    if not p.exists():
        raise FileNotFoundError(p)
    candidates = p.rglob('*') if p.is_dir() else [p]
    for f in candidates:
        rel = f.relative_to(root)
        if f.is_file() and not any(x in excluded or x.startswith('build-') for x in rel.parts):
            if f.suffix not in {'.pyc', '.aux', '.log', '.out', '.toc'}:
                files.add(rel)

for p in ['.gitignore', 'CMakeLists.txt', 'README.md', 'SUBMISSION_CHECKLIST.md', 'requirements.txt', 'requirements-tested-macos.txt',
          'include', 'src', 'tests', 'configs', 'scripts', 'data/metadata',
          'lagrangian', 'report']:
    include(p)
for p in ['docs/data-contract.md', 'docs/numerics.md', 'docs/urban-particle-method.md', 'docs/reproduce-urban.md', 'docs/spatial-aggregation.md', 'docs/multinode-runbook.md', 'docs/data-sources.md']:
    include(p)
for p in ['results/atlas-cpu-1031236.venus01/validation/validation.json',
          'results/atlas-scaling-1031246.venus01/scaling.csv',
          'results/atlas-urban-main-1032157.venus01/review',
          'results/atlas-urban-particles-1032182.venus01/review',
          'results/atlas-urban-pilot-1032143.venus01/review/cropped-wind-audit.json',
          'results/particle-refactor-verification/verification.json',
          'results/atlas-plume-views-1033416.venus01',
          'results/atlas-aggregate-1036035.venus01',
          'data/geometry/00-buildings.stl',
          'results/integrated-parallel-verified/parallel_validation.json',
          'results/integrated-parallel-verified/parallel_validation_report.txt']:
    include(p)
for directory in ['results/atlas-scaling-1031246.venus01',
                  'results/tridiv-update-20261104/vanda-scaling-REVIEW',
                  'results/tridiv-update-20261104/vanda-validation-REVIEW']:
    for f in (root / directory).iterdir():
        if f.is_file() and f.suffix in {'.csv', '.txt', '.json'}:
            include(f.relative_to(root))
for f in (root / 'results/atlas-scaling-1031246.venus01').rglob('*'):
    if f.name in {'timing.csv', 'effective.cfg'}:
        include(f.relative_to(root))
for name in ['building-mask.h5', 'mask-report.json']:
    include(Path('data/geometry') / name)
for name in ['results/lagrangian-urban-regression/validation_report.txt',
             'results/urban-particle-sanitized/validation.json',
             'results/atlas-geometry-1031990.venus01/geometry-report.json',
             'results/atlas-geometry-1031990.venus01/geometry-source.json']:
    include(name)
for f in (root / 'results/tridiv-update-20261104/vanda-scaling-REVIEW/runs').rglob('*'):
    if f.is_file() and f.name in {'timings.csv', 'run_summary.txt', 'rank_counts.csv', 'decomposition.csv'}:
        include(f.relative_to(root))
for review in (root / 'results').glob('atlas-urban-particle-mpi-*/review'):
    include(review.relative_to(root))
for directory in ['results/atlas-multinode-1040644-review', 'results/vanda-multinode-1547560-review']:
    cluster = directory.split('/')[1].split('-')[0]
    job = '1040644.venus01' if cluster == 'atlas' else '1547560.stdct-mgmt-02'
    directory = f'{directory}/results/{cluster}-multinode-{job}'
    include(directory)
    # Placement and measured binding logs are required by the independent review.
    for f in (root / directory).rglob('*.log'):
        files.add(f.relative_to(root))
allowed_markdown = {'README.md', 'SUBMISSION_CHECKLIST.md', 'lagrangian/README.md', 'lagrangian/docs/INTERFACES.md', *['docs/data-contract.md', 'docs/numerics.md', 'docs/urban-particle-method.md', 'docs/reproduce-urban.md', 'docs/spatial-aggregation.md', 'docs/multinode-runbook.md', 'docs/data-sources.md']}
files = {p for p in files if p.suffix.lower() != '.md' or p.as_posix() in allowed_markdown}
archive = root / 'output/dispersion-submission.zip'
archive.parent.mkdir(exist_ok=True)
manifest = []
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        data = (root / path).read_bytes()
        manifest.append(f'{hashlib.sha256(data).hexdigest()}  {path.as_posix()}')
        info = zipfile.ZipInfo('dsa5208-dispersion/' + path.as_posix(), (2026, 11, 15, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o100755 if (root/path).stat().st_mode & 0o111 else 0o100644) << 16
        z.writestr(info, data)
    info = zipfile.ZipInfo('dsa5208-dispersion/SHA256SUMS', (2026, 11, 15, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    z.writestr(info, '\n'.join(manifest) + '\n')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(f'{archive}\n{len(files)} files; {archive.stat().st_size:,} bytes')
