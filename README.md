# DSA5208: parallel pollutant transport

Navneet Melarkode (A0326423L) and Tridiv Swain (A0328390B).

C++17 finite-volume (`src/`) and Lagrangian particle (`lagrangian/`) solvers, each with a serial reference and MPI + Kokkos OpenMP implementation. The report is [report/main.pdf](report/main.pdf). Public source: https://github.com/melarKode/dsa5208-dispersion. The submission ZIP is available on the [Releases page](https://github.com/melarKode/dsa5208-dispersion/releases). Measured CPU results include single-node and 1/2/4-node experiments. GPU performance is not claimed.

Run commands below from the extracted `dsa5208-dispersion/` root unless stated otherwise. Use new output directories for new experiments. Full urban wind and concentration time series remain on Atlas; the package includes source, tests, compact recorded evidence, plotting data, original building STL and the candidate mask.

## 1. Dependencies and Python

Requirements: CMake >=3.23, a C++17 compiler with OpenMP, MPI with C++ support, Git, and Python >=3.10 (3.12 tested). LaTeX is only needed to rebuild the report. Avoid `-ffast-math`. Build both solvers with the same compiler and Kokkos installation.

Select a suitable installed Python; macOS `/usr/bin/python3` may be too old. These commands use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
bash scripts/fetch_kokkos.sh
```

Internet access is required for these dependency downloads. `fetch_kokkos.sh` verifies revision `175257a51ff29a0059ec48bcd233ee096b2c0438` (Kokkos 4.5.01). `requirements-tested-macos.txt` records the complete Python environment used in the final M2 review; install it instead of `requirements.txt` for that exact Python 3.12 environment. Atlas's tested Python pins are in `scripts/atlas_prepare_python.sh`. MPI, compilers and Kokkos are system dependencies, not Python packages.

## 2. Build (choose one platform)

### Linux / cluster allocation

Load your compiler, MPI and CMake environment first. The measured Atlas runs used GCC 13.3.0, Open MPI 5.0.3, CMake 3.29.3 and Kokkos 4.5.1. Supplied Vanda particle benchmarks used Kokkos 4.7.04.

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
  -DDISPERSION_BUILD_PARALLEL=ON -DKokkos_ENABLE_OPENMP=ON
cmake --build build/cpu --parallel 4
ctest --test-dir build/cpu --output-on-failure
cmake -S lagrangian -B build/particles -DCMAKE_BUILD_TYPE=Release \
  -DLPS_PARALLEL=ON -DKokkos_ROOT="$PWD/build/cpu/external/kokkos"
cmake --build build/particles --parallel 4
ctest --test-dir build/particles --output-on-failure
export GRID_BUILD=build/cpu
```

### macOS / M2 Pro

Homebrew CMake, Open MPI and GCC are prerequisites. This machine has `g++-16`; substitute your installed GCC consistently if different.

```bash
DISPERSION_GXX=g++-16 bash scripts/build_local.sh openmp
cmake -S lagrangian -B build/particles -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-16 -DLPS_PARALLEL=ON \
  -DKokkos_ROOT="$PWD/build/openmp/external/kokkos"
cmake --build build/particles --parallel 4
ctest --test-dir build/particles --output-on-failure
export GRID_BUILD=build/openmp
export OMP_PROC_BIND=false
```

Serial-only alternatives require no MPI/Kokkos: `bash scripts/build_local.sh serial` for the grid; configure the particle project without `LPS_PARALLEL` and `Kokkos_ROOT`. The Mac grid helper handles this machine's Apple SDK header discovery. Do not reuse a build directory when changing compiler.

## 3. Small serial and parallel examples

Set `GRID_BUILD` using the matching build section above. The following local MPI examples disable binding for portability; use scheduler-appropriate binding for cluster benchmarks.

```bash
export OMP_NUM_THREADS=2
"$GRID_BUILD/dispersion_serial" configs/gaussian.cfg output_dir=results/example-grid
mpirun --bind-to none -np 2 "$GRID_BUILD/dispersion_mpi" \
  configs/gaussian.cfg output_dir=results/example-grid-mpi
build/particles/lagrangian_solver lagrangian/cases/example_gaussian_uniform.cfg \
  --output results/example-particles
mpirun --bind-to none -np 2 build/particles/lagrangian_parallel \
  lagrangian/cases/example_gaussian_uniform.cfg --output results/example-particles-mpi \
  --kokkos-num-threads=2
```

Grid overrides use `key=value`; particle overrides use named flags. For input formats and urban physics, see `docs/data-contract.md` and `docs/urban-particle-method.md`.

## 4. Correctness checks on a workstation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
OMP_NUM_THREADS=2 OMP_PROC_BIND=false .venv/bin/python tests/validate.py \
  --serial "$GRID_BUILD/dispersion_serial" --parallel "$GRID_BUILD/dispersion_mpi" \
  --output results/grid-validation
.venv/bin/python lagrangian/scripts/validate_parallel.py \
  --serial "$PWD/build/particles/lagrangian_solver" \
  --parallel "$PWD/build/particles/lagrangian_parallel" \
  --mpirun 'mpirun --oversubscribe --bind-to none -n {ranks}' \
  --out results/particle-validation
```

These are local correctness runs, not performance measurements. The grid driver uses up to six MPI ranks by default; `--max-ranks 4` reproduces the rank limit of the recorded Atlas 59-run suite. The particle suite contains 72 configurations and can use up to 16 rank/thread workers. It tests migration, IDs, empty ranks, open escape, walls and output equivalence. On a cluster use an allocation large enough for the chosen workers; the grid driver has `--pbs` to forward the runtime environment. Do not oversubscribe for performance measurements.

## 5. Atlas runs

```bash
bash scripts/atlas_prepare_python.sh
qsub scripts/atlas_all_cpu.pbs
```

Wait for `SUCCESS: CPU validation complete` in that job's log **before** submitting the scaling job, because scaling needs its build:

```bash
qsub scripts/atlas_scaling.pbs
```

Batch jobs survive SSH disconnection. Use `qstat -u "$USER"` and `qstat -xf JOBID` for finished jobs. Logs are in `logs/`, results in `results/`; disappearance from the queue does not establish success. `scripts/atlas_env.sh` retains the tested module/libfabric workaround for this Atlas installation.

For urban reproduction follow `docs/reproduce-urban.md`. Historical job IDs in the batch scripts identify recorded upstream inputs; replace them with your new run paths. The wind download belongs on the cluster. No full urban rerun on the Mac is required to rebuild the report.

### Multi-node CPU results

The final multi-node evidence is Atlas job 1040644 and Vanda job 1547560. Each uses 1/2/4 nodes, 12 CPU cores per node, three repetitions and TCP transport. Node-relative baselines are kept separate from historical serial-relative scaling. `scripts/audit_report_evidence.py` also recomputes the multi-node field comparisons, bindings and timing summaries. See `docs/multinode-runbook.md` for cluster commands and `report/evidence-audit.json` for the computed checks. The new 3D grid/partition diagram is schematic; urban figures use actual saved data.

## 6. Rebuild figures and report from bundled evidence

These commands do not simulate new results. The original STL render can take about a minute and requires more memory than the 2D plots.

```bash
export MPLCONFIGDIR=/tmp/dispersion-mpl
.venv/bin/python scripts/audit_report_evidence.py
.venv/bin/python scripts/make_report_figures.py
.venv/bin/python scripts/plot_urban_plume.py results/atlas-plume-views-1033416.venus01
.venv/bin/python scripts/plot_urban_aggregate.py results/atlas-aggregate-1036035.venus01 \
  --output report/figures
.venv/bin/python scripts/plot_urban_angles.py
.venv/bin/python scripts/plot_grid_partition.py
.venv/bin/python scripts/plot_multinode_review.py \
  results/vanda-multinode-1547560-review/results/vanda-multinode-1547560.stdct-mgmt-02 \
  --cluster Vanda --output-prefix report/figures/multinode_vanda
.venv/bin/python scripts/plot_multinode_review.py \
  results/atlas-multinode-1040644-review/results/atlas-multinode-1040644.venus01 \
  --cluster Atlas --output-prefix report/figures/multinode_atlas
(cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex)
```

The LaTeX uses article, geometry, lmodern, amsmath/amssymb, booktabs, graphicx, float, hyperref, microtype and caption. Generated figures are already included, so LaTeX alone is sufficient if you do not need to regenerate plots.

The 3D urban illustration projects height-integrated concentration beneath STL buildings. Spatial aggregation uses shared conservative bins; definitions are in `docs/spatial-aggregation.md`. No physical forecast or fine-cell convergence is claimed.

## 7. Package and submit

```bash
python3 scripts/package_submission.py
```

The resulting `output/dispersion-submission.zip` includes `SHA256SUMS` and can be repackaged from its own extracted contents. It excludes builds, virtual environments, Kokkos source, full urban wind and full 3D time series. Build dependencies are downloaded separately. Submit the PDF with its public source link by 17 November 2026. The ZIP is an additional downloadable copy. See `SUBMISSION_CHECKLIST.md` for the remaining author and upload checks. The contributions and AI disclosure are in the PDF. Dataset attribution is in `docs/data-sources.md`.
