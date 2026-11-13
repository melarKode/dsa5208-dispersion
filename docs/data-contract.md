# Shared grid/particle contract

Both solvers use the prepared format below. The urban experiment uses an STL-derived candidate mask; preparation and provenance are described in `reproduce-urban.md`.

## Input case

A case directory contains:

- `case.cfg`: nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z; each a key=value line.
- `u.bin`, `v.bin`, `w.bin`: cell-centred velocity components, little-endian float64.
- `solid.bin`: uint8 values, 1 solid and 0 air.
- `provenance.json`: explicit upstream field selection, axes, coordinates, mask interpretation.
- `input.sha256`: checksums of geometry and four binary fields.

Array shape in Python is `(nz, ny, nx)` in C order. Cell index is `i + nx*(j + ny*k)`. Cell centres are `origin + (index+0.5)*spacing`. Verify whether upstream coordinates are points or centres before choosing origin and spacing. Conversion does not rotate velocity components: u/v/w must already correspond to physical x/y/z. `--axis-order` only reorders storage axes.

`prepare_case.py` accepts separate 3D components or a selected vector dataset, an optional leading case index, and explicit coordinate/mask information. It streams z planes. A static 3D mask can accompany time/case-selected wind; a 4D mask uses the same case selection. Only exact binary mask values specified by `--air-value` and `--solid-value` are accepted. Cases without obstacles require explicit `--all-fluid`. Do not use that option to bypass missing urban geometry. No coordinate interpolation, STL rasterization, or time-varying wind is silently inferred.

Run `sha256sum -c input.sha256` after transferring a prepared case. Never modify case files in place after generating their manifest.

## Run configuration

Plain key=value text; `#` begins a comment. Unknown and duplicate keys fail. `case_dir` resolves relative to the configuration file; the case geometry cannot be overridden inconsistently. `output_dir` resolves relative to the launch working directory. Command-line `key=value` arguments override the run file. Output directories must be new or empty.

Physics keys: nx,ny,nz; dx,dy,dz; origin_x,origin_y,origin_z; u,v,w; diffusivity; initial_mass; source_x,source_y,source_z; source_sigma; initial; boundary_x,boundary_y,boundary_z; dt; cfl; end_time; output_interval; velocity_mode; case_dir; wall_i; amplitude.

Runtime keys: output_dir; snapshots (`binary`, `csv`, `none`); decomposition (`slab`, `block`); exchange (`blocking`, `overlap`); px,py. Shared particle keys `particle_count` and `seed`, and `case_name`, are accepted but ignored by grid transport. `initial` defaults to gaussian, `wall_i=-1` means no synthetic wall, and `velocity_mode` defaults to uniform.

## Outputs

- `effective.cfg`: resolved configuration, including defaults.
- `diagnostics.csv`: time,active_particles,mass_inside,mass_escaped,mass_residual,mean_x,mean_y,mean_z,var_x,var_y,var_z,min_concentration,max_concentration. `active_particles=nan` for grid outputs. Additional particle-only columns are fine.
- `snapshots.csv`: index,time,format,ranks. Particle outputs may omit format/ranks when writing one global CSV; the reader defaults to csv/1.
- `concentration_000000.csv`: cell_id,i,j,k,mass,concentration; all cells, zero-based indices. `mass=concentration*cell_volume`.
- Binary alternative: `concentration_000000.bin`, little-endian float64, concentration only, x fastest.
- Multi-rank outputs append `_rankR` before the extension. `domain_rankR.csv` supplies rank,x0,y0,nx,ny,nz. Each rank writes only its interior, without halos.
- `input.sha256` is copied into file-driven runs for matching input provenance.
- `timing.csv`: runtime/resource/algorithm information for the grid implementation.

Solids have zero concentration. Initial output is index 0 at time zero. Always include the requested final time. A particle implementation can use a different integration dt, but compare snapshots at identical physical times. Report error as particle count increases and across seeds. Grid/particle equality is statistical, not bitwise.

`compare.py` checks common physics and compares volume-weighted L1/L2 and maximum concentration errors. Urban comparisons require matching input manifests. It does not by itself establish equal interpolation rules, correct particle wall behavior, or statistically equivalent accuracy.

## Shared boundary treatment

Both methods use the same face velocities, Gaussian release normalized over air, and zero diffusive flux at open boundaries. Particle drift can escape there; diffusion reflects. See `urban-particle-method.md` for the interpolation and collision rules.
