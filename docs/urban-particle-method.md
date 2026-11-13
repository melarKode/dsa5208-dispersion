# Urban particle method

Both solvers use the same prepared case and input checksums. C++ checks binary sizes, finite wind and the binary mask. Geometry comes from `case.cfg`.

## Wind and release

Particle wind is reconstructed from the grid solver's face velocities: average adjacent air-cell velocities, zero at solids/closed faces, interior velocity at open outer faces and periodic averages at periodic faces. Each component varies linearly between its two normal faces. This preserves face fluxes and cell-average discrete divergence. Tangential velocity can be discontinuous at cell boundaries. The input wind is used without divergence correction.

The release samples a Gaussian conditioned on air, matching the grid's cell-integrated Gaussian normalized over air. Rejected samples are retried deterministically using seed, particle ID, attempt and initial stream. The diffusion stream is separate. An explicit error stops a release after 100,000 failed attempts.

## One urban timestep

1. Euler drift uses reconstructed wind at the starting position.
2. A cell-by-cell segment walk follows crossed faces. Solids and closed faces reflect the remaining displacement. Open faces remove drifting particles and add their mass to the escaped budget.
3. Diffusion adds `sqrt(2*K*dt)` times independent standard normal increments. Its walk reflects at solid, closed and open outer faces, matching zero diffusive boundary flux. Periodic faces wrap.
4. MPI transfers each particle to its final spatial owner after the physical step, including multi-subdomain jumps.

Corner encounters use deterministic axis order and a 1e-12-cell nudge. A 100,000-face cap raises an error. Straight-increment reflection and splitting introduce discretization error near complex walls; timestep checks help assess it.

## Parallel implementation and outputs

IDs, positions, image counters, mass and active flags migrate together. Philox counters depend on global particle ID, seed and step, so migration preserves random streams. Every rank loads the full wind/mask; rank zero gathers concentration output. These choices limit memory and I/O scaling.

`velocity_mode=file`, `case_dir`, `boundary_*=closed` and `write_binary=true` select urban behaviour. Binary snapshots store little-endian doubles with x fastest. The shared output format is in `data-contract.md`.

## Recorded experiment

Atlas serial job 1032182 ran three seeds with 100,000 particles at dt=0.1 s, plus one half-timestep and one double-count run. All ran for 600 s with output every 30 s. Job 1033401 replayed seed12345 on 6 MPI ranks x 2 threads; all 21 saved fields matched serial exactly. Diagnostics matched within the specified tolerances. Three seeds give a limited estimate of sampling variation, and changing dt changes the sampled paths.
