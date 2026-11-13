# Particle interfaces

## Configuration and physics

The input is `key=value` text. `--set key=value` overrides settings. Uniform cases use one displacement `u*dt + sqrt(2*K*dt)*normal`. File-driven urban cases use split drift/diffusion and face traversal as described in `../../docs/urban-particle-method.md`.

Each particle carries a global ID, position, periodic image counters, mass and active flag. Initial mass is `initial_mass/particle_count` per particle. Global IDs remain stable across migration.

## Random streams

Philox4x32-10 uses key `{seed_lo,seed_hi}` and counter `{id_lo,id_hi,step,stream}`. Box–Muller transforms the words into normal increments. Initial release and diffusion use separate streams. Reordering particles or moving them to another rank preserves their random increments.

## MPI and Kokkos

Kokkos updates the local structure-of-arrays particle storage. Cartesian decomposition determines spatial owners. After stepping, `MPI_Alltoallv` transfers complete particle states to final owners, including multi-subdomain jumps. MPI rank boundaries transfer ownership; physical boundary rules are applied by the stepper.

Counts and ID fingerprints provide runtime checks; validation dumps check every ID exactly once. Local reductions feed global diagnostics. All ranks retain the full wind/mask and rank zero gathers concentration output.

## Output and timing

The shared concentration and diagnostics layout is in `../../docs/data-contract.md`. `write_binary=true` selects little-endian doubles with x fastest. Otherwise CSV snapshots remain available.

Parallel `timings.csv` records setup, stepping, waiting, migration, output, total and loop times per rank. The scaling metric is the largest per-rank loop time. Maxima of individual categories can come from different ranks and must be interpreted separately. The serial `run_summary.txt` records wall and output times.
