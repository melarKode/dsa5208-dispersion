# Lagrangian particle solver

C++17 serial and MPI + Kokkos OpenMP implementations of passive pollutant transport. Tridiv Swain led the particle solver, validation and original Vanda benchmarks.

Use the [project README](../README.md) for the tested build and MPI commands. The serial executable is `lagrangian_solver`; the parallel executable is `lagrangian_parallel`.

```bash
# From the project root after building:
build/particles/lagrangian_solver lagrangian/cases/example_gaussian_uniform.cfg --output results/particle-example
OMP_NUM_THREADS=2 mpirun --bind-to none -np 2 build/particles/lagrangian_parallel \
  lagrangian/cases/example_gaussian_uniform.cfg --output results/particle-example-mpi --kokkos-num-threads=2
```

`--set key=value` overrides configuration values. `effective_config.txt` records the resolved settings. Outputs include diagnostics, snapshots, a run summary and optional per-particle states.

## Validation

`ctest --test-dir build/particles --output-on-failure` runs unit, urban and decomposition checks. `scripts/validate.py` in this directory runs analytical serial cases. `scripts/validate_parallel.py` compares serial/MPI per-ID states across 72 configurations; full commands are in the root README.

## Files

- `include/lps/`, `src/`: configuration, grid, random streams, wind, particles, boundaries and MPI ownership.
- `cases/`: examples and validation configurations.
- `tests/`, `scripts/`: unit and executable-level validation.
- [Interfaces](docs/INTERFACES.md), [shared format](../docs/data-contract.md), [urban stepping](../docs/urban-particle-method.md).
