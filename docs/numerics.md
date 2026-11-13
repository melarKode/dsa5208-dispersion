# Grid transport model and numerical method

## Equation and representation

Solve `dc/dt + div(u c) = K laplacian(c)` with prescribed, time-independent wind and spatially constant K. Concentration is a cell average in mass/m³. Geometry is a Cartesian grid with whole-cell solid obstacles. Coordinates use the lower domain corner as origin. Time is seconds; lengths are metres.

The initial Gaussian is integrated over each cell using error functions. Periodic cases use a wrapped Gaussian (image sum truncated beyond approximately eight standard deviations). Solid cells receive zero mass; the remaining air-cell mass is normalized to `initial_mass`. Thus a cloud intersecting a wall is a truncated, renormalized cloud, not an unmodified free-space Gaussian. Select an air-contained source region for the urban comparison and use the same normalized initial distribution for particles.

`initial=cosine` is a periodic manufactured benchmark, not an emission model. `initial=uniform` initializes uniform air-cell concentration.

## Conservative flux and timestep

For a face from cell P toward neighbor N, let `u_n` be outward normal velocity and `h` the spacing. The outward concentration flux is

`F = max(u_n,0)c_P + min(u_n,0)c_N - K(c_N-c_P)/h`.

Divide by h to obtain its contribution to `dc_P/dt`. Adjacent cells use the same face velocity with opposite normals, so internal contributions cancel in total mass. File velocities are cell-centred; face velocity is their arithmetic average. Uniform winds use their prescribed components directly.

Precompute six nonnegative incoming rates and the outgoing diagonal. The forward Euler update is `c_new = c_old*(1-dt*loss) + dt*sum(incoming*c_neighbor)`. All right-hand terms use the old field. A combined positivity bound is `dt <= cfl / max(loss)` with `0<cfl<=1`. This includes advection and diffusion together. `dt=0` selects the automatic bound; a positive unsafe dt is rejected. Steps shorten to land exactly on output times and the final time. No negative-value clipping is performed during stepping.

First-order upwind introduces numerical diffusion. Analytical tests report the resulting spatial error; a matching grid and particle observation resolution does not imply matching accuracy.

## Boundaries and solids

- Periodic: connect opposite sides; MPI Cartesian topology and local z halos implement the same rule.
- Closed: zero advective and diffusive normal flux, even if the input velocity points through that boundary.
- Open: zero external incoming pollutant, outgoing advective flux, and zero diffusive normal gradient. Escaped mass is accumulated from the same old-time advective flux used in the update. This is not an absorbing diffusion boundary.
- Solids: all fluxes between air and solid cells are zero; the entire solid cell remains empty. Thin buildings narrower than a cell are not resolved.

The particle solver matches these choices with separate drift and diffusion steps; see `urban-particle-method.md`.

Masking face velocities at walls can introduce divergence. The solver conserves scalar mass even with divergent prescribed velocity, but this does not make that velocity a physically valid incompressible airflow. `audit_case.py` measures the same face reconstruction and masking used in the solver. No pressure projection or wind correction is implemented; urban acceptance depends on inspecting these diagnostics and the source dataset's meaning.

## Parallel design

The serial executable has independent C++ update loops. It shares configuration, geometry, and coefficient construction with the parallel executable; analytical tests therefore provide independent numerical evidence beyond serial/parallel equivalence.

MPI divides x/y and retains full z. Slabs split x only; blocks choose an aspect-balanced factorization. Explicit `px,py` are supported. Uneven partitions and partitions one cell wide are valid. Each rank reads only its own case slab plus one-cell halos.

Kokkos executes the update and halo packing. Blocking exchange uses paired `MPI_Sendrecv`. Overlap posts receives/sends, updates cells independent of x/y halos, waits, unpacks, then updates boundary cells. Actual asynchronous progress is MPI-dependent; nonblocking calls do not guarantee speedup. Corners are unnecessary for the seven-point stencil.

CUDA uses the same kernels and double precision. MPI buffers are staged through host memory, avoiding a dependency on GPU-aware MPI. On CUDA, local MPI rank selects a visible device modulo the visible device count; request one rank per allocated GPU for benchmark runs. The reported runs use CPU/OpenMP, including 1/2/4-node measurements on Atlas and Vanda. GPU performance has not been measured.

## Diagnostics and timing

Mass residual is `initial_mass - mass_inside - mass_escaped`. Outputs also include min/max concentration and mass-weighted cell-centre moments. Moments are ordinary Cartesian moments, not unwrapped periodic moments; do not interpret them as plume spread after boundary wrapping. Cell-centre moments also carry discretization error relative to continuous moments.

`solve_seconds` is the maximum across ranks of accumulated step wall time, including halo handling and escape reductions but excluding output and setup. `elapsed_seconds` includes initial/final diagnostics and requested outputs, but excludes model construction. `update_seconds`, `halo_seconds`, `exposed_wait_seconds`, and `output_seconds` are per-component rank maxima; these maxima need not sum to elapsed time. Exposed wait is a subset of halo time. Host peak RSS is the maximum process peak reported by getrusage, not total cluster memory or device memory.

For scaling, keep the configuration and step count fixed, record actual host bindings, repeat runs, and compare measured serial reference times. Keep Atlas and Vanda baselines separate. M2 runs check correctness and builds.
