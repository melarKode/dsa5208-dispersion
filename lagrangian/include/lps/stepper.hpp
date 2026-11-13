#pragma once

#include <cstdint>

#include "lps/domain.hpp"
#include "lps/particles.hpp"
#include "lps/wind.hpp"

namespace lps {

struct StepParameters {
  std::uint64_t seed = 0;
  double dt = 0.0;           // [s]
  double diffusivity = 0.0;  // K [m^2/s]
};

struct StepResult {
  double escaped_mass = 0.0;           // mass removed at open boundaries during this step
  std::int64_t escaped_particles = 0;
  std::int64_t periodic_wraps = 0;     // number of axis wraps (boundary-crossing diagnostic)
};

// Advances every active particle over step n, from t_n = n*dt to t_{n+1}
// (Euler-Maruyama, exact in distribution for uniform wind and constant K):
//   x_new = x_old + u(x_old, t_n) dt + sqrt(2 K dt) xi,
//   xi = standard_normal3(seed, particle id, n, kStreamDiffusion).
// Each particle's update depends only on its own state, its id and n, so the
// loop can run in any order or in parallel with identical results.
StepResult advance_particles(ParticleSet& particles, const WindField& wind, const Domain& domain,
                             const StepParameters& params, std::uint32_t step_index);

// The per-particle body of advance_particles, shared with the MPI+Kokkos
// solver so both run exactly the same compiled physics. Advances one active
// particle over step n; on return `position` and `image` hold the resolved end
// of the step. When the outcome is Escaped the caller keeps the particle's
// previous position (as advance_particles does). Reads only `wind` and
// `domain`, so concurrent calls for different particles are safe.
StepOutcome step_particle(const WindField& wind, const Domain& domain, const StepParameters& params,
                          std::uint32_t step_index, std::uint64_t id, Vec3& position, std::int32_t image[3]);

}  // namespace lps
