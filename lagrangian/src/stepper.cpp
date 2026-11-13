#include "lps/stepper.hpp"

#include <cmath>

#include "lps/rng.hpp"

namespace lps {

StepOutcome step_particle(const WindField& wind, const Domain& domain, const StepParameters& params,
                          std::uint32_t step_index, std::uint64_t id, Vec3& position, std::int32_t image[3]) {
  const double time = static_cast<double>(step_index) * params.dt;
  const double noise_scale = std::sqrt(2.0 * params.diffusivity * params.dt);
  const bool diffusive = params.diffusivity > 0.0;

  const Vec3 start = position;
  Vec3 end = start + params.dt * wind.velocity(start, time);
  if (diffusive && !domain.has_geometry()) {
    end = end + noise_scale * standard_normal3(params.seed, id, step_index, kStreamDiffusion);
  }

  StepOutcome outcome = domain.resolve_step(start, end, image);
  if (outcome == StepOutcome::Active && diffusive && domain.has_geometry()) {
    const Vec3 drift_end=end;
    end=end+noise_scale*standard_normal3(params.seed,id,step_index,kStreamDiffusion);
    outcome=domain.resolve_step(drift_end,end,image,true);
  }
  position = end;
  return outcome;
}

StepResult advance_particles(ParticleSet& ps, const WindField& wind, const Domain& domain,
                             const StepParameters& params, std::uint32_t step_index) {
  StepResult result;
  CompensatedSum escaped;
  for (std::size_t p = 0; p < ps.size(); ++p) {
    if (!ps.active[p]) continue;

    Vec3 end = ps.position(p);
    std::int32_t image[3] = {ps.image_x[p], ps.image_y[p], ps.image_z[p]};
    const StepOutcome outcome = step_particle(wind, domain, params, step_index, ps.id[p], end, image);
    result.periodic_wraps += std::abs(image[0] - ps.image_x[p]) + std::abs(image[1] - ps.image_y[p]) +
                             std::abs(image[2] - ps.image_z[p]);

    if (outcome == StepOutcome::Escaped) {
      ps.active[p] = 0;
      escaped.add(ps.mass[p]);
      ++result.escaped_particles;
      continue;
    }
    ps.x[p] = end.x;
    ps.y[p] = end.y;
    ps.z[p] = end.z;
    ps.image_x[p] = image[0];
    ps.image_y[p] = image[1];
    ps.image_z[p] = image[2];
  }
  result.escaped_mass = escaped.result();
  return result;
}

}  // namespace lps
