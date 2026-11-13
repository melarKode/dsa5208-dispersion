#include "lps/particles.hpp"

#include "lps/rng.hpp"
#include <stdexcept>

namespace lps {

void ParticleSet::resize(std::size_t n) {
  id.resize(n);
  x.resize(n);
  y.resize(n);
  z.resize(n);
  image_x.resize(n);
  image_y.resize(n);
  image_z.resize(n);
  mass.resize(n);
  active.resize(n);
}

ReleasedParticle release_particle(const Config& config, const Domain& domain, std::uint64_t id) {
  const Vec3 centre{config.source_x, config.source_y, config.source_z};
  ReleasedParticle released;
  const Vec3 xi = standard_normal3(config.seed, id, 0, kStreamInitialPosition);
  Vec3 pos = centre + config.source_sigma * xi;
  StepOutcome outcome;
  if(domain.has_geometry()) {
    bool accepted=false;
    for(std::uint32_t attempt=0;attempt<100000;++attempt) {
      pos=centre+config.source_sigma*standard_normal3(config.seed,id,attempt,kStreamInitialPosition);
      int i,j,k;
      if(domain.grid().locate(pos,i,j,k) && !domain.is_solid_cell(domain.grid().cell_id(i,j,k))) {accepted=true;break;}
    }
    if(!accepted) throw std::runtime_error("Cannot sample Gaussian in air");
    outcome=StepOutcome::Active;
  } else outcome=domain.apply_boundaries(pos,released.image);
  released.position = pos;
  released.active = outcome == StepOutcome::Active;
  return released;
}

ReleaseResult make_gaussian_release(const Config& config, const Domain& domain) {
  ReleaseResult result;
  ParticleSet& ps = result.particles;
  const std::size_t n = static_cast<std::size_t>(config.particle_count);
  ps.resize(n);

  const double particle_mass = config.initial_mass / static_cast<double>(config.particle_count);
  CompensatedSum escaped;

  for (std::size_t p = 0; p < n; ++p) {
    const std::uint64_t id = p;
    const ReleasedParticle released = release_particle(config, domain, id);

    ps.id[p] = id;
    ps.x[p] = released.position.x;
    ps.y[p] = released.position.y;
    ps.z[p] = released.position.z;
    ps.image_x[p] = released.image[0];
    ps.image_y[p] = released.image[1];
    ps.image_z[p] = released.image[2];
    ps.mass[p] = particle_mass;
    ps.active[p] = released.active ? 1 : 0;
    if (!released.active) {
      escaped.add(particle_mass);
      ++result.escaped_particles;
    }
  }
  result.escaped_mass = escaped.result();
  return result;
}

}  // namespace lps
