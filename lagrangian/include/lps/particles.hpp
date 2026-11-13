#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lps/config.hpp"
#include "lps/domain.hpp"
#include "lps/types.hpp"

namespace lps {

// Structure-of-arrays particle storage (maps directly onto Kokkos::Views).
// A particle is identified by its persistent `id`, never by its array slot:
// slots will change when particles migrate between MPI ranks.
struct ParticleSet {
  std::vector<std::uint64_t> id;
  std::vector<double> x, y, z;                          // position inside the domain [m]
  std::vector<std::int32_t> image_x, image_y, image_z;  // periodic image counters
  std::vector<double> mass;
  std::vector<std::uint8_t> active;  // 0 once removed through an open boundary

  std::size_t size() const { return id.size(); }
  void resize(std::size_t n);

  Vec3 position(std::size_t p) const { return {x[p], y[p], z[p]}; }
  Vec3 unwrapped_position(std::size_t p, const Grid& grid) const {
    return {x[p] + image_x[p] * grid.length_x(), y[p] + image_y[p] * grid.length_y(),
            z[p] + image_z[p] * grid.length_z()};
  }
};

struct ReleaseResult {
  ParticleSet particles;
  double escaped_mass = 0.0;  // mass sampled outside an open boundary at t = 0
  std::int64_t escaped_particles = 0;
};

// Instantaneous isotropic Gaussian release: particle `id` (0..N-1) starts at
//   source + source_sigma * xi(seed, id, step 0, kStreamInitialPosition)
// with mass initial_mass / N. Samples are then mapped onto the domain:
// periodic axes wrap (image counters record it), and samples outside an open
// axis are marked inactive and counted as escaped.
ReleaseResult make_gaussian_release(const Config& config, const Domain& domain);

// One released particle: the per-id body of make_gaussian_release, shared
// with the MPI+Kokkos solver (each rank samples its own block of ids).
struct ReleasedParticle {
  Vec3 position;
  std::int32_t image[3] = {0, 0, 0};
  bool active = true;  // false: sampled outside an open boundary (counts as escaped)
};

// Samples particle `id` exactly as make_gaussian_release does. A pure function
// of (config, domain, id); throws if a conditioned release finds no air.
ReleasedParticle release_particle(const Config& config, const Domain& domain, std::uint64_t id);

}  // namespace lps
