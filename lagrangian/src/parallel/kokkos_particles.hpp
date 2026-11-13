#pragma once

// Node-level (Kokkos) part of the MPI+Kokkos solver: particle storage in
// Views and the parallel kernels. No MPI here.
//
// CPU first: kernels run on Kokkos::DefaultHostExecutionSpace (OpenMP) and
// call the serial reference physics (lps::step_particle, lps::release_particle,
// the virtual WindField/Domain) directly, so serial and parallel runs execute
// the same compiled per-particle code. A GPU build needs device-callable wind
// and domain objects; see docs/mpi-kokkos-interfaces.md.

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "lps/config.hpp"
#include "lps/decomposition.hpp"
#include "lps/domain.hpp"
#include "lps/grid.hpp"
#include "lps/particles.hpp"
#include "lps/stepper.hpp"
#include "lps/types.hpp"
#include "lps/wind.hpp"

namespace lps::par {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = Kokkos::HostSpace;
using Index = std::int64_t;
using Policy = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>>;
template <class T>
using Array = Kokkos::View<T*, MemSpace>;

// Structure-of-arrays storage with the same fields as lps::ParticleSet.
// Entries [0, size) are valid. Escaped particles stay stored with active = 0
// (as in the serial solver) and are never migrated.
struct ParticleArrays {
  Array<std::uint64_t> id;
  Array<double> x, y, z;
  Array<std::int32_t> image_x, image_y, image_z;
  Array<double> mass;
  Array<std::uint8_t> active;
  std::size_t size = 0;

  explicit ParticleArrays(const std::string& label = "particles");
  std::size_t capacity() const { return id.extent(0); }
  // Ensures capacity >= n with geometric growth; `preserve` keeps [0, size).
  void reserve(std::size_t n, bool preserve = true);
};

// One particle as a contiguous record, for MPI messages.
struct PackedParticle {
  std::uint64_t id;
  double x, y, z, mass;
  std::int32_t image[3];
  std::uint8_t active;
  std::uint8_t padding[3];
};
static_assert(sizeof(PackedParticle) == 56, "PackedParticle layout changed");
static_assert(std::is_trivially_copyable<PackedParticle>::value, "PackedParticle must be trivially copyable");

KOKKOS_INLINE_FUNCTION PackedParticle pack_particle(const ParticleArrays& ps, Index p) {
  PackedParticle q{};
  q.id = ps.id(p);
  q.x = ps.x(p);
  q.y = ps.y(p);
  q.z = ps.z(p);
  q.mass = ps.mass(p);
  q.image[0] = ps.image_x(p);
  q.image[1] = ps.image_y(p);
  q.image[2] = ps.image_z(p);
  q.active = ps.active(p);
  return q;
}

KOKKOS_INLINE_FUNCTION void unpack_particle(const PackedParticle& q, const ParticleArrays& ps, Index p) {
  ps.id(p) = q.id;
  ps.x(p) = q.x;
  ps.y(p) = q.y;
  ps.z(p) = q.z;
  ps.mass(p) = q.mass;
  ps.image_x(p) = q.image[0];
  ps.image_y(p) = q.image[1];
  ps.image_z(p) = q.image[2];
  ps.active(p) = q.active;
}

KOKKOS_INLINE_FUNCTION void copy_particle(const ParticleArrays& from, Index p, const ParticleArrays& to, Index q) {
  unpack_particle(pack_particle(from, p), to, q);
}

// SplitMix64 finaliser. Sum over stored ids (mod 2^64) is a multiset
// fingerprint: any lost or duplicated id changes it with probability 1 - 2^-64.
KOKKOS_INLINE_FUNCTION std::uint64_t id_fingerprint(std::uint64_t id) {
  std::uint64_t z = id + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Exceptions thrown by the physics inside a parallel region cannot cross
// thread boundaries, so kernels catch them per particle, count them in the
// reduction and keep the first message here; raise_if_any rethrows on the host.
class KernelErrors {
 public:
  void record(std::uint64_t particle_id, const char* what);
  void raise_if_any(std::int64_t count, const char* kernel) const;

 private:
  mutable std::mutex mutex_;
  bool has_error_ = false;
  std::uint64_t particle_id_ = 0;
  std::string message_;
};

// Escaped mass (compensated), counts and errors from one kernel.
struct Tally {
  double escaped_mass = 0.0;
  double escaped_mass_correction = 0.0;
  std::int64_t escaped_particles = 0;
  std::int64_t periodic_wraps = 0;
  std::int64_t errors = 0;

  KOKKOS_INLINE_FUNCTION void add_mass(double m) {
    CompensatedSum s{escaped_mass, escaped_mass_correction};
    s.add(m);
    escaped_mass = s.sum;
    escaped_mass_correction = s.correction;
  }
  KOKKOS_INLINE_FUNCTION void merge(const Tally& other) {
    add_mass(other.escaped_mass);
    add_mass(other.escaped_mass_correction);
    escaped_particles += other.escaped_particles;
    periodic_wraps += other.periodic_wraps;
    errors += other.errors;
  }
  double mass() const { return escaped_mass + escaped_mass_correction; }
};

// Samples ids [first, last) with lps::release_particle and appends them.
Tally release_block(ParticleArrays& ps, const Config& config, const Domain& domain, std::uint64_t first,
                    std::uint64_t last);

// Advances every stored active particle over step `step_index` with lps::step_particle.
Tally advance(ParticleArrays& ps, const WindField& wind, const Domain& domain, const StepParameters& params,
              std::uint32_t step_index);

// Local sums for the diagnostics (unwrapped positions, mass-weighted).
struct MomentSums {
  double mass = 0.0, mass_correction = 0.0;
  double mx = 0.0, my = 0.0, mz = 0.0;
  std::int64_t active = 0;
};
MomentSums first_moments(const ParticleArrays& ps, const Grid& grid);
Vec3 second_moments(const ParticleArrays& ps, const Grid& grid, const Vec3& mean);  // sum m (x - mean)^2

// Counts and id fingerprint of the stored particles.
struct Inventory {
  std::int64_t stored = 0;
  std::int64_t active = 0;
  std::int64_t off_home_image = 0;
  std::uint64_t fingerprint = 0;
};
Inventory take_inventory(const ParticleArrays& ps);

// Nearest-grid-point mass deposit into the rank's own cells, x-fastest within
// `box`. Each cell is a compensated sum, as in lps::deposit_mass_ngp. Throws if
// an active particle lies outside the box (it should have been migrated).
std::vector<double> deposit_box(const ParticleArrays& ps, const Grid& grid, const Decomposition::Box& box);

}  // namespace lps::par
