#include "kokkos_particles.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>

namespace lps::par {

ParticleArrays::ParticleArrays(const std::string& label)
    : id(label + ".id", 0),
      x(label + ".x", 0),
      y(label + ".y", 0),
      z(label + ".z", 0),
      image_x(label + ".image_x", 0),
      image_y(label + ".image_y", 0),
      image_z(label + ".image_z", 0),
      mass(label + ".mass", 0),
      active(label + ".active", 0) {}

void ParticleArrays::reserve(std::size_t n, bool preserve) {
  if (n <= capacity()) return;
  const std::size_t new_capacity = std::max(n, capacity() + capacity() / 2 + 64);
  const auto grow = [&](auto& view) {
    if (preserve)
      Kokkos::resize(Kokkos::WithoutInitializing, view, new_capacity);
    else
      Kokkos::realloc(Kokkos::WithoutInitializing, view, new_capacity);
  };
  grow(id);
  grow(x);
  grow(y);
  grow(z);
  grow(image_x);
  grow(image_y);
  grow(image_z);
  grow(mass);
  grow(active);
}

void KernelErrors::record(std::uint64_t particle_id, const char* what) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_error_ && particle_id >= particle_id_) return;  // keep the lowest id: deterministic message
  has_error_ = true;
  particle_id_ = particle_id;
  message_ = what;
}

void KernelErrors::raise_if_any(std::int64_t count, const char* kernel) const {
  if (count == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  throw std::runtime_error(std::string(kernel) + ": " + std::to_string(count) + " particle(s) failed; particle " +
                           std::to_string(particle_id_) + ": " + message_);
}

namespace {

struct ReleaseFunctor {
  using value_type = Tally;
  ParticleArrays out;
  const Config* config;
  const Domain* domain;
  KernelErrors* errors;
  std::uint64_t first_id;
  Index base;
  double particle_mass;

  void operator()(const Index q, Tally& tally) const {
    const std::uint64_t id = first_id + static_cast<std::uint64_t>(q);
    const Index p = base + q;
    ReleasedParticle released;
    try {
      released = release_particle(*config, *domain, id);
    } catch (const std::exception& error) {
      errors->record(id, error.what());
      ++tally.errors;
      released.active = false;  // keep the slot well-defined; the run aborts after the kernel
    }
    out.id(p) = id;
    out.x(p) = released.position.x;
    out.y(p) = released.position.y;
    out.z(p) = released.position.z;
    out.image_x(p) = released.image[0];
    out.image_y(p) = released.image[1];
    out.image_z(p) = released.image[2];
    out.mass(p) = particle_mass;
    out.active(p) = released.active ? 1 : 0;
    if (!released.active) {
      tally.add_mass(particle_mass);
      ++tally.escaped_particles;
    }
  }
  void init(Tally& tally) const { tally = Tally{}; }
  void join(Tally& into, const Tally& from) const { into.merge(from); }
};

struct StepFunctor {
  using value_type = Tally;
  ParticleArrays ps;
  const WindField* wind;
  const Domain* domain;
  KernelErrors* errors;
  StepParameters params;
  std::uint32_t step_index;

  void operator()(const Index p, Tally& tally) const {
    if (!ps.active(p)) return;
    Vec3 end{ps.x(p), ps.y(p), ps.z(p)};
    std::int32_t image[3] = {ps.image_x(p), ps.image_y(p), ps.image_z(p)};
    StepOutcome outcome;
    try {
      outcome = step_particle(*wind, *domain, params, step_index, ps.id(p), end, image);
    } catch (const std::exception& error) {
      errors->record(ps.id(p), error.what());
      ++tally.errors;
      return;
    }
    tally.periodic_wraps += std::abs(image[0] - ps.image_x(p)) + std::abs(image[1] - ps.image_y(p)) +
                            std::abs(image[2] - ps.image_z(p));
    if (outcome == StepOutcome::Escaped) {
      ps.active(p) = 0;
      tally.add_mass(ps.mass(p));
      ++tally.escaped_particles;
      return;
    }
    ps.x(p) = end.x;
    ps.y(p) = end.y;
    ps.z(p) = end.z;
    ps.image_x(p) = image[0];
    ps.image_y(p) = image[1];
    ps.image_z(p) = image[2];
  }
  void init(Tally& tally) const { tally = Tally{}; }
  void join(Tally& into, const Tally& from) const { into.merge(from); }
};

struct FirstMomentFunctor {
  using value_type = MomentSums;
  ParticleArrays ps;
  double length[3];

  void operator()(const Index p, MomentSums& s) const {
    if (!ps.active(p)) return;
    const double m = ps.mass(p);
    CompensatedSum mass{s.mass, s.mass_correction};
    mass.add(m);
    s.mass = mass.sum;
    s.mass_correction = mass.correction;
    s.mx += m * (ps.x(p) + ps.image_x(p) * length[0]);
    s.my += m * (ps.y(p) + ps.image_y(p) * length[1]);
    s.mz += m * (ps.z(p) + ps.image_z(p) * length[2]);
    ++s.active;
  }
  void init(MomentSums& s) const { s = MomentSums{}; }
  void join(MomentSums& into, const MomentSums& from) const {
    CompensatedSum mass{into.mass, into.mass_correction};
    mass.add(from.mass);
    mass.add(from.mass_correction);
    into.mass = mass.sum;
    into.mass_correction = mass.correction;
    into.mx += from.mx;
    into.my += from.my;
    into.mz += from.mz;
    into.active += from.active;
  }
};

struct SecondMomentFunctor {
  using value_type = Vec3;
  ParticleArrays ps;
  double length[3];
  Vec3 mean;

  void operator()(const Index p, Vec3& s) const {
    if (!ps.active(p)) return;
    const double m = ps.mass(p);
    const double dx = ps.x(p) + ps.image_x(p) * length[0] - mean.x;
    const double dy = ps.y(p) + ps.image_y(p) * length[1] - mean.y;
    const double dz = ps.z(p) + ps.image_z(p) * length[2] - mean.z;
    s.x += m * dx * dx;
    s.y += m * dy * dy;
    s.z += m * dz * dz;
  }
  void init(Vec3& s) const { s = Vec3{}; }
  void join(Vec3& into, const Vec3& from) const { into = into + from; }
};

struct InventoryFunctor {
  using value_type = Inventory;
  ParticleArrays ps;

  void operator()(const Index p, Inventory& inv) const {
    ++inv.stored;
    inv.active += ps.active(p);
    if (ps.image_x(p) != 0 || ps.image_y(p) != 0 || ps.image_z(p) != 0) ++inv.off_home_image;
    inv.fingerprint += id_fingerprint(ps.id(p));
  }
  void init(Inventory& inv) const { inv = Inventory{}; }
  void join(Inventory& into, const Inventory& from) const {
    into.stored += from.stored;
    into.active += from.active;
    into.off_home_image += from.off_home_image;
    into.fingerprint += from.fingerprint;
  }
};

}  // namespace

Tally release_block(ParticleArrays& ps, const Config& config, const Domain& domain, std::uint64_t first,
                    std::uint64_t last) {
  const Index count = static_cast<Index>(last - first);
  const Index base = static_cast<Index>(ps.size);
  ps.reserve(ps.size + static_cast<std::size_t>(count));
  KernelErrors errors;
  const ReleaseFunctor functor{ps,    &config, &domain, &errors, first,
                               base, config.initial_mass / static_cast<double>(config.particle_count)};
  Tally tally;
  Kokkos::parallel_reduce("lps::release", Policy(0, count), functor, tally);
  errors.raise_if_any(tally.errors, "release");
  ps.size += static_cast<std::size_t>(count);
  return tally;
}

Tally advance(ParticleArrays& ps, const WindField& wind, const Domain& domain, const StepParameters& params,
              std::uint32_t step_index) {
  KernelErrors errors;
  const StepFunctor functor{ps, &wind, &domain, &errors, params, step_index};
  Tally tally;
  Kokkos::parallel_reduce("lps::step", Policy(0, static_cast<Index>(ps.size)), functor, tally);
  errors.raise_if_any(tally.errors, ("step " + std::to_string(step_index)).c_str());
  return tally;
}

MomentSums first_moments(const ParticleArrays& ps, const Grid& grid) {
  const FirstMomentFunctor functor{ps, {grid.length_x(), grid.length_y(), grid.length_z()}};
  MomentSums sums;
  Kokkos::parallel_reduce("lps::first_moments", Policy(0, static_cast<Index>(ps.size)), functor, sums);
  return sums;
}

Vec3 second_moments(const ParticleArrays& ps, const Grid& grid, const Vec3& mean) {
  const SecondMomentFunctor functor{ps, {grid.length_x(), grid.length_y(), grid.length_z()}, mean};
  Vec3 sums;
  Kokkos::parallel_reduce("lps::second_moments", Policy(0, static_cast<Index>(ps.size)), functor, sums);
  return sums;
}

Inventory take_inventory(const ParticleArrays& ps) {
  Inventory inventory;
  Kokkos::parallel_reduce("lps::inventory", Policy(0, static_cast<Index>(ps.size)), InventoryFunctor{ps},
                          inventory);
  return inventory;
}

std::vector<double> deposit_box(const ParticleArrays& ps, const Grid& grid, const Decomposition::Box& box) {
  const Index n = static_cast<Index>(ps.size);
  const Array<Index> cell(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lps::deposit.cell"), n);
  const int lo0 = box.lo[0], lo1 = box.lo[1], lo2 = box.lo[2];
  const int hi0 = box.hi[0], hi1 = box.hi[1], hi2 = box.hi[2];
  const int bx = box.extent(0), by = box.extent(1);
  Index misplaced = 0;
  Kokkos::parallel_reduce(
      "lps::deposit.locate", Policy(0, n),
      [=](const Index p, Index& bad) {
        cell(p) = -1;
        if (!ps.active(p)) return;
        int i, j, k;
        if (!grid.locate(Vec3{ps.x(p), ps.y(p), ps.z(p)}, i, j, k) || i < lo0 || i >= hi0 || j < lo1 || j >= hi1 ||
            k < lo2 || k >= hi2) {
          ++bad;
          return;
        }
        cell(p) = (i - lo0) + static_cast<Index>(bx) * ((j - lo1) + static_cast<Index>(by) * (k - lo2));
      },
      misplaced);
  if (misplaced != 0)
    throw std::logic_error("deposit: " + std::to_string(misplaced) + " active particle(s) outside this rank's cells");

  // Per-cell compensated sums in storage order (output time only). Kept
  // serial per rank so cell masses match the serial deposit to round-off.
  std::vector<CompensatedSum> sums(static_cast<std::size_t>(box.num_cells()));
  for (Index p = 0; p < n; ++p)
    if (cell(p) >= 0) sums[static_cast<std::size_t>(cell(p))].add(ps.mass(p));
  std::vector<double> cell_mass(sums.size());
  for (std::size_t c = 0; c < sums.size(); ++c) cell_mass[c] = sums[c].result();
  return cell_mass;
}

}  // namespace lps::par
