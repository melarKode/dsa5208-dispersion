#pragma once

// MPI part of the MPI+Kokkos solver: particle ownership and migration, global
// reductions and gathers to rank 0. Knows nothing about wind or boundary
// physics: it only uses the cell that contains a particle.

#include <mpi.h>

#include <array>
#include <cstdint>
#include <vector>

#include "kokkos_particles.hpp"
#include "lps/decomposition.hpp"
#include "lps/diagnostics.hpp"
#include "lps/grid.hpp"
#include "lps/particles.hpp"
#include "lps/types.hpp"

namespace lps::par {

// MPI datatype for PackedParticle (one element per particle, so counts stay
// in particles rather than bytes). Free it before MPI_Finalize.
class PackedParticleType {
 public:
  PackedParticleType();
  ~PackedParticleType();
  PackedParticleType(const PackedParticleType&) = delete;
  PackedParticleType& operator=(const PackedParticleType&) = delete;
  MPI_Datatype get() const { return type_; }

 private:
  MPI_Datatype type_ = MPI_DATATYPE_NULL;
};

struct MigrationStats {
  std::int64_t sent = 0;
  std::int64_t received = 0;
  std::int64_t non_neighbour = 0;  // sent to a rank that is not adjacent: crossed >= 1 whole subdomain
};

// Routes every stored active particle to the rank that owns its cell, in a
// single all-to-all exchange. A particle that crossed several subdomains in
// one step therefore goes straight to its final owner (no repeated hops).
// All fields travel together: global id, position, periodic image counters,
// mass and active flag. Escaped (inactive) particles stay where they are.
class Migrator {
 public:
  Migrator(const Decomposition& decomposition, const Grid& grid, MPI_Comm comm);
  MigrationStats migrate(ParticleArrays& particles);

 private:
  Decomposition decomposition_;
  Grid grid_;
  MPI_Comm comm_;
  int rank_ = 0;
  int size_ = 1;
  PackedParticleType type_;
  Array<int> destination_{"lps::migration.destination", 0};
  Array<Index> leaving_{"lps::migration.leaving", 0};
  Array<Index> filler_{"lps::migration.filler", 0};
  std::vector<PackedParticle> send_, receive_;
};

// Combines one compensated sum per rank in rank order on every rank, so the
// result does not depend on the MPI implementation's reduction tree.
CompensatedSum allreduce_compensated(const CompensatedSum& local, MPI_Comm comm);

// Global diagnostics row, identical on every rank (two-pass moments over
// unwrapped positions, as lps::compute_diagnostics).
DiagnosticsRow global_diagnostics(const ParticleArrays& particles, const Grid& grid, double time,
                                  double initial_mass, double mass_escaped, MPI_Comm comm);

struct GlobalInventory {
  std::int64_t stored = 0;
  std::int64_t active = 0;
  std::int64_t off_home_image = 0;
  std::uint64_t fingerprint = 0;
  std::vector<std::int64_t> active_per_rank;  // rank 0 only
  std::vector<std::int64_t> stored_per_rank;  // rank 0 only
};
GlobalInventory global_inventory(const ParticleArrays& particles, MPI_Comm comm);

// Full-grid cell masses (cell_id order) on rank 0; empty on other ranks.
std::vector<double> gather_cell_mass(const ParticleArrays& particles, const Grid& grid,
                                     const Decomposition& decomposition, MPI_Comm comm);

// Every stored particle on rank 0 (empty set on other ranks). Rank 0 checks
// that each id 0..expected_count-1 occurs exactly once and throws otherwise.
ParticleSet gather_particles(const ParticleArrays& particles, std::int64_t expected_count, MPI_Comm comm);

}  // namespace lps::par
