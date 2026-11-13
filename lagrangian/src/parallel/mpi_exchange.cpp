#include "mpi_exchange.hpp"

#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace lps::par {

namespace {

int comm_rank(MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int comm_size(MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  return size;
}

int checked_int(std::int64_t value, const char* what) {
  if (value > INT_MAX) throw std::runtime_error(std::string(what) + " exceeds the MPI int count limit");
  return static_cast<int>(value);
}

// Exclusive prefix sums of per-rank counts, as MPI displacements.
std::vector<int> displacements(const std::vector<int>& counts) {
  std::vector<int> offsets(counts.size(), 0);
  std::int64_t running = 0;
  for (std::size_t r = 0; r < counts.size(); ++r) {
    offsets[r] = checked_int(running, "message size");
    running += counts[r];
  }
  checked_int(running, "message size");
  return offsets;
}

}  // namespace

PackedParticleType::PackedParticleType() {
  MPI_Type_contiguous(static_cast<int>(sizeof(PackedParticle)), MPI_BYTE, &type_);
  MPI_Type_commit(&type_);
}

PackedParticleType::~PackedParticleType() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && type_ != MPI_DATATYPE_NULL) MPI_Type_free(&type_);
}

Migrator::Migrator(const Decomposition& decomposition, const Grid& grid, MPI_Comm comm)
    : decomposition_(decomposition), grid_(grid), comm_(comm), rank_(comm_rank(comm)), size_(comm_size(comm)) {
  if (decomposition_.num_ranks() != size_)
    throw std::runtime_error("rank grid has " + std::to_string(decomposition_.num_ranks()) + " ranks but MPI has " +
                             std::to_string(size_));
}

MigrationStats Migrator::migrate(ParticleArrays& ps) {
  MigrationStats stats;
  const Index n = static_cast<Index>(ps.size);
  if (static_cast<Index>(destination_.extent(0)) < n)
    Kokkos::realloc(Kokkos::WithoutInitializing, destination_, static_cast<std::size_t>(n + n / 2 + 64));

  // 1. Owner of every stored particle (inactive ones stay here).
  const Array<int> destination = destination_;
  const ParticleArrays current = ps;
  const Grid grid = grid_;
  const Decomposition decomposition = decomposition_;
  const int me = rank_;
  Index leaving = 0;
  Kokkos::parallel_reduce(
      "lps::migrate.owner", Policy(0, n),
      [=](const Index p, Index& count) {
        int owner = me;
        if (current.active(p)) {
          int i, j, k;
          owner = grid.locate(Vec3{current.x(p), current.y(p), current.z(p)}, i, j, k)
                      ? decomposition.owner_of_cell(i, j, k)
                      : -1;
        }
        destination(p) = owner;
        if (owner != me) ++count;
      },
      leaving);

  // 2. Pack leavers by destination rank (index order inside each bucket).
  std::vector<int> send_counts(static_cast<std::size_t>(size_), 0);
  send_.clear();
  if (leaving > 0) {
    if (static_cast<Index>(leaving_.extent(0)) < leaving)
      Kokkos::realloc(Kokkos::WithoutInitializing, leaving_, static_cast<std::size_t>(leaving + leaving / 2 + 64));
    const Array<Index> leaver = leaving_;
    Kokkos::parallel_scan("lps::migrate.select", Policy(0, n), [=](const Index p, Index& offset, const bool final) {
      if (destination(p) != me) {
        if (final) leaver(offset) = p;
        ++offset;
      }
    });
    for (Index q = 0; q < leaving; ++q) {
      const Index p = leaver(q);
      const int d = destination(p);
      if (d < 0)
        throw std::logic_error("migration: active particle " + std::to_string(current.id(p)) +
                               " is outside the grid");
      ++send_counts[static_cast<std::size_t>(d)];
    }
    std::vector<int> cursor = displacements(send_counts);
    send_.resize(static_cast<std::size_t>(leaving));
    for (Index q = 0; q < leaving; ++q) {
      const Index p = leaver(q);
      const int d = destination(p);
      send_[static_cast<std::size_t>(cursor[static_cast<std::size_t>(d)]++)] = pack_particle(current, p);
      if (!decomposition_.adjacent(me, d)) ++stats.non_neighbour;
    }
    stats.sent = leaving;
  }

  // 3. One all-to-all: every particle reaches its final owner directly.
  std::vector<int> receive_counts(static_cast<std::size_t>(size_), 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, comm_);
  const std::vector<int> send_offsets = displacements(send_counts);
  const std::vector<int> receive_offsets = displacements(receive_counts);
  const Index received =
      static_cast<Index>(receive_offsets.back()) + static_cast<Index>(receive_counts.back());
  receive_.resize(static_cast<std::size_t>(received));
  MPI_Alltoallv(send_.data(), send_counts.data(), send_offsets.data(), type_.get(), receive_.data(),
                receive_counts.data(), receive_offsets.data(), type_.get(), comm_);
  stats.received = received;

  // 4. Close the gaps in place: the leavers' slots below `keep` (the holes) are
  // refilled by the stayers found in the tail [keep, n). Holes and tail stayers
  // are equally many, so the work is O(leaving), not O(n). Storage order
  // changes, which the physics never depends on (updates are keyed by id).
  if (leaving > 0) {
    const Index keep = n - leaving;
    const Array<Index> leaver = leaving_;
    // Leavers are sorted ascending; holes are those below `keep`.
    Index holes = 0;
    while (holes < leaving && leaver(holes) < keep) ++holes;
    if (holes > 0) {
      if (static_cast<Index>(filler_.extent(0)) < holes)
        Kokkos::realloc(Kokkos::WithoutInitializing, filler_, static_cast<std::size_t>(holes + holes / 2 + 64));
      const Array<Index> filler = filler_;
      Kokkos::parallel_scan("lps::migrate.tail", Policy(keep, n), [=](const Index p, Index& offset, const bool final) {
        if (destination(p) == me) {
          if (final) filler(offset) = p;
          ++offset;
        }
      });
      Kokkos::parallel_for("lps::migrate.fill", Policy(0, holes),
                           [=](const Index q) { copy_particle(current, filler(q), current, leaver(q)); });
    }
    ps.size = static_cast<std::size_t>(keep);
  }
  if (received > 0) {
    ps.reserve(ps.size + static_cast<std::size_t>(received));
    const ParticleArrays target = ps;
    const PackedParticle* arrivals = receive_.data();
    const Index base = static_cast<Index>(ps.size);
    Kokkos::parallel_for("lps::migrate.unpack", Policy(0, received),
                         [=](const Index q) { unpack_particle(arrivals[q], target, base + q); });
    ps.size += static_cast<std::size_t>(received);
  }
  return stats;
}

CompensatedSum allreduce_compensated(const CompensatedSum& local, MPI_Comm comm) {
  const int size = comm_size(comm);
  const double mine[2] = {local.sum, local.correction};
  std::vector<double> all(2 * static_cast<std::size_t>(size));
  MPI_Allgather(mine, 2, MPI_DOUBLE, all.data(), 2, MPI_DOUBLE, comm);
  CompensatedSum total;
  for (int r = 0; r < size; ++r) {
    total.add(all[2 * static_cast<std::size_t>(r)]);
    total.add(all[2 * static_cast<std::size_t>(r) + 1]);
  }
  return total;
}

DiagnosticsRow global_diagnostics(const ParticleArrays& ps, const Grid& grid, double time, double initial_mass,
                                  double mass_escaped, MPI_Comm comm) {
  const int size = comm_size(comm);
  DiagnosticsRow row;
  row.time = time;
  row.mass_escaped = mass_escaped;

  // Pass 1: mass (compensated) and first moments, combined in rank order.
  const MomentSums local = first_moments(ps, grid);
  const double mine[6] = {local.mass, local.mass_correction, local.mx, local.my, local.mz,
                          static_cast<double>(local.active)};
  std::vector<double> all(6 * static_cast<std::size_t>(size));
  MPI_Allgather(mine, 6, MPI_DOUBLE, all.data(), 6, MPI_DOUBLE, comm);
  CompensatedSum mass;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (int r = 0; r < size; ++r) {
    const double* v = &all[6 * static_cast<std::size_t>(r)];
    mass.add(v[0]);
    mass.add(v[1]);
    sx += v[2];
    sy += v[3];
    sz += v[4];
    row.active_particles += static_cast<std::int64_t>(std::llround(v[5]));
  }
  row.mass_inside = mass.result();
  row.mass_residual = initial_mass - row.mass_inside - mass_escaped;
  if (row.active_particles == 0 || row.mass_inside <= 0.0) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    row.mean = {nan, nan, nan};
    row.variance = {nan, nan, nan};
    return row;
  }
  row.mean = {sx / row.mass_inside, sy / row.mass_inside, sz / row.mass_inside};

  // Pass 2: central second moments about the global mean.
  const Vec3 second = second_moments(ps, grid, row.mean);
  const double mine2[3] = {second.x, second.y, second.z};
  std::vector<double> all2(3 * static_cast<std::size_t>(size));
  MPI_Allgather(mine2, 3, MPI_DOUBLE, all2.data(), 3, MPI_DOUBLE, comm);
  double vx = 0.0, vy = 0.0, vz = 0.0;
  for (int r = 0; r < size; ++r) {
    vx += all2[3 * static_cast<std::size_t>(r)];
    vy += all2[3 * static_cast<std::size_t>(r) + 1];
    vz += all2[3 * static_cast<std::size_t>(r) + 2];
  }
  row.variance = {vx / row.mass_inside, vy / row.mass_inside, vz / row.mass_inside};
  return row;
}

GlobalInventory global_inventory(const ParticleArrays& ps, MPI_Comm comm) {
  const int rank = comm_rank(comm), size = comm_size(comm);
  const Inventory local = take_inventory(ps);
  GlobalInventory global;
  const std::int64_t counts[3] = {local.stored, local.active, local.off_home_image};
  std::int64_t totals[3] = {0, 0, 0};
  MPI_Allreduce(counts, totals, 3, MPI_INT64_T, MPI_SUM, comm);
  global.stored = totals[0];
  global.active = totals[1];
  global.off_home_image = totals[2];
  // Unsigned sum wraps mod 2^64 on every rank identically.
  std::vector<std::uint64_t> prints(static_cast<std::size_t>(size));
  MPI_Allgather(&local.fingerprint, 1, MPI_UINT64_T, prints.data(), 1, MPI_UINT64_T, comm);
  for (const std::uint64_t f : prints) global.fingerprint += f;

  const std::int64_t mine[2] = {local.active, local.stored};
  std::vector<std::int64_t> per_rank(rank == 0 ? 2 * static_cast<std::size_t>(size) : 0);
  MPI_Gather(mine, 2, MPI_INT64_T, per_rank.data(), 2, MPI_INT64_T, 0, comm);
  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      global.active_per_rank.push_back(per_rank[2 * static_cast<std::size_t>(r)]);
      global.stored_per_rank.push_back(per_rank[2 * static_cast<std::size_t>(r) + 1]);
    }
  }
  return global;
}

std::vector<double> gather_cell_mass(const ParticleArrays& ps, const Grid& grid,
                                     const Decomposition& decomposition, MPI_Comm comm) {
  const int rank = comm_rank(comm), size = comm_size(comm);
  const std::vector<double> local = deposit_box(ps, grid, decomposition.box(rank));

  std::vector<int> counts, offsets;
  std::vector<double> buffer;
  if (rank == 0) {
    for (int r = 0; r < size; ++r) counts.push_back(checked_int(decomposition.box(r).num_cells(), "subdomain size"));
    offsets = displacements(counts);
    buffer.resize(static_cast<std::size_t>(grid.num_cells()));
  }
  MPI_Gatherv(local.data(), checked_int(static_cast<std::int64_t>(local.size()), "subdomain size"), MPI_DOUBLE,
              buffer.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
  if (rank != 0) return {};

  // Scatter each rank's x-fastest box into the global cell_id order.
  std::vector<double> cell_mass(static_cast<std::size_t>(grid.num_cells()), 0.0);
  for (int r = 0; r < size; ++r) {
    const Decomposition::Box b = decomposition.box(r);
    std::size_t q = static_cast<std::size_t>(offsets[static_cast<std::size_t>(r)]);
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
      for (int j = b.lo[1]; j < b.hi[1]; ++j)
        for (int i = b.lo[0]; i < b.hi[0]; ++i) cell_mass[static_cast<std::size_t>(grid.cell_id(i, j, k))] = buffer[q++];
  }
  return cell_mass;
}

ParticleSet gather_particles(const ParticleArrays& ps, std::int64_t expected_count, MPI_Comm comm) {
  const int rank = comm_rank(comm), size = comm_size(comm);
  const PackedParticleType type;
  std::vector<PackedParticle> local(ps.size);
  for (std::size_t p = 0; p < ps.size; ++p) local[p] = pack_particle(ps, static_cast<Index>(p));

  const int mine = checked_int(static_cast<std::int64_t>(local.size()), "particle count");
  std::vector<int> counts(rank == 0 ? static_cast<std::size_t>(size) : 0);
  MPI_Gather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
  std::vector<int> offsets;
  std::vector<PackedParticle> all;
  if (rank == 0) {
    offsets = displacements(counts);
    all.resize(static_cast<std::size_t>(offsets.back()) + static_cast<std::size_t>(counts.back()));
  }
  MPI_Gatherv(local.data(), mine, type.get(), all.data(), counts.data(), offsets.data(), type.get(), 0, comm);
  if (rank != 0) return {};

  // Every id exactly once: detects lost and duplicated particles directly.
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(expected_count), 0);
  for (const PackedParticle& q : all) {
    if (q.id >= static_cast<std::uint64_t>(expected_count))
      throw std::runtime_error("particle id " + std::to_string(q.id) + " out of range");
    if (seen[q.id]++) throw std::runtime_error("particle id " + std::to_string(q.id) + " is duplicated");
  }
  for (std::size_t id = 0; id < seen.size(); ++id)
    if (!seen[id]) throw std::runtime_error("particle id " + std::to_string(id) + " is missing");

  ParticleSet set;
  set.resize(all.size());
  for (std::size_t p = 0; p < all.size(); ++p) {
    const PackedParticle& q = all[p];
    set.id[p] = q.id;
    set.x[p] = q.x;
    set.y[p] = q.y;
    set.z[p] = q.z;
    set.image_x[p] = q.image[0];
    set.image_y[p] = q.image[1];
    set.image_z[p] = q.image[2];
    set.mass[p] = q.mass;
    set.active[p] = q.active;
  }
  return set;
}

}  // namespace lps::par
