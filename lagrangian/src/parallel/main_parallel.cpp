// MPI + Kokkos Lagrangian particle solver. Same physics, configuration and
// output contract as the serial lagrangian_solver (the reference); see
// docs/mpi-kokkos.md for the design, timers and validation.
//
// Per step:  advance (Kokkos kernel, physical step incl. walls/escape/wrap)
//            -> migrate (route each particle to the rank owning its cell).
// Rank boundaries only trigger migration; the physics never sees them.

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kokkos_particles.hpp"
#include "lps/config.hpp"
#include "lps/decomposition.hpp"
#include "lps/diagnostics.hpp"
#include "lps/domain.hpp"
#include "lps/stepper.hpp"
#include "lps/urban.hpp"
#include "lps/wind.hpp"
#include "mpi_exchange.hpp"

namespace {

using namespace lps;
using namespace lps::par;
namespace fs = std::filesystem;

void print_usage(const char* program) {
  std::cerr << "usage: mpirun -n P " << program
            << " CONFIG [--output DIR] [--set key=value]... [--mpi-dims PX,PY,PZ] [--kokkos-...]\n"
            << "  CONFIG              key=value configuration file (same as lagrangian_solver)\n"
            << "  --output DIR        output directory (overrides output_dir)\n"
            << "  --set key=value     override or add one config key (repeatable)\n"
            << "  --mpi-dims PX,PY,PZ rank grid (default: x-y split chosen from the grid)\n"
            << "  threads per rank:   OMP_NUM_THREADS (or --kokkos-num-threads=N)\n";
}

struct Arguments {
  std::string config_path;
  std::string output_override;
  std::vector<std::string> overrides;
  std::string mpi_dims;
  bool help = false;
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "-h" || arg == "--help") {
      args.help = true;
    } else if (arg == "--output" && a + 1 < argc) {
      args.output_override = argv[++a];
    } else if (arg == "--set" && a + 1 < argc) {
      args.overrides.push_back(argv[++a]);
    } else if (arg == "--mpi-dims" && a + 1 < argc) {
      args.mpi_dims = argv[++a];
    } else if (args.config_path.empty() && arg.rfind("--", 0) != 0) {
      args.config_path = arg;
    } else {
      throw std::runtime_error("unexpected argument '" + arg + "' (try --help)");
    }
  }
  if (!args.help && args.config_path.empty()) throw std::runtime_error("missing CONFIG (try --help)");
  return args;
}

// Silences std::cerr on non-root ranks while parsing, so config warnings appear once.
class QuietCerr {
 public:
  explicit QuietCerr(bool quiet) : saved_(quiet ? std::cerr.rdbuf(sink_.rdbuf()) : nullptr) {}
  ~QuietCerr() {
    if (saved_) std::cerr.rdbuf(saved_);
  }

 private:
  std::ostringstream sink_;
  std::streambuf* saved_;
};

// Wall-clock phase timers (MPI_Wtime, per rank). The phases are disjoint and
// sequential: total = setup + stepping + wait + migration + output + other.
// A barrier after each step kernel isolates load imbalance in `wait`, and a
// barrier at the end of each output keeps rank 0's file writing in `output`,
// so `migration` is the particle exchange itself.
struct PhaseTimes {
  double setup = 0.0;      // config, wind/mask loading, release, initial migration
  double stepping = 0.0;   // advance kernel (physics) only
  double wait = 0.0;       // barrier after the step kernel: time spent waiting for slower ranks
  double migration = 0.0;  // owner search, packing, all-to-all, compaction, unpacking
  double output = 0.0;     // global diagnostics, invariant checks, gathers, file writing
  double total = 0.0;      // from after MPI/Kokkos start-up to the end of the run
};

std::string dims_text(const Decomposition& d) {
  return std::to_string(d.dims[0]) + "x" + std::to_string(d.dims[1]) + "x" + std::to_string(d.dims[2]);
}

int run(int argc, char** argv) {
  const double run_start = MPI_Wtime();
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const bool root = rank == 0;

  const Arguments args = parse_arguments(argc, argv);
  if (args.help) {
    if (root) print_usage(argv[0]);
    return 0;
  }

  // ---- configuration: identical on every rank (same code path as the serial driver)
  Config config;
  {
    QuietCerr quiet(!root);
    KeyValueList entries = read_key_value_file(args.config_path);
    for (const auto& assignment : args.overrides) set_key_value(entries, assignment);
    merge_case_directory(entries, args.config_path);
    config = parse_config(entries);
  }
  if (!args.output_override.empty()) config.output_dir = args.output_override;
  const TimeControl time_control = make_time_control(config);
  const Grid grid = config.grid();
  const std::array<int, 3> dims =
      args.mpi_dims.empty() ? default_rank_dims(size, grid) : parse_rank_dims(args.mpi_dims);
  const Decomposition decomposition(grid, config.boundary, dims);
  if (decomposition.num_ranks() != size)
    throw std::runtime_error("--mpi-dims " + dims_text(decomposition) + " needs " +
                             std::to_string(decomposition.num_ranks()) + " ranks, started with " +
                             std::to_string(size));

  // ---- output directory: rank 0 only writes files
  const fs::path out_dir(config.output_dir);
  if (root) {
    if (fs::exists(out_dir) && !fs::is_empty(out_dir)) throw std::runtime_error("Output directory must be empty");
    fs::create_directories(out_dir);
    if (config.velocity_mode == VelocityMode::File)
      fs::copy_file(fs::path(config.case_dir) / "input.sha256", out_dir / "input.sha256");
    write_effective_config(config, (out_dir / "effective_config.txt").string());
    std::ofstream kokkos_config((out_dir / "kokkos_configuration.txt").string());
    Kokkos::print_configuration(kokkos_config, false);
    std::ofstream layout((out_dir / "decomposition.csv").string());
    layout << "rank,i0,i1,j0,j1,k0,k1,cells\n";
    for (int r = 0; r < size; ++r) {
      const Decomposition::Box b = decomposition.box(r);
      layout << r << ',' << b.lo[0] << ',' << b.hi[0] << ',' << b.lo[1] << ',' << b.hi[1] << ',' << b.lo[2] << ','
             << b.hi[2] << ',' << b.num_cells() << '\n';
    }
  }
  MPI_Barrier(comm);

  // ---- physics objects: the serial reference implementations, unchanged
  const Domain domain(grid, config.boundary, load_solid(config));
  const std::unique_ptr<WindField> wind = make_wind_field(config);
  const StepParameters params{config.seed, config.dt, config.diffusivity};
  const std::int64_t n_total = config.particle_count;
  const int threads = ExecSpace().concurrency();

  // ---- release: rank r samples ids [N r / P, N (r+1) / P), then particles move to their owners
  PhaseTimes times;
  ParticleArrays particles("lps::particles");
  const std::uint64_t first = static_cast<std::uint64_t>(n_total) * static_cast<std::uint64_t>(rank) / size;
  const std::uint64_t last = static_cast<std::uint64_t>(n_total) * static_cast<std::uint64_t>(rank + 1) / size;
  const Tally released = release_block(particles, config, domain, first, last);
  CompensatedSum escaped_mass_local;
  escaped_mass_local.add(released.mass());
  std::int64_t escaped_local = released.escaped_particles;
  std::int64_t wraps_local = 0;

  // Expected id fingerprint: every id 0..N-1 exactly once.
  const std::uint64_t expected_fingerprint = global_inventory(particles, comm).fingerprint;

  Migrator migrator(decomposition, grid, comm);
  MigrationStats migration_totals = migrator.migrate(particles);
  std::int64_t max_sent_in_one_step = 0;
  times.setup = MPI_Wtime() - run_start;

  // ---- output files (rank 0)
  std::unique_ptr<DiagnosticsWriter> diagnostics;
  std::ofstream snapshots, rank_counts;
  if (root) {
    diagnostics = std::make_unique<DiagnosticsWriter>((out_dir / "diagnostics.csv").string());
    snapshots.open((out_dir / "snapshots.csv").string());
    snapshots << (config.write_binary ? "snapshot_index,step,time,format,ranks\n" : "snapshot_index,step,time\n");
    rank_counts.open((out_dir / "rank_counts.csv").string());
    rank_counts << "snapshot_index,step,time,rank,active_particles,stored_particles\n";
    std::cout << "case " << config.case_name << ": " << n_total << " particles, " << time_control.num_steps
              << " steps of dt=" << format_double(config.dt) << " s, wind " << wind->description()
              << ", K=" << format_double(config.diffusivity) << " m^2/s; " << size << " MPI ranks ("
              << dims_text(decomposition) << ") x " << threads << " " << ExecSpace::name() << " threads\n";
  }

  long long snapshot_index = 0;
  const auto write_output = [&](std::int64_t step) {
    const double start = MPI_Wtime();
    const double time = static_cast<double>(step) * config.dt;
    const double escaped_mass = allreduce_compensated(escaped_mass_local, comm).result();
    std::int64_t escaped_count = 0;
    MPI_Allreduce(&escaped_local, &escaped_count, 1, MPI_INT64_T, MPI_SUM, comm);
    const DiagnosticsRow row = global_diagnostics(particles, grid, time, config.initial_mass, escaped_mass, comm);

    // Conservation invariants, checked on every rank at every output.
    const GlobalInventory inventory = global_inventory(particles, comm);
    if (inventory.stored != n_total || inventory.fingerprint != expected_fingerprint)
      throw std::runtime_error("particle inventory broken at t=" + format_double(time) + ": " +
                               std::to_string(inventory.stored) + " stored of " + std::to_string(n_total) +
                               (inventory.fingerprint != expected_fingerprint ? ", id fingerprint mismatch" : ""));
    if (inventory.active + escaped_count != n_total || inventory.active != row.active_particles)
      throw std::runtime_error("active + escaped != initial count at t=" + format_double(time));

    std::vector<double> cell_mass;
    if (config.write_concentration) cell_mass = gather_cell_mass(particles, grid, decomposition, comm);
    ParticleSet everyone;
    if (config.write_particles) everyone = gather_particles(particles, n_total, comm);

    if (root) {
      diagnostics->write(row);
      if (config.velocity_mode == VelocityMode::File)
        std::cout << "t=" << time << " mass=" << row.mass_inside << " escaped=" << row.mass_escaped
                  << " residual=" << row.mass_residual << std::endl;
      snapshots << snapshot_index << ',' << step << ',' << format_double(time);
      if (config.write_binary) snapshots << (config.write_concentration ? ",binary,1" : ",none,1");
      snapshots << '\n';
      snapshots.flush();
      for (int r = 0; r < size; ++r)
        rank_counts << snapshot_index << ',' << step << ',' << format_double(time) << ',' << r << ','
                    << inventory.active_per_rank[static_cast<std::size_t>(r)] << ','
                    << inventory.stored_per_rank[static_cast<std::size_t>(r)] << '\n';
      rank_counts.flush();
      if (config.write_concentration && config.write_binary)
        write_concentration_binary((out_dir / snapshot_filename("concentration", snapshot_index, ".bin")).string(),
                                   grid, cell_mass);
      else if (config.write_concentration)
        write_concentration_csv((out_dir / snapshot_filename("concentration", snapshot_index, ".csv")).string(), grid,
                                cell_mass);
      if (config.write_particles)
        write_particles_csv((out_dir / snapshot_filename("particles", snapshot_index, ".csv")).string(), everyone,
                            grid);
    }
    ++snapshot_index;
    // Ranks wait here for rank 0's file writes, so that wait is charged to
    // output instead of leaking into the next step's migration all-to-all.
    MPI_Barrier(comm);
    times.output += MPI_Wtime() - start;
  };

  // ---- time loop
  write_output(0);
  for (std::int64_t n = 0; n < time_control.num_steps; ++n) {
    const double t0 = MPI_Wtime();
    const Tally step = advance(particles, *wind, domain, params, static_cast<std::uint32_t>(n));
    const double t1 = MPI_Wtime();
    MPI_Barrier(comm);
    const double t2 = MPI_Wtime();
    const MigrationStats moved = migrator.migrate(particles);
    const double t3 = MPI_Wtime();
    times.stepping += t1 - t0;
    times.wait += t2 - t1;
    times.migration += t3 - t2;

    escaped_mass_local.add(step.escaped_mass);
    escaped_mass_local.add(step.escaped_mass_correction);
    escaped_local += step.escaped_particles;
    wraps_local += step.periodic_wraps;
    migration_totals.sent += moved.sent;
    migration_totals.received += moved.received;
    migration_totals.non_neighbour += moved.non_neighbour;
    max_sent_in_one_step = std::max(max_sent_in_one_step, moved.sent);

    const std::int64_t completed = n + 1;
    if (completed % time_control.output_stride == 0 || completed == time_control.num_steps) write_output(completed);
  }
  times.total = MPI_Wtime() - run_start;

  // ---- run summary: global counts and per-rank timers (rank 0 writes)
  const GlobalInventory final_inventory = global_inventory(particles, comm);
  std::int64_t sums_local[4] = {escaped_local, wraps_local, migration_totals.sent, migration_totals.non_neighbour};
  std::int64_t sums[4] = {0, 0, 0, 0};
  MPI_Allreduce(sums_local, sums, 4, MPI_INT64_T, MPI_SUM, comm);
  std::int64_t max_sent_global = 0;
  MPI_Allreduce(&max_sent_in_one_step, &max_sent_global, 1, MPI_INT64_T, MPI_MAX, comm);
  const double escaped_mass_final = allreduce_compensated(escaped_mass_local, comm).result();

  // Per rank: setup, stepping, wait, migration, output, other, total, loop (= stepping + wait + migration).
  constexpr int kTimers = 8;
  const double loop = times.stepping + times.wait + times.migration;
  const double mine[kTimers + 2] = {times.setup,
                                    times.stepping,
                                    times.wait,
                                    times.migration,
                                    times.output,
                                    times.total - times.setup - loop - times.output,
                                    times.total,
                                    loop,
                                    static_cast<double>(migration_totals.sent),
                                    static_cast<double>(migration_totals.received)};
  std::vector<double> all(root ? (kTimers + 2) * static_cast<std::size_t>(size) : 0);
  MPI_Gather(mine, kTimers + 2, MPI_DOUBLE, all.data(), kTimers + 2, MPI_DOUBLE, 0, comm);

  if (root) {
    std::ofstream timings((out_dir / "timings.csv").string());
    timings << "rank,setup_seconds,stepping_seconds,wait_seconds,migration_seconds,output_seconds,other_seconds,"
               "total_seconds,loop_seconds,particles_sent,particles_received\n";
    double maxima[kTimers] = {0, 0, 0, 0, 0, 0, 0, 0};
    double stepping_min = std::numeric_limits<double>::infinity();
    for (int r = 0; r < size; ++r) {
      const double* v = &all[(kTimers + 2) * static_cast<std::size_t>(r)];
      timings << r;
      for (int c = 0; c < kTimers; ++c) {
        timings << ',' << format_double(v[c]);
        maxima[c] = std::max(maxima[c], v[c]);
      }
      stepping_min = std::min(stepping_min, v[1]);
      timings << ',' << static_cast<std::int64_t>(v[kTimers]) << ',' << static_cast<std::int64_t>(v[kTimers + 1])
              << '\n';
    }

    std::ofstream summary((out_dir / "run_summary.txt").string());
    summary << "case_name=" << config.case_name << "\n"
            << "particle_count=" << n_total << "\n"
            << "num_steps=" << time_control.num_steps << "\n"
            << "num_snapshots=" << snapshot_index << "\n"
            << "final_active_particles=" << final_inventory.active << "\n"
            << "escaped_particles=" << sums[0] << "\n"
            << "final_mass_escaped=" << format_double(escaped_mass_final) << "\n"
            << "periodic_wraps=" << sums[1] << "\n"
            << "particles_off_home_image=" << final_inventory.off_home_image << "\n"
            << "rng=philox4x32-10 counter={id_lo,id_hi,step,stream} key={seed_lo,seed_hi}\n"
            << "wall_seconds=" << format_double(maxima[6]) << "\n"
            << "output_seconds=" << format_double(maxima[4]) << "\n"
            << "solver=mpi+kokkos\n"
            << "mpi_ranks=" << size << "\n"
            << "mpi_dims=" << dims_text(decomposition) << "\n"
            << "kokkos_execution_space=" << ExecSpace::name() << "\n"
            << "threads_per_rank=" << threads << "\n"
            << "migrated_particles=" << sums[2] << "\n"
            << "non_neighbour_migrations=" << sums[3] << "\n"
            << "max_particles_sent_by_one_rank_in_one_step=" << max_sent_global << "\n"
            << "setup_seconds_max=" << format_double(maxima[0]) << "\n"
            << "stepping_seconds_max=" << format_double(maxima[1]) << "\n"
            << "stepping_seconds_min=" << format_double(stepping_min) << "\n"
            << "wait_seconds_max=" << format_double(maxima[2]) << "\n"
            << "migration_seconds_max=" << format_double(maxima[3]) << "\n"
            << "output_seconds_max=" << format_double(maxima[4]) << "\n"
            << "loop_seconds_max=" << format_double(maxima[7]) << "\n"
            << "total_seconds_max=" << format_double(maxima[6]) << "\n"
            << "timer_categories=disjoint per rank: total = setup + stepping + wait + migration + output + other; "
               "wait = barrier after each step kernel (load imbalance); output ends with a barrier\n";

    std::cout << "done in " << maxima[6] << " s (step " << maxima[1] << " s, wait " << maxima[2] << " s, migration "
              << maxima[3] << " s, output " << maxima[4] << " s); " << final_inventory.active << " active, " << sums[0]
              << " escaped, " << sums[1] << " periodic wraps, " << sums[2] << " migrations ("
              << sums[3] << " non-neighbour) -> " << out_dir.string() << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      status = run(argc, argv);
    } catch (const std::exception& error) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::cerr << "error (rank " << rank << "): " << error.what() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Finalize();
  return status;
}
