#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "lps/grid.hpp"
#include "lps/particles.hpp"
#include "lps/types.hpp"

namespace lps {

// One row of diagnostics.csv:
//   time,active_particles,mass_inside,mass_escaped,mass_residual,
//   mean_x,mean_y,mean_z,var_x,var_y,var_z
// Means and variances are mass-weighted over active particles and use
// UNWRAPPED positions, so they stay meaningful if the plume crosses a periodic
// boundary. Variance is the weighted population variance (divide by total
// mass). All moments are NaN when no particles remain.
struct DiagnosticsRow {
  double time = 0.0;
  std::int64_t active_particles = 0;
  double mass_inside = 0.0;
  double mass_escaped = 0.0;
  double mass_residual = 0.0;  // initial_mass - mass_inside - mass_escaped
  Vec3 mean;
  Vec3 variance;
};

DiagnosticsRow compute_diagnostics(const ParticleSet& particles, const Grid& grid, double time,
                                   double initial_mass, double mass_escaped);

class DiagnosticsWriter {
 public:
  explicit DiagnosticsWriter(const std::string& path);
  void write(const DiagnosticsRow& row);

 private:
  std::ofstream out_;
};

// Nearest-grid-point deposit: each active particle's mass goes to the cell
// containing it. Indexed by cell_id; length grid.num_cells().
std::vector<double> deposit_mass_ngp(const ParticleSet& particles, const Grid& grid);

// cell_id,i,j,k,mass,concentration for every cell (zero-mass cells included),
// in cell_id order; concentration = mass / (dx dy dz).
void write_concentration_csv(const std::string& path, const Grid& grid, const std::vector<double>& cell_mass);

// Little-endian doubles of mass / (dx dy dz) in cell_id (x-fastest) order:
// the write_binary=true snapshot format.
void write_concentration_binary(const std::string& path, const Grid& grid, const std::vector<double>& cell_mass);

// Snapshot file name, e.g. snapshot_filename("concentration", 3, ".csv") -> "concentration_000003.csv".
std::string snapshot_filename(const std::string& stem, long long index, const std::string& extension);

// id,active,x,y,z,x_unwrapped,y_unwrapped,z_unwrapped,mass, sorted by id.
void write_particles_csv(const std::string& path, const ParticleSet& particles, const Grid& grid);

}  // namespace lps
