#include "lps/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "lps/config.hpp"

namespace lps {

namespace {

std::ofstream open_for_writing(const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  return out;
}

}  // namespace

DiagnosticsRow compute_diagnostics(const ParticleSet& ps, const Grid& grid, double time, double initial_mass,
                                   double mass_escaped) {
  DiagnosticsRow row;
  row.time = time;
  row.mass_escaped = mass_escaped;

  // Pass 1: mass and mass-weighted mean.
  CompensatedSum mass;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (std::size_t p = 0; p < ps.size(); ++p) {
    if (!ps.active[p]) continue;
    const Vec3 r = ps.unwrapped_position(p, grid);
    const double m = ps.mass[p];
    mass.add(m);
    sx += m * r.x;
    sy += m * r.y;
    sz += m * r.z;
    ++row.active_particles;
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

  // Pass 2: central second moments (two-pass avoids cancellation).
  double vx = 0.0, vy = 0.0, vz = 0.0;
  for (std::size_t p = 0; p < ps.size(); ++p) {
    if (!ps.active[p]) continue;
    const Vec3 r = ps.unwrapped_position(p, grid);
    const double m = ps.mass[p];
    vx += m * (r.x - row.mean.x) * (r.x - row.mean.x);
    vy += m * (r.y - row.mean.y) * (r.y - row.mean.y);
    vz += m * (r.z - row.mean.z) * (r.z - row.mean.z);
  }
  row.variance = {vx / row.mass_inside, vy / row.mass_inside, vz / row.mass_inside};
  return row;
}

DiagnosticsWriter::DiagnosticsWriter(const std::string& path) : out_(open_for_writing(path)) {
  out_ << "time,active_particles,mass_inside,mass_escaped,mass_residual,mean_x,mean_y,mean_z,var_x,var_y,var_z\n";
}

void DiagnosticsWriter::write(const DiagnosticsRow& r) {
  out_ << format_double(r.time) << ',' << r.active_particles << ',' << format_double(r.mass_inside) << ','
       << format_double(r.mass_escaped) << ',' << format_double(r.mass_residual) << ',' << format_double(r.mean.x)
       << ',' << format_double(r.mean.y) << ',' << format_double(r.mean.z) << ',' << format_double(r.variance.x)
       << ',' << format_double(r.variance.y) << ',' << format_double(r.variance.z) << '\n';
  out_.flush();
}

std::vector<double> deposit_mass_ngp(const ParticleSet& ps, const Grid& grid) {
  // Compensated per-cell sums: a cell holding 3056 particles of mass 1e-5
  // reports 0.030560000000000004 rather than 0.030559999999998925.
  std::vector<CompensatedSum> sums(static_cast<std::size_t>(grid.num_cells()));
  for (std::size_t p = 0; p < ps.size(); ++p) {
    if (!ps.active[p]) continue;
    int i, j, k;
    if (!grid.locate(ps.position(p), i, j, k))
      throw std::logic_error("active particle " + std::to_string(ps.id[p]) + " is outside the grid");
    sums[static_cast<std::size_t>(grid.cell_id(i, j, k))].add(ps.mass[p]);
  }
  std::vector<double> cell_mass(sums.size());
  for (std::size_t c = 0; c < sums.size(); ++c) cell_mass[c] = sums[c].result();
  return cell_mass;
}

void write_concentration_csv(const std::string& path, const Grid& grid, const std::vector<double>& cell_mass) {
  std::ofstream out = open_for_writing(path);
  out << "cell_id,i,j,k,mass,concentration\n";
  const double volume = grid.cell_volume();
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::int64_t id = grid.cell_id(i, j, k);
        const double m = cell_mass[static_cast<std::size_t>(id)];
        out << id << ',' << i << ',' << j << ',' << k << ',' << format_double(m) << ',' << format_double(m / volume)
            << '\n';
      }
    }
  }
}

void write_concentration_binary(const std::string& path, const Grid& grid, const std::vector<double>& cell_mass) {
  std::vector<double> concentration(cell_mass);
  for (auto& m : concentration) m /= grid.cell_volume();
  std::ofstream binary(path, std::ios::binary);
  if (!binary.write(reinterpret_cast<const char*>(concentration.data()),
                    static_cast<std::streamsize>(concentration.size() * sizeof(double))))
    throw std::runtime_error("Snapshot write failed");
}

std::string snapshot_filename(const std::string& stem, long long index, const std::string& extension) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "_%06lld", index);
  return stem + buffer + extension;
}

void write_particles_csv(const std::string& path, const ParticleSet& ps, const Grid& grid) {
  std::vector<std::size_t> order(ps.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ps.id[a] < ps.id[b]; });

  std::ofstream out = open_for_writing(path);
  out << "id,active,x,y,z,x_unwrapped,y_unwrapped,z_unwrapped,mass\n";
  for (const std::size_t p : order) {
    const Vec3 u = ps.unwrapped_position(p, grid);
    out << ps.id[p] << ',' << int(ps.active[p]) << ',' << format_double(ps.x[p]) << ',' << format_double(ps.y[p])
        << ',' << format_double(ps.z[p]) << ',' << format_double(u.x) << ',' << format_double(u.y) << ','
        << format_double(u.z) << ',' << format_double(ps.mass[p]) << '\n';
  }
}

}  // namespace lps
