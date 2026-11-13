#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace dispersion {
using Options = std::map<std::string, std::string>;
struct Config {
  int nx = 32, ny = 32, nz = 32;
  double dx = 1, dy = 1, dz = 1, ox = 0, oy = 0, oz = 0;
  double u = .2, v = 0, w = 0, diffusion = .1, mass = 1, sigma = 1;
  double sx = 12, sy = 16, sz = 16, dt = .05, end = 2, output_interval = .5, cfl = .9;
  std::array<std::string, 3> boundary{"periodic", "periodic", "periodic"};
  std::string initial = "gaussian", velocity = "uniform", case_dir, output = "results/run";
  std::string decomposition = "block", exchange = "blocking", snapshots = "binary";
  int wall_i = -1, px = 0, py = 0;
  double amplitude = .2;
  Options effective;
};
Config read_config(const std::filesystem::path &, const Options &overrides = {});
struct Domain {
  int nx, ny, nz, x0, y0, gx, gy, rank = 0, ranks = 1;
  std::size_t count() const;
  std::size_t padded() const;
  std::size_t index(int i, int j, int k) const;
};
struct Model {
  Config config;
  Domain domain;
  std::vector<double> rates, loss, escape, initial;
  std::vector<std::uint8_t> solid;
  double max_loss = 0, max_divergence = 0, initial_sum = 0;
  Model(Config, Domain);
};
struct Moments {
  double mass = 0, min = 0, max = 0;
  std::array<double, 3> mean{}, variance{};
};
Moments moments(const std::vector<double> &, const Model &);
void serial_step(const Model &, const std::vector<double> &, std::vector<double> &, double);
void serial_periodic(const Domain &, const Config &, std::vector<double> &);
void normalize(Model &, double global_sum);
void write_config(const Config &, const std::filesystem::path &);
void write_field(const std::filesystem::path &, const std::vector<double> &, const Model &,
                 bool csv);
double stable_dt(const Config &, double max_loss);
std::string snapshot_name(int);
} // namespace dispersion
