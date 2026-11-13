#include "dispersion/core.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace dispersion {
namespace {
std::string trim(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
Options parse(const std::filesystem::path &p) {
  std::ifstream f(p);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  Options out;
  std::string line;
  int number = 0;
  while (std::getline(f, line)) {
    ++number;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    auto pos = line.find('=');
    if (pos == std::string::npos)
      throw std::runtime_error("Expected key=value at line " + std::to_string(number));
    auto key = trim(line.substr(0, pos)), value = trim(line.substr(pos + 1));
    if (key.empty() || value.empty() || !out.emplace(key, value).second)
      throw std::runtime_error("Empty or duplicate key: " + key);
  }
  return out;
}
double number(const std::string &s) {
  std::size_t n = 0;
  double v = std::stod(s, &n);
  if (n != s.size() || !std::isfinite(v))
    throw std::runtime_error("Invalid number: " + s);
  return v;
}
int integer(const std::string &s) {
  double v = number(s);
  if (v != std::floor(v) || v > 100000000 || v < -1)
    throw std::runtime_error("Invalid integer: " + s);
  return static_cast<int>(v);
}
bool little_endian() {
  std::uint16_t one = 1;
  return *reinterpret_cast<unsigned char *>(&one) == 1;
}
int wrap(int x, int n) {
  return (x % n + n) % n;
}
std::size_t checked_size(int x, int y, int z) {
  const auto a = static_cast<std::size_t>(x), b = static_cast<std::size_t>(y),
             c = static_cast<std::size_t>(z);
  if (a > std::numeric_limits<std::size_t>::max() / b / c / 8)
    throw std::runtime_error("Grid size overflow");
  return a * b * c;
}
template <class T>
std::vector<T> load_padded(const Config &c, const Domain &d, const std::string &filename) {
  auto path = std::filesystem::path(c.case_dir) / filename;
  const auto bytes = checked_size(c.nx, c.ny, c.nz) * sizeof(T);
  if (!little_endian() || std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("Invalid binary field size/endianness: " + path.string());
  std::ifstream f(path, std::ios::binary);
  std::vector<T> data(d.padded(), T{});
  for (int k = 0; k < d.nz + 2; ++k)
    for (int j = 0; j < d.ny + 2; ++j) {
      int gz = k - 1, gy = d.y0 + j - 1;
      if (gz < 0 || gz >= c.nz) {
        if (c.boundary[2] != "periodic")
          continue;
        gz = wrap(gz, c.nz);
      }
      if (gy < 0 || gy >= c.ny) {
        if (c.boundary[1] != "periodic")
          continue;
        gy = wrap(gy, c.ny);
      }
      for (int i = 0; i < d.nx + 2;) {
        int gx = d.x0 + i - 1;
        if ((gx < 0 || gx >= c.nx) && c.boundary[0] != "periodic") {
          ++i;
          continue;
        }
        gx = wrap(gx, c.nx);
        int len = std::min(d.nx + 2 - i, c.nx - gx);
        auto offset = (static_cast<std::size_t>(gz) * c.ny + gy) * c.nx + gx;
        f.seekg(static_cast<std::streamoff>(offset * sizeof(T)));
        f.read(reinterpret_cast<char *>(&data[d.index(i, j, k)]),
               static_cast<std::streamsize>(len * sizeof(T)));
        if (!f)
          throw std::runtime_error("Failed binary read: " + path.string());
        i += len;
      }
    }
  return data;
}
} // namespace
Config read_config(const std::filesystem::path &path, const Options &overrides) {
  Config c;
  auto opts = parse(path);
  for (const auto &[k, v] : overrides)
    opts[k] = v;
  if (opts.count("case_dir")) {
    auto dir = std::filesystem::path(opts.at("case_dir"));
    if (dir.is_relative())
      dir = path.parent_path() / dir;
    c.case_dir = std::filesystem::absolute(dir).lexically_normal().string();
    if (std::filesystem::exists(dir / "INCOMPLETE"))
      throw std::runtime_error("Case conversion is incomplete");
    auto meta = parse(dir / "case.cfg");
    for (const auto *key : {"nx", "ny", "nz", "dx", "dy", "dz", "origin_x", "origin_y", "origin_z"})
      if (!meta.count(key))
        throw std::runtime_error(std::string("Case missing geometry key: ") + key);
    for (const auto &[key, value] : meta) {
      if (key != "nx" && key != "ny" && key != "nz" && key != "dx" && key != "dy" && key != "dz" &&
          key != "origin_x" && key != "origin_y" && key != "origin_z")
        throw std::runtime_error("Unknown case geometry key: " + key);
      if (opts.count(key) && number(opts.at(key)) != number(value))
        throw std::runtime_error("Config disagrees with case geometry: " + key);
      opts[key] = value;
    }
    opts["velocity_mode"] = "file";
    opts["case_dir"] = c.case_dir;
  }
  for (const auto &[key, value] : opts) {
    if (key == "nx")
      c.nx = integer(value);
    else if (key == "ny")
      c.ny = integer(value);
    else if (key == "nz")
      c.nz = integer(value);
    else if (key == "dx")
      c.dx = number(value);
    else if (key == "dy")
      c.dy = number(value);
    else if (key == "dz")
      c.dz = number(value);
    else if (key == "origin_x")
      c.ox = number(value);
    else if (key == "origin_y")
      c.oy = number(value);
    else if (key == "origin_z")
      c.oz = number(value);
    else if (key == "u")
      c.u = number(value);
    else if (key == "v")
      c.v = number(value);
    else if (key == "w")
      c.w = number(value);
    else if (key == "diffusivity")
      c.diffusion = number(value);
    else if (key == "initial_mass")
      c.mass = number(value);
    else if (key == "source_x")
      c.sx = number(value);
    else if (key == "source_y")
      c.sy = number(value);
    else if (key == "source_z")
      c.sz = number(value);
    else if (key == "source_sigma")
      c.sigma = number(value);
    else if (key == "dt")
      c.dt = number(value);
    else if (key == "end_time")
      c.end = number(value);
    else if (key == "output_interval")
      c.output_interval = number(value);
    else if (key == "cfl")
      c.cfl = number(value);
    else if (key == "boundary_x")
      c.boundary[0] = value;
    else if (key == "boundary_y")
      c.boundary[1] = value;
    else if (key == "boundary_z")
      c.boundary[2] = value;
    else if (key == "initial")
      c.initial = value;
    else if (key == "velocity_mode")
      c.velocity = value;
    else if (key == "case_dir")
      c.case_dir = value;
    else if (key == "output_dir")
      c.output = value;
    else if (key == "decomposition")
      c.decomposition = value;
    else if (key == "exchange")
      c.exchange = value;
    else if (key == "snapshots")
      c.snapshots = value;
    else if (key == "wall_i")
      c.wall_i = integer(value);
    else if (key == "px")
      c.px = integer(value);
    else if (key == "py")
      c.py = integer(value);
    else if (key == "amplitude")
      c.amplitude = number(value);
    else if (key != "case_name" && key != "particle_count" && key != "seed")
      throw std::runtime_error("Unknown configuration key: " + key);
  }
  if (c.nx < 1 || c.ny < 1 || c.nz < 1 || c.dx <= 0 || c.dy <= 0 || c.dz <= 0 || c.sigma <= 0 ||
      c.mass <= 0 || c.diffusion < 0 || c.dt < 0 || c.end < 0 || c.output_interval <= 0 ||
      c.cfl <= 0 || c.cfl > 1 || c.px < 0 || c.py < 0)
    throw std::runtime_error("Invalid dimensions, time, mass, diffusivity, or CFL");
  checked_size(c.nx + 2, c.ny + 2, c.nz + 2);
  for (auto &b : c.boundary)
    if (b != "periodic" && b != "closed" && b != "open")
      throw std::runtime_error("Unknown boundary: " + b);
  if (c.initial != "gaussian" && c.initial != "uniform" && c.initial != "cosine")
    throw std::runtime_error("Unknown initial condition");
  if (c.velocity != "uniform" && c.velocity != "file")
    throw std::runtime_error("Unknown velocity mode");
  if (c.velocity == "file" && c.case_dir.empty())
    throw std::runtime_error("File velocity needs case_dir");
  if (c.wall_i >= c.nx)
    throw std::runtime_error("Wall index outside domain");
  if (c.decomposition != "slab" && c.decomposition != "block")
    throw std::runtime_error("Unknown decomposition");
  if (c.exchange != "blocking" && c.exchange != "overlap")
    throw std::runtime_error("Unknown exchange mode");
  if (c.snapshots != "none" && c.snapshots != "binary" && c.snapshots != "csv")
    throw std::runtime_error("Unknown snapshots format");
  if (std::abs(c.amplitude) > 1)
    throw std::runtime_error("Cosine amplitude must be between -1 and 1");
  c.effective = opts;
  return c;
}
std::size_t Domain::count() const {
  return checked_size(nx, ny, nz);
}
std::size_t Domain::padded() const {
  return checked_size(nx + 2, ny + 2, nz + 2);
}
std::size_t Domain::index(int i, int j, int k) const {
  return (static_cast<std::size_t>(k) * (ny + 2) + j) * (nx + 2) + i;
}
Model::Model(Config c, Domain d)
    : config(std::move(c)), domain(d), rates(d.count() * 6), loss(d.count()), escape(d.count()),
      initial(d.padded()), solid(d.padded()) {
  std::array<std::vector<double>, 3> wind;
  const std::array<double, 3> speed{config.u, config.v, config.w};
  if (config.velocity == "file") {
    wind[0] = load_padded<double>(config, d, "u.bin");
    wind[1] = load_padded<double>(config, d, "v.bin");
    wind[2] = load_padded<double>(config, d, "w.bin");
    solid = load_padded<std::uint8_t>(config, d, "solid.bin");
    for (auto mask : solid)
      if (mask > 1)
        throw std::runtime_error("Solid mask must be 0 or 1");
    for (int a = 0; a < 3; ++a)
      for (std::size_t p = 0; p < solid.size(); ++p)
        if (!solid[p] && !std::isfinite(wind[a][p]))
          throw std::runtime_error("Nonfinite wind in air");
  }
  if (config.wall_i >= 0)
    for (int k = 0; k < d.nz + 2; ++k)
      for (int j = 0; j < d.ny + 2; ++j)
        for (int i = 0; i < d.nx + 2; ++i)
          if (wrap(d.x0 + i - 1, config.nx) == config.wall_i)
            solid[d.index(i, j, k)] = 1;
  const std::array<int, 3> extent{config.nx, config.ny, config.nz};
  const std::array<double, 3> h{config.dx, config.dy, config.dz},
      origin{config.ox, config.oy, config.oz}, source{config.sx, config.sy, config.sz};
  const double pi = std::acos(-1.0);
  for (int k = 1; k <= d.nz; ++k)
    for (int j = 1; j <= d.ny; ++j)
      for (int i = 1; i <= d.nx; ++i) {
        const auto p = d.index(i, j, k),
                   q = (static_cast<std::size_t>(k - 1) * d.ny + j - 1) * d.nx + i - 1;
        if (solid[p])
          continue;
        const std::array<int, 3> g{d.x0 + i - 1, d.y0 + j - 1, k - 1};
        double div = 0;
        for (int a = 0; a < 3; ++a)
          for (int side = 0; side < 2; ++side) {
            const int sign = side ? 1 : -1, face = 2 * a + side;
            bool outside = g[a] + sign < 0 || g[a] + sign >= extent[a];
            if (outside && config.boundary[a] == "closed")
              continue;
            if (outside && config.boundary[a] == "open") {
              const double un = sign * (config.velocity == "file" ? wind[a][p] : speed[a]);
              const double rate = std::max(un, 0.0) / h[a];
              loss[q] += rate;
              escape[q] += rate;
              div += un / h[a];
              continue;
            }
            int ni = i, nj = j, nk = k;
            if (a == 0)
              ni += sign;
            if (a == 1)
              nj += sign;
            if (a == 2)
              nk += sign;
            const auto nb = d.index(ni, nj, nk);
            if (solid[nb])
              continue;
            const double un =
                sign * (config.velocity == "file" ? .5 * (wind[a][p] + wind[a][nb]) : speed[a]);
            const double diff = config.diffusion / (h[a] * h[a]);
            rates[6 * q + face] = std::max(-un, 0.0) / h[a] + diff;
            loss[q] += std::max(un, 0.0) / h[a] + diff;
            div += un / h[a];
          }
        if (!std::isfinite(loss[q]) || !std::isfinite(div))
          throw std::runtime_error("Transport coefficients overflow");
        max_loss = std::max(max_loss, loss[q]);
        max_divergence = std::max(max_divergence, std::abs(div));
        double value = 1;
        if (config.initial == "gaussian") {
          for (int a = 0; a < 3; ++a) {
            const double x = origin[a] + (g[a] + .5) * h[a], L = extent[a] * h[a];
            double weight = 0;
            const int images = config.boundary[a] == "periodic"
                                   ? static_cast<int>(std::ceil(8 * config.sigma / L)) + 1
                                   : 0;
            const double delta =
                config.boundary[a] == "periodic" ? std::remainder(x - source[a], L) : x - source[a];
            for (int m = -images; m <= images; ++m) {
              double lo = (delta + m * L - .5 * h[a]) / (std::sqrt(2.0) * config.sigma),
                     hi = lo + h[a] / (std::sqrt(2.0) * config.sigma);
              weight += .5 * (std::erf(hi) - std::erf(lo)) / h[a];
            }
            value *= weight;
          }
        } else if (config.initial == "cosine") {
          double product = 1;
          for (int a = 0; a < 3; ++a) {
            double wave = 2 * pi / extent[a];
            product *= std::cos(wave * (g[a] + .5)) * std::sin(.5 * wave) / (.5 * wave);
          }
          value = 1 + config.amplitude * product;
        }
        initial[p] = std::max(0.0, value);
        initial_sum += initial[p] * config.dx * config.dy * config.dz;
      }
}
void normalize(Model &m, double sum) {
  if (!(sum > 0) || !std::isfinite(sum))
    throw std::runtime_error("Initial cloud has no representable mass in air");
  for (auto &x : m.initial)
    x *= m.config.mass / sum;
}
double stable_dt(const Config &c, double loss) {
  const double safe = loss > 0 ? c.cfl / loss : std::numeric_limits<double>::infinity();
  if (c.dt > safe * (1 + 1e-12))
    throw std::runtime_error("Requested dt exceeds combined positivity bound; set dt=0 for "
                             "automatic stepping. Safe dt=" +
                             std::to_string(safe));
  return c.dt > 0 ? c.dt : safe;
}
void serial_periodic(const Domain &d, const Config &c, std::vector<double> &f) {
  if (c.boundary[0] == "periodic")
    for (int k = 1; k <= d.nz; ++k)
      for (int j = 1; j <= d.ny; ++j) {
        f[d.index(0, j, k)] = f[d.index(d.nx, j, k)];
        f[d.index(d.nx + 1, j, k)] = f[d.index(1, j, k)];
      }
  if (c.boundary[1] == "periodic")
    for (int k = 1; k <= d.nz; ++k)
      for (int i = 1; i <= d.nx; ++i) {
        f[d.index(i, 0, k)] = f[d.index(i, d.ny, k)];
        f[d.index(i, d.ny + 1, k)] = f[d.index(i, 1, k)];
      }
  if (c.boundary[2] == "periodic")
    for (int j = 1; j <= d.ny; ++j)
      for (int i = 1; i <= d.nx; ++i) {
        f[d.index(i, j, 0)] = f[d.index(i, j, d.nz)];
        f[d.index(i, j, d.nz + 1)] = f[d.index(i, j, 1)];
      }
}
void serial_step(const Model &m, const std::vector<double> &current, std::vector<double> &next,
                 double dt) {
  const auto &d = m.domain;
  for (int k = 1; k <= d.nz; ++k)
    for (int j = 1; j <= d.ny; ++j)
      for (int i = 1; i <= d.nx; ++i) {
        const auto p = d.index(i, j, k),
                   q = (static_cast<std::size_t>(k - 1) * d.ny + j - 1) * d.nx + i - 1;
        double change = -m.loss[q] * current[p];
        change += m.rates[6 * q] * current[d.index(i - 1, j, k)] +
                  m.rates[6 * q + 1] * current[d.index(i + 1, j, k)];
        change += m.rates[6 * q + 2] * current[d.index(i, j - 1, k)] +
                  m.rates[6 * q + 3] * current[d.index(i, j + 1, k)];
        change += m.rates[6 * q + 4] * current[d.index(i, j, k - 1)] +
                  m.rates[6 * q + 5] * current[d.index(i, j, k + 1)];
        next[p] = current[p] + dt * change;
      }
}
Moments moments(const std::vector<double> &f, const Model &m) {
  Moments r;
  r.min = std::numeric_limits<double>::infinity();
  const auto &d = m.domain;
  const auto &c = m.config;
  for (int k = 1; k <= d.nz; ++k)
    for (int j = 1; j <= d.ny; ++j)
      for (int i = 1; i <= d.nx; ++i) {
        auto p = d.index(i, j, k);
        if (m.solid[p])
          continue;
        double value = f[p];
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite concentration");
        double mass = value * c.dx * c.dy * c.dz;
        r.mass += mass;
        r.min = std::min(r.min, value);
        r.max = std::max(r.max, value);
        std::array<double, 3> x{c.ox + (d.x0 + i - .5) * c.dx, c.oy + (d.y0 + j - .5) * c.dy,
                                c.oz + (k - .5) * c.dz};
        for (int a = 0; a < 3; ++a) {
          r.mean[a] += mass * x[a];
          r.variance[a] += mass * x[a] * x[a];
        }
      }
  return r;
}
std::string snapshot_name(int index) {
  std::ostringstream s;
  s << "concentration_" << std::setw(6) << std::setfill('0') << index;
  return s.str();
}
void write_field(const std::filesystem::path &path, const std::vector<double> &f, const Model &m,
                 bool csv) {
  const auto &d = m.domain;
  const auto &c = m.config;
  std::ofstream out(path, csv ? std::ios::out : std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot create snapshot");
  if (csv)
    out << "cell_id,i,j,k,mass,concentration\n" << std::setprecision(17);
  for (int k = 1; k <= d.nz; ++k)
    for (int j = 1; j <= d.ny; ++j) {
      if (!csv)
        out.write(reinterpret_cast<const char *>(&f[d.index(1, j, k)]), d.nx * sizeof(double));
      else
        for (int i = 1; i <= d.nx; ++i) {
          auto id = (static_cast<std::size_t>(k - 1) * c.ny + d.y0 + j - 1) * c.nx + d.x0 + i - 1;
          auto v = f[d.index(i, j, k)];
          out << id << ',' << d.x0 + i - 1 << ',' << d.y0 + j - 1 << ',' << k - 1 << ','
              << v * c.dx * c.dy * c.dz << ',' << v << '\n';
        }
    }
  if (!out)
    throw std::runtime_error("Snapshot write failed");
}
void write_config(const Config &c, const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("Cannot write effective configuration");
  f << std::setprecision(17);
  f << "nx=" << c.nx << "\nny=" << c.ny << "\nnz=" << c.nz << "\ndx=" << c.dx << "\ndy=" << c.dy
    << "\ndz=" << c.dz << "\norigin_x=" << c.ox << "\norigin_y=" << c.oy << "\norigin_z=" << c.oz
    << "\nu=" << c.u << "\nv=" << c.v << "\nw=" << c.w << "\ndiffusivity=" << c.diffusion
    << "\ninitial_mass=" << c.mass << "\nsource_x=" << c.sx << "\nsource_y=" << c.sy
    << "\nsource_z=" << c.sz << "\nsource_sigma=" << c.sigma << "\ndt=" << c.dt
    << "\nend_time=" << c.end << "\noutput_interval=" << c.output_interval << "\ncfl=" << c.cfl
    << "\nboundary_x=" << c.boundary[0] << "\nboundary_y=" << c.boundary[1]
    << "\nboundary_z=" << c.boundary[2] << "\ninitial=" << c.initial
    << "\nvelocity_mode=" << c.velocity << "\noutput_dir=" << c.output
    << "\ndecomposition=" << c.decomposition << "\nexchange=" << c.exchange
    << "\nsnapshots=" << c.snapshots << "\nwall_i=" << c.wall_i << "\npx=" << c.px
    << "\npy=" << c.py << "\namplitude=" << c.amplitude << '\n';
  if (!c.case_dir.empty())
    f << "case_dir=" << c.case_dir << '\n';
}
} // namespace dispersion
