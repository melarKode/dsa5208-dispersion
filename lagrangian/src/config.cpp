#include "lps/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lps {

namespace {

std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const auto first = s.find_first_not_of(ws);
  if (first == std::string::npos) return "";
  const auto last = s.find_last_not_of(ws);
  return s.substr(first, last - first + 1);
}

std::runtime_error key_error(const std::string& key, const std::string& message) {
  return std::runtime_error("config key '" + key + "': " + message);
}

// Looks up keys, converts them, and remembers which ones were used.
class KeyReader {
 public:
  explicit KeyReader(const KeyValueList& entries) {
    for (const auto& kv : entries) values_[kv.first] = kv.second;
  }

  bool has(const std::string& key) const { return values_.count(key) != 0; }

  std::string get_string(const std::string& key, const std::string* fallback) {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      if (!fallback) throw key_error(key, "required but missing");
      return *fallback;
    }
    used_.insert(key);
    return it->second;
  }

  std::string required_string(const std::string& key) { return get_string(key, nullptr); }
  std::string optional_string(const std::string& key, const std::string& fallback) {
    return get_string(key, &fallback);
  }

  double required_double(const std::string& key) { return to_double(key, required_string(key)); }
  double optional_double(const std::string& key, double fallback) {
    return has(key) ? to_double(key, required_string(key)) : fallback;
  }

  std::int64_t required_int(const std::string& key) {
    const std::string text = required_string(key);
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || *end != '\0' || errno == ERANGE) throw key_error(key, "not an integer: '" + text + "'");
    return value;
  }

  std::uint64_t required_uint64(const std::string& key) {
    const std::string text = required_string(key);
    errno = 0;
    char* end = nullptr;
    if (text.empty() || text[0] == '-') throw key_error(key, "not a non-negative integer: '" + text + "'");
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE) throw key_error(key, "not a non-negative integer: '" + text + "'");
    return value;
  }

  bool optional_bool(const std::string& key, bool fallback) {
    if (!has(key)) return fallback;
    const std::string text = required_string(key);
    if (text == "true" || text == "1" || text == "yes") return true;
    if (text == "false" || text == "0" || text == "no") return false;
    throw key_error(key, "expected true/false, got '" + text + "'");
  }

  BoundaryType optional_boundary(const std::string& key) {
    const std::string text = optional_string(key, "periodic");
    if (text == "periodic") return BoundaryType::Periodic;
    if (text == "closed") return BoundaryType::Closed;
    if (text == "open") return BoundaryType::Open;
    throw key_error(key, "expected periodic, closed or open, got '" + text + "'");
  }

  std::vector<std::pair<std::string, std::string>> unused(const KeyValueList& entries) const {
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& kv : entries)
      if (!used_.count(kv.first)) result.push_back(kv);
    return result;
  }

 private:
  static double to_double(const std::string& key, const std::string& text) {
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || *end != '\0' || errno == ERANGE || !std::isfinite(value))
      throw key_error(key, "not a finite number: '" + text + "'");
    return value;
  }

  std::map<std::string, std::string> values_;
  std::set<std::string> used_;
};

void require(bool condition, const std::string& key, const std::string& message) {
  if (!condition) throw key_error(key, message);
}

}  // namespace

Grid Config::grid() const {
  Grid g;
  g.nx = nx;
  g.ny = ny;
  g.nz = nz;
  g.dx = dx;
  g.dy = dy;
  g.dz = dz;
  g.origin_x = origin_x;
  g.origin_y = origin_y;
  g.origin_z = origin_z;
  return g;
}

KeyValueList read_key_value_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config file '" + path + "'");
  KeyValueList entries;
  std::set<std::string> seen;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const std::string text = trim(line);
    if (text.empty() || text[0] == '#') continue;
    const auto eq = text.find('=');
    const std::string where = path + ":" + std::to_string(line_number);
    if (eq == std::string::npos) throw std::runtime_error(where + ": expected key=value, got '" + text + "'");
    const std::string key = trim(text.substr(0, eq));
    const std::string value = trim(text.substr(eq + 1));
    if (key.empty()) throw std::runtime_error(where + ": empty key");
    if (!seen.insert(key).second) throw std::runtime_error(where + ": duplicate key '" + key + "'");
    entries.emplace_back(key, value);
  }
  return entries;
}

void set_key_value(KeyValueList& entries, const std::string& assignment) {
  const auto eq = assignment.find('=');
  if (eq == std::string::npos) throw std::runtime_error("override must be key=value, got '" + assignment + "'");
  const std::string key = trim(assignment.substr(0, eq));
  const std::string value = trim(assignment.substr(eq + 1));
  if (key.empty()) throw std::runtime_error("override has an empty key: '" + assignment + "'");
  for (auto& kv : entries) {
    if (kv.first == key) {
      kv.second = value;
      return;
    }
  }
  entries.emplace_back(key, value);
}

void merge_case_directory(KeyValueList& entries, const std::string& config_path) {
  namespace fs = std::filesystem;
  for(const auto& kv:KeyValueList(entries)) if(kv.first=="case_dir") {
    auto path=fs::path(kv.second);
    if(path.is_relative()) path=fs::path(config_path).parent_path()/path;
    path=fs::absolute(path);
    for(const auto& geometry:read_key_value_file((path/"case.cfg").string())) {
      for(const auto& existing:entries) if(existing.first==geometry.first && std::stod(existing.second)!=std::stod(geometry.second))
        throw std::runtime_error("Conflicting case geometry: "+geometry.first);
      set_key_value(entries,geometry.first+"="+geometry.second);
    }
    set_key_value(entries,"case_dir="+path.string());
    set_key_value(entries,"velocity_mode=file");
  }
}

Config parse_config(const KeyValueList& entries) {
  KeyReader r(entries);
  Config c;

  c.case_name = r.optional_string("case_name", c.case_name);

  c.nx = static_cast<int>(r.required_int("nx"));
  c.ny = static_cast<int>(r.required_int("ny"));
  c.nz = static_cast<int>(r.required_int("nz"));
  c.dx = r.required_double("dx");
  c.dy = r.required_double("dy");
  c.dz = r.required_double("dz");
  c.origin_x = r.optional_double("origin_x", 0.0);
  c.origin_y = r.optional_double("origin_y", 0.0);
  c.origin_z = r.optional_double("origin_z", 0.0);

  const std::string mode = r.optional_string("velocity_mode", "uniform");
  if (mode != "uniform" && mode != "file") throw key_error("velocity_mode", "expected uniform or file");
  c.velocity_mode = mode == "file" ? VelocityMode::File : VelocityMode::Uniform;
  c.case_dir = r.optional_string("case_dir", "");
  if(r.has("particle_boundary_model") && r.required_string("particle_boundary_model")!="split_advective_open_reflected_diffusion")
    throw key_error("particle_boundary_model", "unsupported model");
  if (c.velocity_mode == VelocityMode::File && c.case_dir.empty()) throw key_error("case_dir", "required for file wind");
  c.u = r.optional_double("u", 0);
  c.v = r.optional_double("v", 0);
  c.w = r.optional_double("w", 0);
  c.diffusivity = r.required_double("diffusivity");

  c.initial_mass = r.optional_double("initial_mass", 1.0);
  c.source_x = r.required_double("source_x");
  c.source_y = r.required_double("source_y");
  c.source_z = r.required_double("source_z");
  c.source_sigma = r.required_double("source_sigma");
  c.particle_count = r.required_int("particle_count");
  c.seed = r.required_uint64("seed");

  c.dt = r.required_double("dt");
  c.end_time = r.required_double("end_time");
  c.output_interval = r.required_double("output_interval");

  c.boundary[0] = r.optional_boundary("boundary_x");
  c.boundary[1] = r.optional_boundary("boundary_y");
  c.boundary[2] = r.optional_boundary("boundary_z");

  c.output_dir = r.optional_string("output_dir", "output/" + c.case_name);
  c.write_concentration = r.optional_bool("write_concentration", true);
  c.write_particles = r.optional_bool("write_particles", false);
  c.write_binary = r.optional_bool("write_binary", false);

  c.unused_keys = r.unused(entries);

  require(c.nx > 0, "nx", "must be positive");
  require(c.ny > 0, "ny", "must be positive");
  require(c.nz > 0, "nz", "must be positive");
  require(c.dx > 0, "dx", "must be positive");
  require(c.dy > 0, "dy", "must be positive");
  require(c.dz > 0, "dz", "must be positive");
  require(c.diffusivity >= 0, "diffusivity", "must be non-negative");
  require(c.initial_mass > 0, "initial_mass", "must be positive");
  require(c.source_sigma >= 0, "source_sigma", "must be non-negative");
  require(c.particle_count > 0, "particle_count", "must be positive");
  require(c.dt > 0, "dt", "must be positive");
  require(c.end_time >= 0, "end_time", "must be non-negative");
  require(c.output_interval > 0, "output_interval", "must be positive");

  int i, j, k;
  if (!c.grid().locate({c.source_x, c.source_y, c.source_z}, i, j, k))
    std::cerr << "warning: source centre lies outside the domain\n";
  for (const auto& kv : c.unused_keys)
    std::cerr << "warning: config key '" << kv.first << "' is not used by the particle solver\n";
  return c;
}

const char* to_string(BoundaryType type) {
  switch (type) {
    case BoundaryType::Periodic: return "periodic";
    case BoundaryType::Open: return "open";
    case BoundaryType::Closed: return "closed";
  }
  return "?";
}

const char* to_string(VelocityMode mode) {
  switch (mode) {
    case VelocityMode::Uniform: return "uniform";
    case VelocityMode::File: return "file";
  }
  return "?";
}

std::string format_double(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  char buffer[40];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    if (std::strtod(buffer, nullptr) == value) break;
  }
  return buffer;
}

void write_effective_config(const Config& c, const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  const auto d = [](double x) { return format_double(x); };
  out << "# Effective configuration written by lagrangian_solver (defaults filled in).\n"
      << "case_name=" << c.case_name << "\n"
      << "nx=" << c.nx << "\nny=" << c.ny << "\nnz=" << c.nz << "\n"
      << "dx=" << d(c.dx) << "\ndy=" << d(c.dy) << "\ndz=" << d(c.dz) << "\n"
      << "origin_x=" << d(c.origin_x) << "\norigin_y=" << d(c.origin_y) << "\norigin_z=" << d(c.origin_z) << "\n"
      << "velocity_mode=" << to_string(c.velocity_mode) << "\n"
      << "u=" << d(c.u) << "\nv=" << d(c.v) << "\nw=" << d(c.w) << "\n"
      << "diffusivity=" << d(c.diffusivity) << "\n"
      << "initial_mass=" << d(c.initial_mass) << "\n"
      << "source_x=" << d(c.source_x) << "\nsource_y=" << d(c.source_y) << "\nsource_z=" << d(c.source_z) << "\n"
      << "source_sigma=" << d(c.source_sigma) << "\n"
      << "particle_count=" << c.particle_count << "\n"
      << "seed=" << c.seed << "\n"
      << "dt=" << d(c.dt) << "\nend_time=" << d(c.end_time) << "\noutput_interval=" << d(c.output_interval) << "\n"
      << "boundary_x=" << to_string(c.boundary[0]) << "\n"
      << "boundary_y=" << to_string(c.boundary[1]) << "\n"
      << "boundary_z=" << to_string(c.boundary[2]) << "\n"
      << "output_dir=" << c.output_dir << "\n"
      << "write_concentration=" << (c.write_concentration ? "true" : "false") << "\n"
      << "write_binary=" << (c.write_binary ? "true" : "false") << "\n"
      << "write_particles=" << (c.write_particles ? "true" : "false") << "\n";
  if (c.velocity_mode == VelocityMode::File)
    out << "case_dir=" << c.case_dir << "\nparticle_boundary_model=split_advective_open_reflected_diffusion\n";
  if (!c.unused_keys.empty()) {
    out << "# Keys not used by the particle solver (kept for the shared contract):\n";
    for (const auto& kv : c.unused_keys) out << kv.first << "=" << kv.second << "\n";
  }
}

TimeControl make_time_control(const Config& c) {
  const auto integer_multiple = [&](double value, const char* key) {
    const double ratio = value / c.dt;
    const double rounded = std::round(ratio);
    if (std::fabs(ratio - rounded) > 1e-9 * std::fmax(1.0, ratio))
      throw key_error(key, "must be an integer multiple of dt (" + format_double(c.dt) + ")");
    return static_cast<std::int64_t>(rounded);
  };
  TimeControl tc;
  tc.num_steps = integer_multiple(c.end_time, "end_time");
  tc.output_stride = integer_multiple(c.output_interval, "output_interval");
  if (tc.output_stride < 1) throw key_error("output_interval", "must be at least dt");
  // Step numbers are a 32-bit field of the RNG counter.
  if (tc.num_steps >= (std::int64_t{1} << 32)) throw key_error("end_time", "more than 2^32 - 1 timesteps");
  return tc;
}

}  // namespace lps
