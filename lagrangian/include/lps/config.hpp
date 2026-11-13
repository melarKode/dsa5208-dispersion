#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lps/grid.hpp"

namespace lps {

enum class BoundaryType {
  Periodic,
  Closed,
  Open,  // PROVISIONAL absorbing outflow; urban open-boundary rules still to be agreed
};

enum class VelocityMode {
  Uniform,  // constant (u, v, w) everywhere
  File,
};

// Effective run configuration. Every field is written back to
// effective_config.txt so a run can be reproduced from its output directory.
struct Config {
  std::string case_name = "unnamed";

  int nx = 0, ny = 0, nz = 0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;

  VelocityMode velocity_mode = VelocityMode::Uniform;
  double u = 0.0, v = 0.0, w = 0.0;
  double diffusivity = 0.0;
  std::string case_dir;

  double initial_mass = 1.0;
  double source_x = 0.0, source_y = 0.0, source_z = 0.0;
  double source_sigma = 0.0;
  std::int64_t particle_count = 0;
  std::uint64_t seed = 0;

  double dt = 0.0;
  double end_time = 0.0;
  double output_interval = 0.0;

  std::array<BoundaryType, 3> boundary = {BoundaryType::Periodic, BoundaryType::Periodic, BoundaryType::Periodic};

  std::string output_dir;  // default: output/<case_name>
  bool write_concentration = true;
  bool write_particles = false;
  bool write_binary = false;

  // Keys present in the input that this solver does not use (for example
  // settings meant for the grid solver). Kept and echoed, not rejected.
  std::vector<std::pair<std::string, std::string>> unused_keys;

  Grid grid() const;
};

using KeyValueList = std::vector<std::pair<std::string, std::string>>;

// Reads `key=value` lines. Blank lines and lines whose first non-blank
// character is '#' are ignored. Whitespace around keys and values is trimmed.
// Throws std::runtime_error on a line without '=' or a duplicated key.
KeyValueList read_key_value_file(const std::string& path);

// Applies a command-line override "key=value", replacing an existing key or appending it.
void set_key_value(KeyValueList& entries, const std::string& assignment);

// If `entries` has case_dir: resolves it relative to the config file, merges
// the prepared case's case.cfg geometry (throwing on conflicting values) and
// selects velocity_mode=file. Shared by the serial and MPI drivers.
void merge_case_directory(KeyValueList& entries, const std::string& config_path);

// Converts and validates. Throws std::runtime_error with the offending key.
Config parse_config(const KeyValueList& entries);

void write_effective_config(const Config& config, const std::string& path);

const char* to_string(BoundaryType type);
const char* to_string(VelocityMode mode);

// Shortest decimal text that reads back to exactly the same double; "nan" for NaN.
std::string format_double(double value);

// Timestep bookkeeping derived from dt, end_time and output_interval.
struct TimeControl {
  std::int64_t num_steps = 0;      // end_time / dt
  std::int64_t output_stride = 0;  // output_interval / dt
};

// Requires end_time and output_interval to be integer multiples of dt
// (relative tolerance 1e-9). Throws otherwise.
TimeControl make_time_control(const Config& config);

}  // namespace lps
