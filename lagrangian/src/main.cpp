// Serial Lagrangian particle solver for passive pollutant dispersion:
//   dc/dt + div(u c) = K laplacian(c)
// See README.md for usage and docs/INTERFACES.md for the file contracts.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lps/config.hpp"
#include "lps/diagnostics.hpp"
#include "lps/domain.hpp"
#include "lps/particles.hpp"
#include "lps/stepper.hpp"
#include "lps/wind.hpp"
#include "lps/urban.hpp"

namespace {

void print_usage(const char* program) {
  std::cerr << "usage: " << program << " CONFIG [--output DIR] [--set key=value]...\n"
            << "  CONFIG          key=value configuration file\n"
            << "  --output DIR    output directory (overrides output_dir)\n"
            << "  --set key=value override or add one config key (repeatable)\n";
}

std::string numbered(const std::string& stem, long long index) {
  return lps::snapshot_filename(stem, index, ".csv");
}

}  // namespace

int main(int argc, char** argv) {
  using namespace lps;
  namespace fs = std::filesystem;

  std::string config_path;
  std::string output_override;
  std::vector<std::string> overrides;
  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--output" && a + 1 < argc) {
      output_override = argv[++a];
    } else if (arg == "--set" && a + 1 < argc) {
      overrides.push_back(argv[++a]);
    } else if (config_path.empty() && arg.rfind("--", 0) != 0) {
      config_path = arg;
    } else {
      print_usage(argv[0]);
      return 2;
    }
  }
  if (config_path.empty()) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    KeyValueList entries = read_key_value_file(config_path);
    for (const auto& assignment : overrides) set_key_value(entries, assignment);
    merge_case_directory(entries, config_path);
    Config config = parse_config(entries);
    if (!output_override.empty()) config.output_dir = output_override;
    const TimeControl time_control = make_time_control(config);

    const fs::path out_dir(config.output_dir);
    if(fs::exists(out_dir) && !fs::is_empty(out_dir)) throw std::runtime_error("Output directory must be empty");
    fs::create_directories(out_dir);
    if(config.velocity_mode==VelocityMode::File)
      fs::copy_file(fs::path(config.case_dir)/"input.sha256",out_dir/"input.sha256");
    write_effective_config(config, (out_dir / "effective_config.txt").string());

    const Grid grid = config.grid();
    const Domain domain(grid, config.boundary, load_solid(config));
    const std::unique_ptr<WindField> wind = make_wind_field(config);

    const auto wall_start = std::chrono::steady_clock::now();
    ReleaseResult release = make_gaussian_release(config, domain);
    ParticleSet& particles = release.particles;
    CompensatedSum mass_escaped;
    mass_escaped.add(release.escaped_mass);
    std::int64_t escaped_particles = release.escaped_particles;
    std::int64_t periodic_wraps = 0;

    DiagnosticsWriter diagnostics((out_dir / "diagnostics.csv").string());
    std::ofstream snapshots((out_dir / "snapshots.csv").string());
    snapshots << (config.write_binary ? "snapshot_index,step,time,format,ranks\n" : "snapshot_index,step,time\n");
    long long snapshot_index = 0;
    double io_seconds = 0.0;

    const auto write_output = [&](std::int64_t step) {
      const auto io_start = std::chrono::steady_clock::now();
      const double time = static_cast<double>(step) * config.dt;
      const auto row=compute_diagnostics(particles, grid, time, config.initial_mass, mass_escaped.result());
      diagnostics.write(row);
      if(config.velocity_mode==VelocityMode::File)
        std::cout << "t=" << time << " mass=" << row.mass_inside << " escaped=" << row.mass_escaped
                  << " residual=" << row.mass_residual << std::endl;
      snapshots << snapshot_index << ',' << step << ',' << format_double(time);
      if(config.write_binary) snapshots << (config.write_concentration ? ",binary,1" : ",none,1");
      snapshots << '\n';
      if(config.write_concentration && config.write_binary) {
        write_concentration_binary((out_dir / snapshot_filename("concentration", snapshot_index, ".bin")).string(),
                                   grid, deposit_mass_ngp(particles, grid));
      } else if (config.write_concentration)
        write_concentration_csv((out_dir / numbered("concentration", snapshot_index)).string(), grid,
                                deposit_mass_ngp(particles, grid));
      if (config.write_particles)
        write_particles_csv((out_dir / numbered("particles", snapshot_index)).string(), particles, grid);
      ++snapshot_index;
      io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
    };

    std::cout << "case " << config.case_name << ": " << config.particle_count << " particles, "
              << time_control.num_steps << " steps of dt=" << format_double(config.dt) << " s, wind "
              << wind->description() << ", K=" << format_double(config.diffusivity) << " m^2/s\n";

    const StepParameters params{config.seed, config.dt, config.diffusivity};
    write_output(0);
    for (std::int64_t n = 0; n < time_control.num_steps; ++n) {
      const StepResult step = advance_particles(particles, *wind, domain, params, static_cast<std::uint32_t>(n));
      mass_escaped.add(step.escaped_mass);
      escaped_particles += step.escaped_particles;
      periodic_wraps += step.periodic_wraps;
      const std::int64_t completed = n + 1;
      if (completed % time_control.output_stride == 0 || completed == time_control.num_steps)
        write_output(completed);
    }
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    std::int64_t active = 0, particles_off_home_image = 0;
    for (std::size_t p = 0; p < particles.size(); ++p) {
      active += particles.active[p];
      if (particles.image_x[p] != 0 || particles.image_y[p] != 0 || particles.image_z[p] != 0)
        ++particles_off_home_image;
    }

    std::ofstream summary((out_dir / "run_summary.txt").string());
    summary << "case_name=" << config.case_name << "\n"
            << "particle_count=" << config.particle_count << "\n"
            << "num_steps=" << time_control.num_steps << "\n"
            << "num_snapshots=" << snapshot_index << "\n"
            << "final_active_particles=" << active << "\n"
            << "escaped_particles=" << escaped_particles << "\n"
            << "final_mass_escaped=" << format_double(mass_escaped.result()) << "\n"
            << "periodic_wraps=" << periodic_wraps << "\n"
            << "particles_off_home_image=" << particles_off_home_image << "\n"
            << "rng=philox4x32-10 counter={id_lo,id_hi,step,stream} key={seed_lo,seed_hi}\n"
            << "wall_seconds=" << format_double(wall_seconds) << "\n"
            << "output_seconds=" << format_double(io_seconds) << "\n";

    std::cout << "done in " << wall_seconds << " s (" << io_seconds << " s output); " << active << " active, "
              << escaped_particles << " escaped, " << periodic_wraps << " periodic wraps -> " << out_dir.string()
              << "\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
