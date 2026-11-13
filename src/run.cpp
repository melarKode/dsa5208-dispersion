#include "dispersion/run.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/resource.h>
namespace dispersion {
Options command_overrides(int argc, char **argv) {
  Options opts;
  for (int a = 2; a < argc; ++a) {
    std::string text = argv[a];
    auto p = text.find('=');
    if (p == std::string::npos || p == 0 || p + 1 == text.size())
      throw std::runtime_error("Overrides must be key=value");
    opts[text.substr(0, p)] = text.substr(p + 1);
  }
  return opts;
}
void run(Model &m, Engine &engine) {
  auto &c = m.config;
  const auto &d = m.domain;
  const auto dir = std::filesystem::path(c.output);
  const double loss = engine.maximum(m.max_loss), dt_limit = stable_dt(c, loss);
  const double divergence = engine.maximum(m.max_divergence);
  int occupied = 0;
  if (d.rank == 0) {
    occupied = std::filesystem::exists(dir) && !std::filesystem::is_empty(dir);
    if (!occupied)
      std::filesystem::create_directories(dir);
  }
  if (engine.maximum(occupied) > 0)
    throw std::runtime_error("Output directory is not empty; choose a new output_dir");
  engine.barrier();
  std::ofstream diag, snapshots;
  if (d.rank == 0) {
    write_config(c, dir / "effective.cfg");
    if (!c.case_dir.empty() &&
        std::filesystem::exists(std::filesystem::path(c.case_dir) / "input.sha256"))
      std::filesystem::copy_file(std::filesystem::path(c.case_dir) / "input.sha256",
                                 dir / "input.sha256");
    diag.open(dir / "diagnostics.csv");
    snapshots.open(dir / "snapshots.csv");
    if (!diag || !snapshots)
      throw std::runtime_error("Cannot write diagnostics");
    diag << "time,active_particles,mass_inside,mass_escaped,mass_residual,mean_x,mean_y,mean_z,var_"
            "x,var_y,var_z,min_concentration,max_concentration\n"
         << std::setprecision(17);
    snapshots << "index,time,format,ranks\n" << std::setprecision(17);
  }
  {
    std::ofstream domain(dir / ("domain_rank" + std::to_string(d.rank) + ".csv"));
    domain << "rank,x0,y0,nx,ny,nz\n"
           << d.rank << ',' << d.x0 << ',' << d.y0 << ',' << d.nx << ',' << d.ny << ',' << d.nz
           << '\n';
  }
  auto output = [&](int index, double time) {
    double start = seconds();
    const auto &f = engine.host();
    auto stats = moments(f, m);
    stats.mass = engine.sum(stats.mass);
    stats.min = engine.minimum(stats.min);
    stats.max = engine.maximum(stats.max);
    for (int a = 0; a < 3; ++a) {
      stats.mean[a] = engine.sum(stats.mean[a]);
      stats.variance[a] = engine.sum(stats.variance[a]);
    }
    double escaped = engine.sum(engine.escaped);
    if (stats.min < -1e-12 || !std::isfinite(stats.mass))
      throw std::runtime_error("Invalid concentration diagnostic");
    if (d.rank == 0) {
      diag << time << ",nan," << stats.mass << ',' << escaped << ','
           << c.mass - stats.mass - escaped;
      for (int a = 0; a < 3; ++a)
        diag << ','
             << (stats.mass > 0 ? stats.mean[a] / stats.mass
                                : std::numeric_limits<double>::quiet_NaN());
      for (int a = 0; a < 3; ++a)
        diag << ','
             << (stats.mass > 0 ? std::max(0.0, stats.variance[a] / stats.mass -
                                                    std::pow(stats.mean[a] / stats.mass, 2))
                                : std::numeric_limits<double>::quiet_NaN());
      diag << ',' << stats.min << ',' << stats.max << '\n';
      snapshots << index << ',' << time << ',' << c.snapshots << ',' << d.ranks << '\n';
      std::cout << "t=" << time << " mass=" << std::setprecision(12) << stats.mass
                << " escaped=" << escaped << " residual=" << c.mass - stats.mass - escaped << '\n';
    }
    if (c.snapshots != "none") {
      auto name = snapshot_name(index);
      if (d.ranks > 1)
        name += "_rank" + std::to_string(d.rank);
      write_field(dir / (name + (c.snapshots == "csv" ? ".csv" : ".bin")), f, m,
                  c.snapshots == "csv");
    }
    engine.timing.output += seconds() - start;
  };
  engine.barrier();
  double start = seconds(), time = 0, solve = 0;
  std::uint64_t steps = 0;
  int index = 0;
  output(index++, 0);
  while (time < c.end) {
    const double target = std::min(c.end, index * c.output_interval);
    while (time < target) {
      double dt = std::min(dt_limit, target - time);
      if (!(dt > 0) || !std::isfinite(dt) || time + dt == time)
        throw std::runtime_error("Timestep cannot advance time");
      const double step_start = seconds();
      engine.advance(dt);
      solve += seconds() - step_start;
      time += dt;
      ++steps;
      if (target - time <= 8 * std::numeric_limits<double>::epsilon() * std::max(1.0, target))
        time = target;
    }
    output(index++, time);
  }
  engine.barrier();
  const double elapsed = engine.maximum(seconds() - start);
  solve = engine.maximum(solve);
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  const double rss = engine.maximum(usage.ru_maxrss / 1048576.0);
#else
  const double rss = engine.maximum(usage.ru_maxrss / 1024.0);
#endif
  const double update = engine.maximum(engine.timing.update),
               halo = engine.maximum(engine.timing.halo),
               wait = engine.maximum(engine.timing.exposed_wait),
               io = engine.maximum(engine.timing.output);
  if (d.rank == 0) {
    std::ofstream f(dir / "timing.csv");
    f << std::setprecision(17);
    f << "backend,ranks,threads,nx,ny,nz,steps,elapsed_seconds,solve_seconds,update_seconds,halo_"
         "seconds,exposed_wait_seconds,output_seconds,max_rank_host_rss_mib,max_loss,max_discrete_"
         "divergence,decomposition,exchange\n";
    f << engine.backend() << ',' << d.ranks << ',' << engine.concurrency() << ',' << c.nx << ','
      << c.ny << ',' << c.nz << ',' << steps << ',' << elapsed << ',' << solve << ',' << update
      << ',' << halo << ',' << wait << ',' << io << ',' << rss << ',' << loss << ',' << divergence
      << ',' << c.decomposition << ',' << c.exchange << '\n';
  }
}
} // namespace dispersion
