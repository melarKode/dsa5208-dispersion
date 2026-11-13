#include "dispersion/core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace dispersion;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    Config c;
    c.nx = 13;
    c.ny = 9;
    c.nz = 5;
    c.sx = 3;
    c.sy = 4;
    c.sz = 2;
    c.dt = 0;
    Domain d{c.nx, c.ny, c.nz, 0, 0, c.nx, c.ny};
    Model m(c, d);
    normalize(m, m.initial_sum);
    auto f = m.initial;
    std::vector<double> next(d.padded());
    double dt = stable_dt(c, m.max_loss);
    double original = moments(f, m).mass;
    for (int t = 0; t < 20; ++t) {
      serial_periodic(d, c, f);
      serial_step(m, f, next, dt);
      f.swap(next);
    }
    auto stats = moments(f, m);
    require(std::abs(stats.mass - original) < 1e-12, "Periodic mass conservation");
    require(stats.min >= 0, "Positivity");
    c.u = 0;
    c.diffusion = 0;
    Model stationary(c, d);
    normalize(stationary, stationary.initial_sum);
    f = stationary.initial;
    serial_periodic(d, c, f);
    serial_step(stationary, f, next, 1);
    require(moments(next, stationary).mass == moments(f, stationary).mass, "Stationary field");
    c.diffusion = .1;
    c.u = 1;
    c.boundary = {"open", "closed", "closed"};
    Model open(c, d);
    normalize(open, open.initial_sum);
    f = open.initial;
    double escaped = 0;
    dt = stable_dt(c, open.max_loss);
    for (int t = 0; t < 100; ++t) {
      double flux = 0;
      for (int k = 1; k <= d.nz; ++k)
        for (int j = 1; j <= d.ny; ++j)
          for (int i = 1; i <= d.nx; ++i) {
            auto q = (static_cast<std::size_t>(k - 1) * d.ny + j - 1) * d.nx + i - 1;
            flux += open.escape[q] * f[d.index(i, j, k)];
          }
      escaped += dt * flux * c.dx * c.dy * c.dz;
      serial_step(open, f, next, dt);
      f.swap(next);
    }
    require(escaped > .01, "Open boundary should lose mass");
    require(std::abs(moments(f, open).mass + escaped - 1) < 1e-12, "Open mass budget");
    c.wall_i = 6;
    c.boundary = {"closed", "closed", "closed"};
    Model wall(c, d);
    f.assign(d.padded(), 0);
    f[d.index(3, 4, 2)] = 1;
    dt = stable_dt(c, wall.max_loss);
    for (int t = 0; t < 50; ++t) {
      serial_step(wall, f, next, dt);
      f.swap(next);
    }
    double right = 0;
    for (int k = 1; k <= d.nz; ++k)
      for (int j = 1; j <= d.ny; ++j)
        for (int i = 8; i <= d.nx; ++i)
          right += f[d.index(i, j, k)];
    require(right == 0, "Pollutant crossed impermeable wall");
    require(std::abs(moments(f, wall).mass - 1) < 1e-12, "Wall mass conservation");
    bool rejected = false;
    c.dt = 100;
    try {
      stable_dt(c, wall.max_loss);
    } catch (const std::exception &) {
      rejected = true;
    }
    require(rejected, "Unsafe dt accepted");
    std::cout << "PASS: periodic mass, positivity, stationary, open budget, wall, CFL\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
