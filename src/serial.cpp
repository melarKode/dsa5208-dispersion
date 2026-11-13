#include "dispersion/run.hpp"
#include <iostream>
#include <stdexcept>
namespace dispersion {
class SerialEngine final : public Engine {
  Model &m;
  std::vector<double> current, next;

public:
  explicit SerialEngine(Model &model)
      : m(model), current(model.initial), next(model.domain.padded()) {}
  void advance(double dt) override {
    double start = seconds();
    serial_periodic(m.domain, m.config, current);
    const auto &d = m.domain;
    double flux = 0;
    for (int k = 1; k <= d.nz; ++k)
      for (int j = 1; j <= d.ny; ++j)
        for (int i = 1; i <= d.nx; ++i) {
          auto q = (static_cast<std::size_t>(k - 1) * d.ny + j - 1) * d.nx + i - 1;
          flux += m.escape[q] * current[d.index(i, j, k)];
        }
    escaped += dt * flux * m.config.dx * m.config.dy * m.config.dz;
    serial_step(m, current, next, dt);
    current.swap(next);
    timing.update += seconds() - start;
  }
  const std::vector<double> &host() override {
    return current;
  }
  double sum(double v) override {
    return v;
  }
  double maximum(double v) override {
    return v;
  }
  double minimum(double v) override {
    return v;
  }
  void barrier() override {}
  std::string backend() const override {
    return "serial";
  }
  int concurrency() const override {
    return 1;
  }
};
} // namespace dispersion
int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::runtime_error("Usage: dispersion_serial CONFIG [key=value ...]");
    auto c = dispersion::read_config(argv[1], dispersion::command_overrides(argc, argv));
    dispersion::Domain d{c.nx, c.ny, c.nz, 0, 0, c.nx, c.ny};
    dispersion::Model model(c, d);
    dispersion::normalize(model, model.initial_sum);
    dispersion::SerialEngine engine(model);
    dispersion::run(model, engine);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
