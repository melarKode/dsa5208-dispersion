#pragma once
#include "dispersion/core.hpp"
#include <chrono>
namespace dispersion {
struct Timing {
  double update = 0, halo = 0, exposed_wait = 0, output = 0;
};
class Engine {
public:
  virtual ~Engine() = default;
  virtual void advance(double) = 0;
  virtual const std::vector<double> &host() = 0;
  virtual double sum(double) = 0;
  virtual double maximum(double) = 0;
  virtual double minimum(double) = 0;
  virtual void barrier() = 0;
  virtual std::string backend() const = 0;
  virtual int concurrency() const = 0;
  double escaped = 0;
  Timing timing;
};
inline double seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
Options command_overrides(int argc, char **argv);
void run(Model &, Engine &);
} // namespace dispersion
