#include "lps/wind.hpp"
#include "lps/urban.hpp"

#include <stdexcept>

namespace lps {

std::string UniformWind::description() const {
  return "uniform (" + format_double(velocity_.x) + ", " + format_double(velocity_.y) + ", " +
         format_double(velocity_.z) + ") m/s";
}

std::unique_ptr<WindField> make_wind_field(const Config& config) {
  switch (config.velocity_mode) {
    case VelocityMode::File: return std::make_unique<UrbanWind>(config);
    case VelocityMode::Uniform: return std::make_unique<UniformWind>(Vec3{config.u, config.v, config.w});
  }
  throw std::runtime_error("unsupported velocity_mode");
}

}  // namespace lps
