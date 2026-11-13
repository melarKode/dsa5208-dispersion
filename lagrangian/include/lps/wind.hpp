#pragma once

#include <memory>
#include <string>

#include "lps/config.hpp"
#include "lps/types.hpp"

namespace lps {

// Wind evaluation at a particle position. Knows nothing about particles,
// boundaries or file formats: a gridded urban wind will be a second
// implementation that reads the normalised wind/geometry format (still to be
// agreed with the grid solver) and interpolates with the shared rules.
class WindField {
 public:
  virtual ~WindField() = default;
  // Velocity [m/s] at position [m] (inside the domain) and time [s].
  virtual Vec3 velocity(const Vec3& position, double time) const = 0;
  virtual std::string description() const = 0;
};

class UniformWind final : public WindField {
 public:
  explicit UniformWind(const Vec3& velocity) : velocity_(velocity) {}
  Vec3 velocity(const Vec3&, double) const override { return velocity_; }
  std::string description() const override;

 private:
  Vec3 velocity_;
};

std::unique_ptr<WindField> make_wind_field(const Config& config);

}  // namespace lps
