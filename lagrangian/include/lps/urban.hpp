#pragma once
#include <vector>
#include "lps/wind.hpp"

namespace lps {
std::vector<unsigned char> load_solid(const Config& config);

class UrbanWind final : public WindField {
 public:
  explicit UrbanWind(const Config& config);
  Vec3 velocity(const Vec3& position, double time) const override;
  std::string description() const override { return "file wind, face-normal linear reconstruction"; }
 private:
  Grid grid_;
  std::array<BoundaryType,3> boundary_;
  std::vector<unsigned char> mask_;
  std::array<std::vector<double>,3> wind_;
};
}
