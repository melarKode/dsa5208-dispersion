#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "lps/config.hpp"
#include "lps/grid.hpp"
#include "lps/types.hpp"

namespace lps {

enum class StepOutcome {
  Active,   // particle is inside the domain (wrapped onto periodic axes)
  Escaped,  // particle left through an open boundary; its mass becomes mass_escaped
};

class Domain {
 public:
  Domain(const Grid& grid, const std::array<BoundaryType, 3>& boundary, std::vector<unsigned char> mask = {});

  const Grid& grid() const { return grid_; }
  BoundaryType boundary(int axis) const { return boundary_[axis]; }
  double length(int axis) const;

  // Maps a position onto the domain: wraps periodic axes (updating `image`)
  // and reports Escaped if an open axis is left.
  StepOutcome apply_boundaries(Vec3& position, std::int32_t image[3]) const;

  StepOutcome resolve_step(const Vec3& start, Vec3& end, std::int32_t image[3], bool reflect_open = false) const;
  bool is_solid_cell(std::int64_t id) const { return !mask_.empty() && mask_.at(id) != 0; }
  bool has_geometry() const { return !mask_.empty(); }

 private:
  std::vector<unsigned char> mask_;
  Grid grid_;
  std::array<BoundaryType, 3> boundary_;
};

}  // namespace lps
