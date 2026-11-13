#pragma once

#include <cmath>
#include <cstdint>

#include "lps/types.hpp"

namespace lps {

// The shared observation grid. Must agree exactly with the finite-volume solver:
//   origin = lower corner of the domain
//   cell (i,j,k) centre = origin + ((i+1/2)dx, (j+1/2)dy, (k+1/2)dz), zero-based
//   cell_id = i + nx * (j + ny * k)
struct Grid {
  int nx = 0, ny = 0, nz = 0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;

  LPS_INLINE std::int64_t num_cells() const {
    return static_cast<std::int64_t>(nx) * static_cast<std::int64_t>(ny) * static_cast<std::int64_t>(nz);
  }
  LPS_INLINE double cell_volume() const { return dx * dy * dz; }
  LPS_INLINE double length_x() const { return nx * dx; }
  LPS_INLINE double length_y() const { return ny * dy; }
  LPS_INLINE double length_z() const { return nz * dz; }

  LPS_INLINE std::int64_t cell_id(int i, int j, int k) const {
    return static_cast<std::int64_t>(i) +
           static_cast<std::int64_t>(nx) * (static_cast<std::int64_t>(j) + static_cast<std::int64_t>(ny) * k);
  }

  LPS_INLINE Vec3 cell_centre(int i, int j, int k) const {
    return {origin_x + (i + 0.5) * dx, origin_y + (j + 0.5) * dy, origin_z + (k + 0.5) * dz};
  }

  // Index of the cell containing coordinate `x` along an axis with n cells of
  // width d starting at `origin`. Returns -1 outside [origin, origin + n*d).
  LPS_INLINE static int axis_index(double x, double origin, double d, int n) {
    const double s = (x - origin) / d;
    if (!(s >= 0.0) || !(x < origin + n * d)) return -1;
    int i = static_cast<int>(s);
    if (i >= n) i = n - 1;  // x just below the upper face, rounded up by the division
    return i;
  }

  // Containing cell of a position; false if the position is outside the grid.
  LPS_INLINE bool locate(const Vec3& p, int& i, int& j, int& k) const {
    i = axis_index(p.x, origin_x, dx, nx);
    j = axis_index(p.y, origin_y, dy, ny);
    k = axis_index(p.z, origin_z, dz, nz);
    return i >= 0 && j >= 0 && k >= 0;
  }
};

}  // namespace lps
