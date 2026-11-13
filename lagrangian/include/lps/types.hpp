#pragma once

#include <cstdint>

// LPS_INLINE marks small functions that must remain callable from inside
// parallel kernels. The Kokkos build will define it as KOKKOS_INLINE_FUNCTION.
#ifndef LPS_INLINE
#define LPS_INLINE inline
#endif

namespace lps {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

LPS_INLINE Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
LPS_INLINE Vec3 operator*(double s, const Vec3& a) { return {s * a.x, s * a.y, s * a.z}; }

// Neumaier compensated summation. Used for mass totals so that conservation
// checks (mass_residual) are limited by the masses themselves, not by
// accumulated round-off from summing ~1e5..1e8 equal particle masses.
struct CompensatedSum {
  double sum = 0.0;
  double correction = 0.0;

  LPS_INLINE void add(double value) {
    const double t = sum + value;
    if ((sum >= 0 ? sum : -sum) >= (value >= 0 ? value : -value)) {
      correction += (sum - t) + value;
    } else {
      correction += (value - t) + sum;
    }
    sum = t;
  }
  LPS_INLINE double result() const { return sum + correction; }
};

}  // namespace lps
