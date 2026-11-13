#pragma once

#include <cmath>
#include <cstdint>

#include "lps/types.hpp"

// Counter-based random numbers.
//
// Every random draw is a pure function of
//   (seed, particle_id, step, stream)  ->  three standard normals (x, y, z).
// There is no generator state stored per particle or per rank, so a particle
// that migrates between MPI ranks, or is processed by a different thread or in
// a different order, receives exactly the same random numbers. Nothing can be
// restarted or duplicated by migration.
//
// Generator: Philox4x32-10 (Salmon, Moraes, Dror & Shaw, SC'11), the same
// algorithm as Random123 and Kokkos/cuRAND Philox. One call maps a 128-bit
// counter and a 64-bit key to 128 random bits:
//   key     = { seed[31:0], seed[63:32] }
//   counter = { particle_id[31:0], particle_id[63:32], step, stream }
// The four 32-bit output words become four uniforms on (0,1), and Box-Muller
// turns them into normals:
//   (u0,u1) -> (xi_x, xi_y),  (u2,u3) -> (xi_z, unused)
// So the "component" is the position of the word pair inside the block.

namespace lps {

// Separate streams keep draws for different purposes independent even when
// they share a step number. Append new purposes; never renumber.
enum RngStream : std::uint32_t {
  kStreamInitialPosition = 0,  // sampling the release, step = 0
  kStreamDiffusion = 1,        // random walk, step = timestep index n (step n -> n+1)
};

struct Philox4x32Block {
  std::uint32_t v[4];
};

LPS_INLINE void philox_mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
  const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
  hi = static_cast<std::uint32_t>(product >> 32);
  lo = static_cast<std::uint32_t>(product);
}

LPS_INLINE Philox4x32Block philox4x32_10(Philox4x32Block ctr, std::uint32_t k0, std::uint32_t k1) {
  constexpr std::uint32_t kM0 = 0xD2511F53u;
  constexpr std::uint32_t kM1 = 0xCD9E8D57u;
  constexpr std::uint32_t kW0 = 0x9E3779B9u;
  constexpr std::uint32_t kW1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k0 += kW0;
      k1 += kW1;
    }
    std::uint32_t hi0, lo0, hi1, lo1;
    philox_mulhilo(kM0, ctr.v[0], hi0, lo0);
    philox_mulhilo(kM1, ctr.v[2], hi1, lo1);
    ctr = Philox4x32Block{{hi1 ^ ctr.v[1] ^ k0, lo1, hi0 ^ ctr.v[3] ^ k1, lo0}};
  }
  return ctr;
}

// Uniform on the open interval (0,1); never returns exactly 0 or 1, so
// log() in Box-Muller is always finite (largest |normal| is about 6.7).
LPS_INLINE double uniform_open01(std::uint32_t word) {
  return (static_cast<double>(word) + 0.5) * 2.3283064365386962890625e-10;  // 2^-32
}

LPS_INLINE Vec3 standard_normal3(std::uint64_t seed, std::uint64_t particle_id, std::uint32_t step,
                                 std::uint32_t stream) {
  const Philox4x32Block counter{{static_cast<std::uint32_t>(particle_id),
                                 static_cast<std::uint32_t>(particle_id >> 32), step, stream}};
  const Philox4x32Block bits =
      philox4x32_10(counter, static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32));

  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double radius_a = std::sqrt(-2.0 * std::log(uniform_open01(bits.v[0])));
  const double angle_a = kTwoPi * uniform_open01(bits.v[1]);
  const double radius_b = std::sqrt(-2.0 * std::log(uniform_open01(bits.v[2])));
  const double angle_b = kTwoPi * uniform_open01(bits.v[3]);
  return {radius_a * std::cos(angle_a), radius_a * std::sin(angle_a), radius_b * std::cos(angle_b)};
}

}  // namespace lps
