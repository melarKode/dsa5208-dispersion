#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "lps/config.hpp"
#include "lps/grid.hpp"
#include "lps/types.hpp"

namespace lps {

// Block decomposition of the cell grid over a px * py * pz grid of MPI ranks.
// Contains no MPI calls: ownership is a pure function of the cell index, so
// every rank (and the serial unit tests) agree on the owner of any position.
//
//   rank r has coordinates (cx, cy, cz) with r = cx + px * (cy + py * cz)
//   on each axis it owns cells [begin(axis, c), begin(axis, c + 1)),
//   begin(axis, c) = ceil(c * n_axis / p_axis), so block sizes differ by at most one.
//
// A particle belongs to the rank that owns the cell containing it, using the
// same half-open [lo, hi) rule as the concentration deposit. Rank boundaries
// are communication boundaries only; the physics never sees them.
struct Decomposition {
  struct Box {
    int lo[3] = {0, 0, 0};
    int hi[3] = {0, 0, 0};  // exclusive
    LPS_INLINE int extent(int axis) const { return hi[axis] - lo[axis]; }
    LPS_INLINE std::int64_t num_cells() const {
      return static_cast<std::int64_t>(extent(0)) * extent(1) * extent(2);
    }
  };

  int dims[3] = {1, 1, 1};
  int cells[3] = {1, 1, 1};
  bool periodic[3] = {false, false, false};  // only used to classify rank adjacency

  Decomposition() = default;
  // Throws std::runtime_error unless 1 <= dims[a] <= cells[a] on every axis.
  Decomposition(const Grid& grid, const std::array<BoundaryType, 3>& boundary, const std::array<int, 3>& dims);

  int num_ranks() const { return dims[0] * dims[1] * dims[2]; }

  LPS_INLINE int begin(int axis, int c) const {
    return static_cast<int>((static_cast<std::int64_t>(c) * cells[axis] + dims[axis] - 1) / dims[axis]);
  }
  LPS_INLINE int axis_owner(int axis, int i) const {
    return static_cast<int>((static_cast<std::int64_t>(i) * dims[axis]) / cells[axis]);
  }
  LPS_INLINE int rank_of(int cx, int cy, int cz) const { return cx + dims[0] * (cy + dims[1] * cz); }
  LPS_INLINE int owner_of_cell(int i, int j, int k) const {
    return rank_of(axis_owner(0, i), axis_owner(1, j), axis_owner(2, k));
  }
  LPS_INLINE void coords(int rank, int c[3]) const {
    c[0] = rank % dims[0];
    c[1] = (rank / dims[0]) % dims[1];
    c[2] = rank / (dims[0] * dims[1]);
  }
  // True when ranks a and b are the same or touch (face, edge or corner),
  // counting periodic wrap-around. Migrations between non-adjacent ranks are
  // reported as "non-neighbour": the particle crossed at least one whole subdomain.
  bool adjacent(int a, int b) const;

  Box box(int rank) const;
};

// Default rank grid for `num_ranks` ranks: split x and y only (pz = 1, as the
// grid solver's partitioned output requires full columns), choosing the
// factorisation px * py that makes subdomains closest to square in cells
// (ties prefer more ranks along x). Throws if no factorisation fits the grid.
std::array<int, 3> default_rank_dims(int num_ranks, const Grid& grid);

// Parses "PX,PY,PZ" or "PXxPYxPZ".
std::array<int, 3> parse_rank_dims(const std::string& text);

}  // namespace lps
