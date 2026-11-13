#include "lps/decomposition.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lps {

Decomposition::Decomposition(const Grid& grid, const std::array<BoundaryType, 3>& boundary,
                             const std::array<int, 3>& d) {
  const int n[3] = {grid.nx, grid.ny, grid.nz};
  for (int a = 0; a < 3; ++a) {
    if (d[a] < 1 || d[a] > n[a])
      throw std::runtime_error("rank grid " + std::to_string(d[0]) + "x" + std::to_string(d[1]) + "x" +
                               std::to_string(d[2]) + " does not fit " + std::to_string(n[0]) + "x" +
                               std::to_string(n[1]) + "x" + std::to_string(n[2]) + " cells");
    dims[a] = d[a];
    cells[a] = n[a];
    periodic[a] = boundary[a] == BoundaryType::Periodic;
  }
}

bool Decomposition::adjacent(int a, int b) const {
  int ca[3], cb[3];
  coords(a, ca);
  coords(b, cb);
  for (int axis = 0; axis < 3; ++axis) {
    int distance = std::abs(ca[axis] - cb[axis]);
    if (periodic[axis]) distance = std::min(distance, dims[axis] - distance);
    if (distance > 1) return false;
  }
  return true;
}

Decomposition::Box Decomposition::box(int rank) const {
  int c[3];
  coords(rank, c);
  Box b;
  for (int a = 0; a < 3; ++a) {
    b.lo[a] = begin(a, c[a]);
    b.hi[a] = begin(a, c[a] + 1);
  }
  return b;
}

std::array<int, 3> default_rank_dims(int num_ranks, const Grid& grid) {
  std::array<int, 3> best = {0, 0, 0};
  double best_cost = std::numeric_limits<double>::infinity();
  // Descending px: on ties, prefer more ranks along x (the x-fastest axis).
  for (int px = num_ranks; px >= 1; --px) {
    if (num_ranks % px != 0) continue;
    const int py = num_ranks / px;
    if (px > grid.nx || py > grid.ny) continue;
    // Perimeter of a subdomain in cells: proportional to the particle flux
    // through rank boundaries for a spread-out plume.
    const double cost = static_cast<double>(grid.nx) / px + static_cast<double>(grid.ny) / py;
    if (cost < best_cost - 1e-12) {
      best_cost = cost;
      best = {px, py, 1};
    }
  }
  if (best[0] == 0)
    throw std::runtime_error("cannot split " + std::to_string(grid.nx) + "x" + std::to_string(grid.ny) +
                             " columns over " + std::to_string(num_ranks) + " ranks; pass --mpi-dims");
  return best;
}

std::array<int, 3> parse_rank_dims(const std::string& text) {
  std::string normalised = text;
  for (char& ch : normalised)
    if (ch == 'x' || ch == 'X') ch = ',';
  std::stringstream in(normalised);
  std::array<int, 3> dims = {0, 0, 0};
  std::string part;
  int a = 0;
  while (std::getline(in, part, ',')) {
    if (a == 3) throw std::runtime_error("--mpi-dims needs three values, got '" + text + "'");
    char* end = nullptr;
    const long value = std::strtol(part.c_str(), &end, 10);
    if (part.empty() || *end != '\0' || value < 1) throw std::runtime_error("bad --mpi-dims '" + text + "'");
    dims[a++] = static_cast<int>(value);
  }
  if (a != 3) throw std::runtime_error("--mpi-dims needs three values, got '" + text + "'");
  return dims;
}

}  // namespace lps
