// Unit tests for the MPI-free rank decomposition used by lagrangian_parallel.

#include <cstdio>
#include <set>
#include <stdexcept>
#include <vector>

#include "lps/decomposition.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                  \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(condition)) {                                                   \
      ++g_failures;                                                       \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    }                                                                     \
  } while (0)

template <class F>
bool throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

using namespace lps;

Grid make_grid(int nx, int ny, int nz) {
  Grid g;
  g.nx = nx;
  g.ny = ny;
  g.nz = nz;
  g.dx = g.dy = g.dz = 1.0;
  return g;
}

const std::array<BoundaryType, 3> kPeriodic = {BoundaryType::Periodic, BoundaryType::Periodic,
                                               BoundaryType::Periodic};

// Every cell has exactly one owner, owners match the boxes, and boxes tile the grid.
void check_partition(const Grid& g, const std::array<int, 3>& dims) {
  const Decomposition d(g, kPeriodic, dims);
  std::vector<int> owner(static_cast<std::size_t>(g.num_cells()), -1);
  std::int64_t covered = 0;
  for (int r = 0; r < d.num_ranks(); ++r) {
    const Decomposition::Box b = d.box(r);
    CHECK(b.num_cells() > 0);
    covered += b.num_cells();
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
      for (int j = b.lo[1]; j < b.hi[1]; ++j)
        for (int i = b.lo[0]; i < b.hi[0]; ++i) {
          const auto id = static_cast<std::size_t>(g.cell_id(i, j, k));
          CHECK(owner[id] == -1);
          owner[id] = r;
          CHECK(d.owner_of_cell(i, j, k) == r);
        }
  }
  CHECK(covered == g.num_cells());
  // Block sizes along an axis differ by at most one cell.
  for (int a = 0; a < 3; ++a) {
    int smallest = 1 << 30, largest = 0;
    for (int c = 0; c < d.dims[a]; ++c) {
      const int width = d.begin(a, c + 1) - d.begin(a, c);
      smallest = std::min(smallest, width);
      largest = std::max(largest, width);
    }
    CHECK(largest - smallest <= 1);
  }
}

void test_partitions() {
  std::printf("partitions tile the grid exactly once\n");
  check_partition(make_grid(32, 32, 32), {1, 1, 1});
  check_partition(make_grid(32, 32, 32), {2, 2, 1});
  check_partition(make_grid(32, 32, 32), {4, 1, 1});
  check_partition(make_grid(10, 7, 3), {3, 7, 1});  // uneven blocks, one cell per rank in y
  check_partition(make_grid(10, 7, 3), {4, 2, 3});
  check_partition(make_grid(100, 100, 5), {4, 4, 1});
  // Urban-size grid: boxes only (walking all 25M cells would just slow the test).
  const Decomposition urban(make_grid(1000, 1000, 25), kPeriodic, {4, 4, 1});
  std::int64_t covered = 0;
  for (int r = 0; r < urban.num_ranks(); ++r) covered += urban.box(r).num_cells();
  CHECK(covered == 1000LL * 1000 * 25);
  CHECK(urban.box(15).hi[0] == 1000 && urban.box(15).hi[1] == 1000 && urban.box(5).lo[0] == 250);
  CHECK(throws([] { Decomposition(make_grid(4, 4, 4), kPeriodic, {5, 1, 1}); }));
  CHECK(throws([] { Decomposition(make_grid(4, 4, 4), kPeriodic, {0, 1, 1}); }));
}

void test_default_dims() {
  std::printf("default rank grid: x-y split, square subdomains\n");
  const Grid cube = make_grid(32, 32, 32);
  CHECK((default_rank_dims(1, cube) == std::array<int, 3>{1, 1, 1}));
  CHECK((default_rank_dims(2, cube) == std::array<int, 3>{2, 1, 1}));
  CHECK((default_rank_dims(4, cube) == std::array<int, 3>{2, 2, 1}));
  CHECK((default_rank_dims(8, cube) == std::array<int, 3>{4, 2, 1}));
  const Grid urban = make_grid(1000, 1000, 25);
  CHECK((default_rank_dims(16, urban) == std::array<int, 3>{4, 4, 1}));
  const Grid strip = make_grid(64, 8, 8);
  CHECK((default_rank_dims(8, strip) == std::array<int, 3>{8, 1, 1}));
  CHECK(throws([] { default_rank_dims(7, make_grid(4, 4, 4)); }));
  CHECK((parse_rank_dims("4,2,1") == std::array<int, 3>{4, 2, 1}));
  CHECK((parse_rank_dims("4x1x1") == std::array<int, 3>{4, 1, 1}));
  CHECK(throws([] { parse_rank_dims("4,2"); }));
  CHECK(throws([] { parse_rank_dims("4,0,1"); }));
}

void test_adjacency() {
  std::printf("rank adjacency with and without periodic wrap\n");
  const Grid g = make_grid(32, 32, 32);
  const Decomposition periodic(g, kPeriodic, {4, 1, 1});
  CHECK(periodic.adjacent(0, 1));
  CHECK(!periodic.adjacent(0, 2));
  CHECK(periodic.adjacent(0, 3));  // wraps around
  const Decomposition open(g, {BoundaryType::Open, BoundaryType::Periodic, BoundaryType::Periodic}, {4, 1, 1});
  CHECK(!open.adjacent(0, 3));
  const Decomposition grid2d(g, kPeriodic, {2, 2, 1});
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) CHECK(grid2d.adjacent(a, b));  // 2x2: everyone touches everyone
}

// Ownership by position agrees with the half-open cell rule at subdomain faces.
void test_face_ownership() {
  std::printf("positions on a subdomain face belong to the upper rank\n");
  const Grid g = make_grid(32, 32, 32);
  const Decomposition d(g, kPeriodic, {4, 1, 1});
  int i, j, k;
  CHECK(g.locate({8.0, 1.0, 1.0}, i, j, k) && d.owner_of_cell(i, j, k) == 1);
  CHECK(g.locate({std::nextafter(8.0, 0.0), 1.0, 1.0}, i, j, k) && d.owner_of_cell(i, j, k) == 0);
  CHECK(g.locate({std::nextafter(32.0, 0.0), 1.0, 1.0}, i, j, k) && d.owner_of_cell(i, j, k) == 3);
}

}  // namespace

int main() {
  test_partitions();
  test_default_dims();
  test_adjacency();
  test_face_ownership();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
