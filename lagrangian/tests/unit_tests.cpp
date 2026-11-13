// Unit tests for the building blocks. The physics validation cases live in
// scripts/validate.py because they need multiple seeds and statistics.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "lps/config.hpp"
#include "lps/diagnostics.hpp"
#include "lps/domain.hpp"
#include "lps/grid.hpp"
#include "lps/particles.hpp"
#include "lps/rng.hpp"
#include "lps/stepper.hpp"
#include "lps/wind.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(condition)) {                                                     \
      ++g_failures;                                                         \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);   \
    }                                                                       \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

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

Config small_config() {
  KeyValueList kv = {{"nx", "8"},          {"ny", "6"},           {"nz", "4"},         {"dx", "1"},
                     {"dy", "2"},          {"dz", "0.5"},         {"u", "0.3"},        {"v", "-0.2"},
                     {"w", "0.1"},         {"diffusivity", "0.1"}, {"source_x", "4"},   {"source_y", "6"},
                     {"source_z", "1"},    {"source_sigma", "0.5"}, {"particle_count", "1000"},
                     {"seed", "7"},        {"dt", "0.1"},         {"end_time", "1"},   {"output_interval", "0.5"}};
  return parse_config(kv);
}

void test_philox_known_answers() {
  std::printf("philox known-answer vectors (Random123 kat_vectors)\n");
  const Philox4x32Block zero = philox4x32_10({{0, 0, 0, 0}}, 0, 0);
  CHECK(zero.v[0] == 0x6627e8d5u && zero.v[1] == 0xe169c58du && zero.v[2] == 0xbc57ac4cu &&
        zero.v[3] == 0x9b00dbd8u);
  const Philox4x32Block ones =
      philox4x32_10({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, 0xffffffffu, 0xffffffffu);
  CHECK(ones.v[0] == 0x408f276du && ones.v[1] == 0x41c83b0eu && ones.v[2] == 0xa20bc7c6u &&
        ones.v[3] == 0x6d5451fdu);
  const Philox4x32Block pi =
      philox4x32_10({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, 0xa4093822u, 0x299f31d0u);
  CHECK(pi.v[0] == 0xd16cfe09u && pi.v[1] == 0x94fdccebu && pi.v[2] == 0x5001e420u && pi.v[3] == 0x24126ea1u);
}

void test_normal_draws() {
  std::printf("standard_normal3: determinism, independence of inputs, moments\n");
  const Vec3 a = standard_normal3(12345, 42, 7, kStreamDiffusion);
  const Vec3 b = standard_normal3(12345, 42, 7, kStreamDiffusion);
  CHECK(a.x == b.x && a.y == b.y && a.z == b.z);
  const Vec3 other_step = standard_normal3(12345, 42, 8, kStreamDiffusion);
  const Vec3 other_id = standard_normal3(12345, 43, 7, kStreamDiffusion);
  const Vec3 other_seed = standard_normal3(12346, 42, 7, kStreamDiffusion);
  const Vec3 other_stream = standard_normal3(12345, 42, 7, kStreamInitialPosition);
  CHECK(a.x != other_step.x && a.x != other_id.x && a.x != other_seed.x && a.x != other_stream.x);
  // High id bits must matter (MPI runs may use large ids).
  CHECK(standard_normal3(1, 5, 0, 0).x != standard_normal3(1, 5 + (std::uint64_t{1} << 32), 0, 0).x);

  const int n = 1000000;
  double s[3] = {0, 0, 0}, s2[3] = {0, 0, 0}, s4[3] = {0, 0, 0}, cxy = 0, cyz = 0, cxz = 0;
  for (int i = 0; i < n; ++i) {
    const Vec3 z = standard_normal3(99, static_cast<std::uint64_t>(i), 3, kStreamDiffusion);
    const double c[3] = {z.x, z.y, z.z};
    for (int d = 0; d < 3; ++d) {
      s[d] += c[d];
      s2[d] += c[d] * c[d];
      s4[d] += c[d] * c[d] * c[d] * c[d];
    }
    cxy += z.x * z.y;
    cyz += z.y * z.z;
    cxz += z.x * z.z;
  }
  // 5-sigma bands: SE(mean)=1/sqrt(n), SE(var)=sqrt(2/n), SE(4th moment)=sqrt(96/n).
  const double se = 1.0 / std::sqrt(double(n));
  for (int d = 0; d < 3; ++d) {
    CHECK_NEAR(s[d] / n, 0.0, 5 * se);
    CHECK_NEAR(s2[d] / n, 1.0, 5 * std::sqrt(2.0) * se);
    CHECK_NEAR(s4[d] / n, 3.0, 5 * std::sqrt(96.0) * se);
  }
  CHECK_NEAR(cxy / n, 0.0, 5 * se);
  CHECK_NEAR(cyz / n, 0.0, 5 * se);
  CHECK_NEAR(cxz / n, 0.0, 5 * se);
}

void test_grid() {
  std::printf("grid: cell_id, cell centres, locate\n");
  Grid g;
  g.nx = 4;
  g.ny = 3;
  g.nz = 2;
  g.dx = 1.0;
  g.dy = 2.0;
  g.dz = 0.5;
  g.origin_x = -2.0;
  g.origin_y = 10.0;
  g.origin_z = 0.0;
  CHECK(g.num_cells() == 24);
  CHECK(g.cell_id(0, 0, 0) == 0);
  CHECK(g.cell_id(3, 0, 0) == 3);
  CHECK(g.cell_id(0, 1, 0) == 4);
  CHECK(g.cell_id(0, 0, 1) == 12);
  CHECK(g.cell_id(3, 2, 1) == 23);
  const Vec3 c = g.cell_centre(1, 2, 1);
  CHECK_NEAR(c.x, -0.5, 1e-15);
  CHECK_NEAR(c.y, 15.0, 1e-15);
  CHECK_NEAR(c.z, 0.75, 1e-15);
  int i, j, k;
  CHECK(g.locate(c, i, j, k) && i == 1 && j == 2 && k == 1);
  CHECK(g.locate({-2.0, 10.0, 0.0}, i, j, k) && i == 0 && j == 0 && k == 0);  // lower faces belong to the grid
  CHECK(!g.locate({2.0, 11.0, 0.1}, i, j, k));                               // upper face is outside
  CHECK(!g.locate({-2.0000001, 11.0, 0.1}, i, j, k));
  CHECK(g.locate({std::nextafter(2.0, 0.0), 11.0, 0.1}, i, j, k) && i == 3);
}

void test_boundaries() {
  std::printf("domain: periodic wrap with image counters, open escape\n");
  Grid g;
  g.nx = g.ny = g.nz = 10;
  g.dx = g.dy = g.dz = 1.0;
  g.origin_x = 5.0;
  const Domain periodic(g, {BoundaryType::Periodic, BoundaryType::Periodic, BoundaryType::Periodic});
  const double cases[][2] = {{5.0, 0}, {14.5, 0}, {15.0, 1}, {16.25, 1}, {4.5, -1}, {-25.5, -4}, {36.0, 3}};
  for (const auto& c : cases) {
    Vec3 p{c[0], 5.0, 5.0};
    std::int32_t image[3] = {0, 0, 0};
    CHECK(periodic.apply_boundaries(p, image) == StepOutcome::Active);
    CHECK(p.x >= 5.0 && p.x < 15.0);
    CHECK(image[0] == static_cast<int>(c[1]));
    CHECK_NEAR(p.x + image[0] * 10.0, c[0], 1e-12);
  }
  // A tiny negative offset must not produce x == upper face.
  Vec3 p{5.0 - 1e-17, 0.0, 0.0};
  std::int32_t image[3] = {0, 0, 0};
  periodic.apply_boundaries(p, image);
  CHECK(p.x >= 5.0 && p.x < 15.0);
  int i, j, k;
  CHECK(g.locate(p, i, j, k));

  const Domain open(g, {BoundaryType::Open, BoundaryType::Periodic, BoundaryType::Periodic});
  Vec3 inside{14.9, 3.0, 3.0}, outside_hi{15.0, 3.0, 3.0}, outside_lo{4.99, 3.0, 3.0};
  std::int32_t im[3] = {0, 0, 0};
  CHECK(open.apply_boundaries(inside, im) == StepOutcome::Active);
  CHECK(open.apply_boundaries(outside_hi, im) == StepOutcome::Escaped);
  CHECK(open.apply_boundaries(outside_lo, im) == StepOutcome::Escaped);
}

void test_config_parsing() {
  std::printf("config: comments, whitespace, overrides, errors, effective config round trip\n");
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "lps_unit_test_config";
  fs::create_directories(dir);
  const std::string path = (dir / "in.cfg").string();
  {
    std::ofstream out(path);
    out << "# comment\n\n   # indented comment\ncase_name = demo\nnx=8\nny=6\nnz=4\ndx=1\ndy=2\ndz=0.5\n"
        << "u=0.1\nv=0\nw=0\ndiffusivity=0.1\nsource_x=4\nsource_y=6\nsource_z=1\nsource_sigma=0.5\n"
        << "particle_count=100\nseed=18446744073709551615\ndt=0.05\nend_time=2.0\noutput_interval=0.5\n"
        << "boundary_z = open\nfv_scheme=upwind\n";
  }
  KeyValueList kv = read_key_value_file(path);
  set_key_value(kv, "seed=5");
  set_key_value(kv, "write_particles=true");
  const Config c = parse_config(kv);
  CHECK(c.case_name == "demo");
  CHECK(c.nx == 8 && c.dy == 2.0 && c.dz == 0.5);
  CHECK(c.seed == 5);
  CHECK(c.write_particles && c.write_concentration);
  CHECK(c.boundary[0] == BoundaryType::Periodic && c.boundary[2] == BoundaryType::Open);
  CHECK(c.output_dir == "output/demo");
  CHECK(c.unused_keys.size() == 1 && c.unused_keys[0].first == "fv_scheme");
  const TimeControl tc = make_time_control(c);
  CHECK(tc.num_steps == 40 && tc.output_stride == 10);

  // The effective config must reproduce the same configuration.
  const std::string effective = (dir / "effective.cfg").string();
  write_effective_config(c, effective);
  const Config again = parse_config(read_key_value_file(effective));
  CHECK(again.seed == c.seed && again.dt == c.dt && again.dz == c.dz && again.boundary == c.boundary &&
        again.particle_count == c.particle_count && again.unused_keys.size() == 1);

  {
    std::ofstream out(path);
    out << "nx=8\nnx=9\n";
  }
  CHECK(throws([&] { read_key_value_file(path); }));  // duplicate key
  {
    std::ofstream out(path);
    out << "nx 8\n";
  }
  CHECK(throws([&] { read_key_value_file(path); }));  // missing '='

  KeyValueList bad = kv;
  set_key_value(bad, "dx=abc");
  CHECK(throws([&] { parse_config(bad); }));
  bad = kv;
  set_key_value(bad, "seed=-1");
  CHECK(throws([&] { parse_config(bad); }));
  bad = kv;
  bad.erase(std::remove_if(bad.begin(), bad.end(), [](const auto& e) { return e.first == "dt"; }), bad.end());
  CHECK(throws([&] { parse_config(bad); }));
  bad = kv;
  set_key_value(bad, "output_interval=0.12");  // not a multiple of dt
  CHECK(throws([&] { make_time_control(parse_config(bad)); }));

  CHECK(format_double(0.1) == "0.1");
  CHECK(std::strtod(format_double(1.0 / 3.0).c_str(), nullptr) == 1.0 / 3.0);
  CHECK(format_double(std::nan("")) == "nan");
  fs::remove_all(dir);
}

void test_diagnostics_known_values() {
  std::printf("diagnostics: weighted moments, unwrapped positions, NaN when empty\n");
  Grid g;
  g.nx = g.ny = g.nz = 10;
  g.dx = g.dy = g.dz = 1.0;
  ParticleSet ps;
  ps.resize(3);
  const double xs[3] = {1.0, 9.5, 3.0};
  const double ms[3] = {0.25, 0.75, 0.5};
  for (int p = 0; p < 3; ++p) {
    ps.id[p] = p;
    ps.x[p] = xs[p];
    ps.y[p] = 2.0;
    ps.z[p] = 3.0;
    ps.mass[p] = ms[p];
    ps.active[p] = p < 2;  // third particle inactive
  }
  ps.image_x[1] = -1;  // unwrapped x = -0.5
  const DiagnosticsRow r = compute_diagnostics(ps, g, 1.5, 1.5, 0.5);
  // Active: x = 1.0 (m=0.25) and -0.5 (m=0.75): mean = -0.125, var = 0.421875
  CHECK(r.active_particles == 2);
  CHECK_NEAR(r.mass_inside, 1.0, 1e-15);
  CHECK_NEAR(r.mass_residual, 0.0, 1e-15);
  CHECK_NEAR(r.mean.x, -0.125, 1e-14);
  CHECK_NEAR(r.variance.x, 0.421875, 1e-14);
  CHECK_NEAR(r.mean.y, 2.0, 1e-14);
  CHECK_NEAR(r.variance.y, 0.0, 1e-14);

  std::fill(ps.active.begin(), ps.active.end(), 0);
  const DiagnosticsRow empty = compute_diagnostics(ps, g, 2.0, 1.5, 1.5);
  CHECK(empty.active_particles == 0 && std::isnan(empty.mean.x) && std::isnan(empty.variance.z));
}

void test_deposit_conserves_mass() {
  std::printf("deposit: NGP mass conservation and placement\n");
  Config c = small_config();
  const Domain domain(c.grid(), c.boundary);
  const ReleaseResult release = make_gaussian_release(c, domain);
  const std::vector<double> cell_mass = deposit_mass_ngp(release.particles, c.grid());
  CompensatedSum total;
  for (double m : cell_mass) total.add(m);
  CHECK_NEAR(total.result(), c.initial_mass, 1e-14);
  CHECK(cell_mass.size() == static_cast<std::size_t>(c.grid().num_cells()));
}

// The property that makes MPI migration safe: a particle's trajectory depends
// on its id, not on its position in the arrays or the order of processing.
void test_order_independence() {
  std::printf("stepper: trajectories independent of particle storage order\n");
  Config c = small_config();
  const Domain domain(c.grid(), c.boundary);
  const auto wind = make_wind_field(c);
  ParticleSet a = make_gaussian_release(c, domain).particles;

  ParticleSet b;
  b.resize(a.size());
  std::vector<std::size_t> perm(a.size());
  for (std::size_t i = 0; i < perm.size(); ++i) perm[i] = i;
  std::mt19937 shuffle_rng(3);
  std::shuffle(perm.begin(), perm.end(), shuffle_rng);
  for (std::size_t i = 0; i < perm.size(); ++i) {
    const std::size_t s = perm[i];
    b.id[i] = a.id[s];
    b.x[i] = a.x[s];
    b.y[i] = a.y[s];
    b.z[i] = a.z[s];
    b.image_x[i] = a.image_x[s];
    b.image_y[i] = a.image_y[s];
    b.image_z[i] = a.image_z[s];
    b.mass[i] = a.mass[s];
    b.active[i] = a.active[s];
  }
  const StepParameters params{c.seed, c.dt, c.diffusivity};
  for (std::uint32_t n = 0; n < 10; ++n) {
    advance_particles(a, *wind, domain, params, n);
    advance_particles(b, *wind, domain, params, n);
  }
  std::map<std::uint64_t, std::size_t> slot_of_id;
  for (std::size_t i = 0; i < b.size(); ++i) slot_of_id[b.id[i]] = i;
  bool identical = slot_of_id.size() == a.size();
  for (std::size_t p = 0; p < a.size() && identical; ++p) {
    const std::size_t q = slot_of_id.at(a.id[p]);
    identical = a.x[p] == b.x[q] && a.y[p] == b.y[q] && a.z[p] == b.z[q] && a.image_x[p] == b.image_x[q];
  }
  CHECK(identical);
}

}  // namespace

int main() {
  test_philox_known_answers();
  test_normal_draws();
  test_grid();
  test_boundaries();
  test_config_parsing();
  test_diagnostics_known_values();
  test_deposit_conserves_mass();
  test_order_independence();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
