#include "dispersion/run.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace dispersion {
using View = Kokkos::View<double *>;
using HostView = Kokkos::View<double *, Kokkos::HostSpace>;
using Policy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::size_t>>;
class ParallelEngine final : public Engine {
  Model &model;
  MPI_Comm comm;
  std::array<int, 4> neighbors{};
  std::array<int, 4> lengths{}, offsets{};
  std::array<MPI_Request, 8> requests{};
  View current, next, rates, loss, escape_rate, send, recv;
  HostView host_send, host_recv, host_current;
  std::vector<double> cached;
  bool has_escape = false;
  int nx, ny, nz;
  std::size_t cells;
  void pack() {
    auto f = current, b = send;
    const int x = nx, y = ny, z = nz;
    Kokkos::parallel_for(
        "pack_x", Policy(0, static_cast<std::size_t>(y) * z), KOKKOS_LAMBDA(std::size_t q) {
          int j = q % y + 1, k = q / y + 1;
          std::size_t p = (static_cast<std::size_t>(k) * (y + 2) + j) * (x + 2);
          b(q) = f(p + 1);
          b(y * z + q) = f(p + x);
        });
    Kokkos::parallel_for(
        "pack_y", Policy(0, static_cast<std::size_t>(x) * z), KOKKOS_LAMBDA(std::size_t q) {
          int i = q % x + 1, k = q / x + 1;
          std::size_t p = static_cast<std::size_t>(k) * (y + 2) * (x + 2) + i;
          b(2 * y * z + q) = f(p + x + 2);
          b(2 * y * z + x * z + q) = f(p + y * (x + 2));
        });
    Kokkos::deep_copy(host_send, send);
  }
  void unpack() {
    Kokkos::deep_copy(recv, host_recv);
    auto f = current, b = recv;
    const int x = nx, y = ny, z = nz;
    const bool left = neighbors[0] != MPI_PROC_NULL, right = neighbors[1] != MPI_PROC_NULL,
               bottom = neighbors[2] != MPI_PROC_NULL, top = neighbors[3] != MPI_PROC_NULL;
    Kokkos::parallel_for(
        "unpack_x", Policy(0, static_cast<std::size_t>(y) * z), KOKKOS_LAMBDA(std::size_t q) {
          int j = q % y + 1, k = q / y + 1;
          std::size_t p = (static_cast<std::size_t>(k) * (y + 2) + j) * (x + 2);
          if (left)
            f(p) = b(q);
          if (right)
            f(p + x + 1) = b(y * z + q);
        });
    Kokkos::parallel_for(
        "unpack_y", Policy(0, static_cast<std::size_t>(x) * z), KOKKOS_LAMBDA(std::size_t q) {
          int i = q % x + 1, k = q / x + 1;
          std::size_t p = static_cast<std::size_t>(k) * (y + 2) * (x + 2) + i;
          if (bottom)
            f(p) = b(2 * y * z + q);
          if (top)
            f(p + (y + 1) * (x + 2)) = b(2 * y * z + x * z + q);
        });
  }
  void periodic_z() {
    if (model.config.boundary[2] != "periodic")
      return;
    auto f = current;
    const int x = nx, y = ny, z = nz;
    Kokkos::parallel_for(
        "periodic_z", Policy(0, static_cast<std::size_t>(x) * y), KOKKOS_LAMBDA(std::size_t q) {
          int i = q % x + 1, j = q / x + 1;
          std::size_t p = static_cast<std::size_t>(j) * (x + 2) + i,
                      plane = static_cast<std::size_t>(x + 2) * (y + 2);
          f(p) = f(p + z * plane);
          f(p + (z + 1) * plane) = f(p + plane);
        });
  }
  void update(double dt, int region) {
    auto a = current, b = next, r = rates, l = loss;
    const int x = nx, y = ny;
    Kokkos::parallel_for(
        "transport", Policy(0, cells), KOKKOS_LAMBDA(std::size_t q) {
          int i = q % x + 1, j = (q / x) % y + 1, k = q / (static_cast<std::size_t>(x) * y) + 1;
          const bool interior = i > 1 && i < x && j > 1 && j < y;
          if ((region == 1 && !interior) || (region == 2 && interior))
            return;
          const std::size_t row = x + 2, plane = row * (y + 2),
                            p = (static_cast<std::size_t>(k) * (y + 2) + j) * row + i;
          double change = -l(q) * a(p) + r(6 * q) * a(p - 1) + r(6 * q + 1) * a(p + 1) +
                          r(6 * q + 2) * a(p - row) + r(6 * q + 3) * a(p + row) +
                          r(6 * q + 4) * a(p - plane) + r(6 * q + 5) * a(p + plane);
          b(p) = a(p) + dt * change;
        });
    Kokkos::fence();
  }

public:
  ParallelEngine(Model &m, MPI_Comm cart)
      : model(m), comm(cart), nx(m.domain.nx), ny(m.domain.ny), nz(m.domain.nz),
        cells(m.domain.count()) {
    if (static_cast<std::size_t>(nx) * nz > INT_MAX / 4 ||
        static_cast<std::size_t>(ny) * nz > INT_MAX / 4)
      throw std::runtime_error("MPI halo exceeds count range");
    lengths = {ny * nz, ny * nz, nx * nz, nx * nz};
    offsets = {0, ny * nz, 2 * ny * nz, 2 * ny * nz + nx * nz};
    MPI_Cart_shift(comm, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(comm, 1, 1, &neighbors[2], &neighbors[3]);
    auto copy = [&](const std::string &name, const std::vector<double> &values) {
      View v(name, values.size());
      auto h = Kokkos::create_mirror_view(v);
      std::copy(values.begin(), values.end(), h.data());
      Kokkos::deep_copy(v, h);
      return v;
    };
    current = copy("concentration", m.initial);
    next = View("next", m.domain.padded());
    rates = copy("incoming_rates", m.rates);
    loss = copy("outgoing_rates", m.loss);
    escape_rate = copy("escape_rates", m.escape);
    has_escape = std::any_of(m.escape.begin(), m.escape.end(), [](double x) { return x > 0; });
    const auto count = static_cast<std::size_t>(2) * nz * (nx + ny);
    send = View("send", count);
    recv = View("recv", count);
    host_send = HostView("host_send", count);
    host_recv = HostView("host_recv", count);
    host_current = HostView("host_concentration", m.domain.padded());
    cached.resize(m.domain.padded());
  }
  void advance(double dt) override {
    double start = seconds();
    periodic_z();
    if (has_escape) {
      double flux = 0;
      auto a = current, e = escape_rate;
      const int x = nx, y = ny;
      Kokkos::parallel_reduce(
          "escape", Policy(0, cells),
          KOKKOS_LAMBDA(std::size_t q, double &sum) {
            int i = q % x + 1, j = (q / x) % y + 1, k = q / (static_cast<std::size_t>(x) * y) + 1;
            auto p = (static_cast<std::size_t>(k) * (y + 2) + j) * (x + 2) + i;
            sum += e(q) * a(p);
          },
          flux);
      escaped += flux * dt * model.config.dx * model.config.dy * model.config.dz;
    }
    Kokkos::fence();
    timing.update += seconds() - start;
    start = seconds();
    pack();
    if (model.config.exchange == "blocking") {
      for (int axis = 0; axis < 2; ++axis) {
        int lo = 2 * axis, hi = lo + 1;
        MPI_Sendrecv(host_send.data() + offsets[lo], lengths[lo], MPI_DOUBLE, neighbors[lo], lo,
                     host_recv.data() + offsets[hi], lengths[hi], MPI_DOUBLE, neighbors[hi], lo,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(host_send.data() + offsets[hi], lengths[hi], MPI_DOUBLE, neighbors[hi], hi,
                     host_recv.data() + offsets[lo], lengths[lo], MPI_DOUBLE, neighbors[lo], hi,
                     comm, MPI_STATUS_IGNORE);
      }
      unpack();
      Kokkos::fence();
      timing.halo += seconds() - start;
      start = seconds();
      update(dt, 0);
      timing.update += seconds() - start;
    } else {
      for (int face = 0; face < 4; ++face) {
        MPI_Irecv(host_recv.data() + offsets[face], lengths[face], MPI_DOUBLE, neighbors[face],
                  face ^ 1, comm, &requests[face]);
        MPI_Isend(host_send.data() + offsets[face], lengths[face], MPI_DOUBLE, neighbors[face],
                  face, comm, &requests[4 + face]);
      }
      timing.halo += seconds() - start;
      start = seconds();
      update(dt, 1);
      timing.update += seconds() - start;
      start = seconds();
      MPI_Waitall(8, requests.data(), MPI_STATUSES_IGNORE);
      timing.exposed_wait += seconds() - start;
      unpack();
      Kokkos::fence();
      timing.halo += seconds() - start;
      start = seconds();
      update(dt, 2);
      timing.update += seconds() - start;
    }
    std::swap(current, next);
  }
  const std::vector<double> &host() override {
    Kokkos::deep_copy(host_current, current);
    std::copy(host_current.data(), host_current.data() + cached.size(), cached.begin());
    return cached;
  }
  double sum(double v) override {
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, comm);
    return r;
  }
  double maximum(double v) override {
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, comm);
    return r;
  }
  double minimum(double v) override {
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MIN, comm);
    return r;
  }
  void barrier() override {
    MPI_Barrier(comm);
  }
  std::string backend() const override {
    return Kokkos::DefaultExecutionSpace::name();
  }
  int concurrency() const override {
    return Kokkos::DefaultExecutionSpace().concurrency();
  }
};
std::array<int, 2> choose_dims(const Config &c, int ranks) {
  if (c.px || c.py) {
    if (c.px < 1 || c.py < 1 || c.px * c.py != ranks || c.px > c.nx || c.py > c.ny)
      throw std::runtime_error("Invalid px,py for rank count and grid");
    return {c.px, c.py};
  }
  if (c.decomposition == "slab") {
    if (ranks > c.nx)
      throw std::runtime_error("More ranks than x cells");
    return {ranks, 1};
  }
  double best = std::numeric_limits<double>::infinity();
  std::array<int, 2> result{0, 0};
  for (int x = 1; x <= ranks; ++x)
    if (ranks % x == 0 && x <= c.nx && ranks / x <= c.ny) {
      int y = ranks / x;
      double score =
          std::abs(std::log((static_cast<double>(c.nx) / x) / (static_cast<double>(c.ny) / y)));
      if (score < best) {
        best = score;
        result = {x, y};
      }
    }
  if (result[0] == 0)
    throw std::runtime_error("No nonempty horizontal decomposition");
  return result;
}
} // namespace dispersion
int main(int argc, char **argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED)
    MPI_Abort(MPI_COMM_WORLD, 1);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc < 2)
      throw std::runtime_error("Usage: dispersion_mpi CONFIG [key=value ...]");
    auto c = dispersion::read_config(argv[1], dispersion::command_overrides(argc, argv));
    auto dims = dispersion::choose_dims(c, ranks);
    int periods[2] = {c.boundary[0] == "periodic", c.boundary[1] == "periodic"};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &cart);
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int x0 = c.nx * coords[0] / dims[0], x1 = c.nx * (coords[0] + 1) / dims[0],
        y0 = c.ny * coords[1] / dims[1], y1 = c.ny * (coords[1] + 1) / dims[1];
    c.px = dims[0];
    c.py = dims[1];
    dispersion::Domain d{x1 - x0, y1 - y0, c.nz, x0, y0, c.nx, c.ny, rank, ranks};
    dispersion::Model model(c, d);
    double initial_sum = 0;
    MPI_Allreduce(&model.initial_sum, &initial_sum, 1, MPI_DOUBLE, MPI_SUM, cart);
    dispersion::normalize(model, initial_sum);
    Kokkos::InitializationSettings settings;
#ifdef KOKKOS_ENABLE_CUDA
    MPI_Comm local;
    MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank = 0;
    MPI_Comm_rank(local, &local_rank);
    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices < 1)
      throw std::runtime_error("No visible CUDA device");
    settings.set_device_id(local_rank % devices);
    MPI_Comm_free(&local);
#endif
    Kokkos::initialize(settings);
    {
      dispersion::ParallelEngine engine(model, cart);
      dispersion::run(model, engine);
    }
    Kokkos::finalize();
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << " ERROR: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
