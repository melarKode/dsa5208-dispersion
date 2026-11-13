#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int failed = 0;
  Kokkos::initialize(argc, argv);
  {
    constexpr int n = 10000;
    long long local = 0;
    Kokkos::parallel_reduce(
        "known_sum", Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const int i, long long &sum) {
          sum += static_cast<long long>(i + 1) * (rank + 1);
        },
        local);
    Kokkos::fence();
    long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    const long long expected = (1LL * n * (n + 1) / 2) * (1LL * size * (size + 1) / 2);
    failed = global != expected;
    char host[MPI_MAX_PROCESSOR_NAME];
    int length = 0;
    MPI_Get_processor_name(host, &length);
    for (int turn = 0; turn < size; ++turn) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == turn) {
        std::cout << "rank=" << rank << " host=" << host
                  << " backend=" << Kokkos::DefaultExecutionSpace::name()
                  << " concurrency=" << Kokkos::DefaultExecutionSpace().concurrency() << std::endl;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << (failed ? "FAIL" : "PASS") << " ranks=" << size << " sum=" << global
                << " expected=" << expected << std::endl;
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return failed;
}
