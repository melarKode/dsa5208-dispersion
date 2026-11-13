cmake_minimum_required(VERSION 3.16)
project(lagrangian_particle_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# MPI ranks started over ssh (as OpenMPI does on Vanda) do not inherit the
# module environment and would load the older system libstdc++/libgomp. Embed
# the compiler's own runtime-library directory so every launch finds them.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                  OUTPUT_VARIABLE _lps_libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_lps_gcc_libdir "${_lps_libstdcxx}" DIRECTORY)
  if(IS_DIRECTORY "${_lps_gcc_libdir}")
    list(APPEND CMAKE_BUILD_RPATH "${_lps_gcc_libdir}")
    list(APPEND CMAKE_INSTALL_RPATH "${_lps_gcc_libdir}")
  endif()
endif()

# Serial reference library: physics, configuration and output. Both the serial
# and the MPI+Kokkos executables link this one library, so they run the same
# compiled per-particle code.
add_library(lps
  src/config.cpp
  src/decomposition.cpp
  src/diagnostics.cpp
  src/domain.cpp
  src/particles.cpp
  src/stepper.cpp
  src/wind.cpp
  src/urban.cpp)
target_include_directories(lps PUBLIC include)
target_compile_options(lps PRIVATE -Wall -Wextra -Wpedantic)

add_executable(lagrangian_solver src/main.cpp)
target_link_libraries(lagrangian_solver PRIVATE lps)

add_executable(lps_unit_tests tests/unit_tests.cpp)
target_link_libraries(lps_unit_tests PRIVATE lps)

enable_testing()
add_test(NAME unit_tests COMMAND lps_unit_tests)

add_executable(lps_urban_tests tests/urban_tests.cpp)
target_link_libraries(lps_urban_tests PRIVATE lps)
add_test(NAME urban_tests COMMAND lps_urban_tests ${CMAKE_CURRENT_BINARY_DIR}/urban-fixture)

add_executable(lps_decomposition_tests tests/decomposition_tests.cpp)
target_link_libraries(lps_decomposition_tests PRIVATE lps)
add_test(NAME decomposition_tests COMMAND lps_decomposition_tests)

# MPI + Kokkos solver (CPU / OpenMP first). Off by default so the serial build
# needs neither MPI nor Kokkos.
option(LPS_PARALLEL "Build the MPI+Kokkos solver lagrangian_parallel" OFF)
option(LPS_FETCH_KOKKOS "Download and build Kokkos (OpenMP) when no installed Kokkos is found" OFF)
set(LPS_KOKKOS_TAG "4.7.04" CACHE STRING "Kokkos release used with LPS_FETCH_KOKKOS")

if(LPS_PARALLEL)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_package(Kokkos CONFIG QUIET)
  if(NOT Kokkos_FOUND)
    if(NOT LPS_FETCH_KOKKOS)
      message(FATAL_ERROR "Kokkos not found. Set Kokkos_ROOT to an installation, or pass -DLPS_FETCH_KOKKOS=ON.")
    endif()
    message(STATUS "Fetching Kokkos ${LPS_KOKKOS_TAG} with the OpenMP backend")
    include(FetchContent)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG ${LPS_KOKKOS_TAG}
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(kokkos)
  endif()

  add_executable(lagrangian_parallel
    src/parallel/main_parallel.cpp
    src/parallel/kokkos_particles.cpp
    src/parallel/mpi_exchange.cpp)
  target_link_libraries(lagrangian_parallel PRIVATE lps Kokkos::kokkos MPI::MPI_CXX)
  target_compile_options(lagrangian_parallel PRIVATE -Wall -Wextra)
endif()
