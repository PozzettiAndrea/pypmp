cmake_minimum_required(VERSION 3.16...3.27)
project(pypmp LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Configure pmp-library as a static library (no GUI, no apps)
# These MUST be set before add_subdirectory so pmp's CMakeLists.txt sees them.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use double precision (matches numpy float64)
add_definitions(-DPMP_SCALAR_TYPE_64)

# Disable everything we don't need
set(PMP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PMP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(PMP_BUILD_VIEWERS OFF CACHE BOOL "" FORCE)
set(PMP_INSTALL OFF CACHE BOOL "" FORCE)
set(PMP_STRICT_COMPILATION OFF CACHE BOOL "" FORCE)

# Apply patches to pmp-library before building
execute_process(
  COMMAND git apply --check ${CMAKE_CURRENT_SOURCE_DIR}/patches/remeshing_verbose.patch
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pmp-library
  RESULT_VARIABLE patch_check_result
  ERROR_QUIET
)
if(patch_check_result EQUAL 0)
  execute_process(
    COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/remeshing_verbose.patch
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pmp-library
  )
endif()

# Build pmp — EXCLUDE_FROM_ALL suppresses pmp's own install rules
# so only our nanobind module gets packaged into the wheel.
add_subdirectory(pmp-library EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Build the nanobind extension module
# ---------------------------------------------------------------------------
nanobind_add_module(
  _pypmp
  STABLE_ABI
  src/_pypmp.cpp
)

# Link pmp + OpenMP (pmp uses OpenMP internally)
find_package(OpenMP)
target_link_libraries(_pypmp PRIVATE pmp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_pypmp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(_pypmp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/pmp-library/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler-specific optimization
if(MSVC)
  target_compile_options(_pypmp PRIVATE /O2)
else()
  target_compile_options(_pypmp PRIVATE -O3)
endif()

# Install into the pypmp Python package directory
install(TARGETS _pypmp LIBRARY DESTINATION pypmp)
