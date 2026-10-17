# Opt-in HIP configuration. Strata's CUDA-shaped kernels target wave32 RDNA3 / RDNA4 (64 KiB LDS per workgroup,
# the signed dot4 instruction). CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it
# is not on PATH.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
endif()
# Validated on real cards: gfx1100 (RX 7900 XT / XTX) and gfx1201 (RX 9070 / 9070 XT, Radeon AI PRO R9700).
# The other RDNA3 / RDNA4 wave32 chips have the same LDS limit and dot4 instruction and build the same code, but
# the maintainers have not run them (community reports: gfx1102 #192, gfx1200 #176).
set(_strata_hip_validated gfx1100 gfx1201)
set(_strata_hip_unvalidated gfx1101 gfx1102 gfx1200 gfx1030 gfx1031 gfx1032 gfx1034)   # gfx103x: RDNA2 port (wave32, 64 KiB LDS, sdot4)
# "Companion" architectures are compiled in but never run on: on Windows the HIP loader registers a code object for
# every GPU in the PC, including the CPU's integrated Radeon (gfx1036 on Ryzen 7000/9000), and fails the launch
# with "device kernel image is invalid" on the real card when one of them has no code.  They stay out of
# STRATA_HIP_ARCHS, so the engine's own device check still refuses to run on them.
set(_strata_hip_companion gfx1035 gfx1036 gfx1037 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153)   # integrated Radeons
set(STRATA_HIP_ARCH_LIST "")
foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")      # gfx1100:xnack- -> gfx1100
  if(_base IN_LIST _strata_hip_companion)
    continue()
  endif()
  if(_base IN_LIST _strata_hip_validated)
  elseif(_base IN_LIST _strata_hip_unvalidated)
    message(WARNING "Strata HIP: ${_base} builds, but it is not validated on a real card yet; please report results")
  else()
    message(FATAL_ERROR
      "Strata HIP supports wave32 gfx1100 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}); "
      "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND STRATA_HIP_ARCH_LIST "${_base}")
endforeach()
list(REMOVE_DUPLICATES STRATA_HIP_ARCH_LIST)
if(NOT STRATA_HIP_ARCH_LIST)
  message(FATAL_ERROR "Strata HIP: CMAKE_HIP_ARCHITECTURES is empty")
endif()
# The compiled architectures reach the runtime device check (src/core/device.cu) as "gfx1100,gfx1201": a binary
# carried to a card it has no code for stops at startup with a clear message instead of "invalid device function".
string(REPLACE ";" "," STRATA_HIP_ARCHS "${STRATA_HIP_ARCH_LIST}")

# Windows: the newest MSVC STL (14.51, Visual Studio 2026) declares constexpr isless/isgreater/... overloads that
# Clang's HIP math wrapper (__clang_hip_cmath.h) re-declares as __device__, which stops every HIP compile
# ("__device__ function 'isless' cannot overload __host__ __device__ function").  The overloads are an optimisation
# only: the HIP compiles get a copy of <cmath> without them, from a folder in the build tree - the Visual Studio
# install is not touched.  (Set before enable_language(HIP): its compiler test compile needs the flag.)
if(WIN32)
  set(_strata_cmath_dirs ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} $ENV{INCLUDE})
  find_file(STRATA_MSVC_CMATH cmath PATHS ${_strata_cmath_dirs} NO_DEFAULT_PATH)
  if(STRATA_MSVC_CMATH)
    file(READ "${STRATA_MSVC_CMATH}" _strata_cmath)
    string(REGEX REPLACE "#ifdef __clang__([\r\n]+)// Clang provides efficient builtins"
           "#if defined(__clang__) && !defined(__HIP__)\\1// Clang provides efficient builtins"
           _strata_cmath_fixed "${_strata_cmath}")
    if(NOT _strata_cmath_fixed STREQUAL _strata_cmath)
      set(STRATA_MSVC_OVERLAY "${CMAKE_BINARY_DIR}/msvc-overlay")
      file(MAKE_DIRECTORY "${STRATA_MSVC_OVERLAY}")
      file(WRITE "${STRATA_MSVC_OVERLAY}/cmath" "${_strata_cmath_fixed}")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem \"${STRATA_MSVC_OVERLAY}\"")
      message(STATUS "Strata: HIP builds use ${STRATA_MSVC_OVERLAY}/cmath (MSVC STL constexpr isless vs Clang's HIP wrapper)")
    endif()
  endif()
endif()

enable_language(HIP)

find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "The ROCm hip CMake package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "The ROCm hipblas CMake package did not provide roc::hipblas")
endif()
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Strata: hipBLASLt not found; solution-table dispatch is unavailable")
endif()

# HIP's link step produces a PIE; make Strata and ggml objects PIC for the ROCm linker.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(STRATA_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
add_library(strata_hip_runtime INTERFACE)
target_include_directories(strata_hip_runtime BEFORE INTERFACE
  "${STRATA_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1 "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\"")
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
foreach(_language IN ITEMS CXX HIP)
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# CMake does not infer HIP from Strata's existing CUDA-shaped .cu suffixes.
file(GLOB_RECURSE _strata_hip_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
if(_strata_hip_sources)
  set_source_files_properties(${_strata_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
foreach(_source IN ITEMS tests/hip/intrinsics.cpp tests/hip/native_qsa_score.cpp)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    set_source_files_properties("${_source}" PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "Strata: HIP enabled, arch ${STRATA_HIP_ARCHS}")
