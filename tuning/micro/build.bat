@echo off
rem Builds tuning\micro\mmvq_bench.exe from the CURRENT src/kernels/cuda/{native_mmvq,iq_kernels}.cu with the engine's
rem own HIP flags (build-hip/build.ninja), so a kernel variant can be timed without rebuilding the engine.
setlocal
call "%~dp0env.bat" || exit /b 1
set OUT=%~dp0
"%R%/lib/llvm/bin/clang++.exe" -x hip --rocm-path=%R% --rocm-device-lib-path=%R%/lib/llvm/amdgcn/bitcode ^
  -isystem "%S%/build-hip/msvc-overlay" -O3 -DNDEBUG -std=c++20 --offload-arch=%ARCH% --offload-arch=gfx1036 ^
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -include %S%/include/strata/hip_compat/cuda_runtime.h ^
  -DSTRATA_HIP_ARCHS=\"%ARCH%\" -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 ^
  -I%S%/third_party/ggml -I%S%/include -I%S%/include/strata/hip_compat -isystem %R%/include ^
  %S%/tuning/micro/mmvq_bench.cu %S%/src/kernels/cuda/native_mmvq.cu %S%/src/kernels/cuda/iq_kernels.cu ^
  -L%R%/lib -lamdhip64 -o "%OUT%mmvq_bench.exe" || exit /b 1
echo built %OUT%mmvq_bench.exe
