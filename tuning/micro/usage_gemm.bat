@echo off
rem Prints the kernel resource usage (VGPRs, LDS, occupancy) of src/prefill/gemm.cu for %ARCH%.
setlocal
call "%~dp0env.bat" || exit /b 1
"%R%/lib/llvm/bin/clang++.exe" -x hip --rocm-path=%R% --rocm-device-lib-path=%R%/lib/llvm/amdgcn/bitcode ^
  -isystem "%S%/build-hip/msvc-overlay" -O3 -DNDEBUG -std=c++20 --offload-arch=%ARCH% --cuda-device-only -c ^
  -D_DLL -D_MT -include %S%/include/strata/hip_compat/cuda_runtime.h ^
  -DSTRATA_HIP_ARCHS=\"%ARCH%\" -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 ^
  -I%S%/third_party/ggml -I%S%/include -I%S%/include/strata/hip_compat -isystem %R%/include ^
  -Rpass-analysis=kernel-resource-usage %S%/src/prefill/gemm.cu -I%S%/src/prefill -o NUL
