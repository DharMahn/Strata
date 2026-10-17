@echo off
rem Builds tuning\micro\attn_bench.exe from the CURRENT src/kernels/cuda/qsa_{prompt,decode}_attn.cu with the engine's
rem HIP flags, so an attention kernel variant can be checked and timed without rebuilding the engine.
setlocal
call "%~dp0env.bat" || exit /b 1
set OUT=%~dp0
"%R%/lib/llvm/bin/clang++.exe" -x hip --rocm-path=%R% --rocm-device-lib-path=%R%/lib/llvm/amdgcn/bitcode ^
  -isystem "%S%/build-hip/msvc-overlay" -O3 -DNDEBUG -std=c++20 --offload-arch=%ARCH% --offload-arch=gfx1036 ^
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -include %S%/include/strata/hip_compat/cuda_runtime.h ^
  -DSTRATA_HIP_ARCHS=\"%ARCH%\" -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 ^
  -I%S%/third_party/ggml -I%S%/include -I%S%/include/strata/hip_compat -isystem %R%/include ^
  %S%/tuning/micro/attn_bench.cu %S%/src/kernels/cuda/qsa_prompt_attn.cu %S%/src/kernels/cuda/qsa_decode_attn.cu ^
  -L%R%/lib -lamdhip64 -o "%OUT%attn_bench.exe" || exit /b 1
echo built %OUT%attn_bench.exe
