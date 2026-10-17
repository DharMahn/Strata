@echo off
rem Builds tuning\micro\probe.exe (the prompt path's BLAS shapes, BF16 vs FP16) with the engine's ROCm SDK.
setlocal
call "%~dp0env.bat" || exit /b 1
set OUT=%~dp0
"%R%/lib/llvm/bin/clang++.exe" -x hip --rocm-path=%R% --rocm-device-lib-path=%R%/lib/llvm/amdgcn/bitcode ^
  -O3 -DNDEBUG -std=c++20 --offload-arch=%ARCH% -D_DLL -D_MT -Xclang --dependent-lib=msvcrt ^
  -isystem "%S%/build-hip/msvc-overlay" -D__HIP_PLATFORM_AMD__=1 -isystem %R%/include %~dp0probe.cu ^
  -L%R%/lib -lamdhip64 -o "%OUT%probe.exe" || exit /b 1
echo built %OUT%probe.exe
