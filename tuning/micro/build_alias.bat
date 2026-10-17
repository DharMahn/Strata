@echo off
rem Builds tuning\micro\alias_test.exe (plain HIP, no engine sources): a host buffer at a registered memory's device
rem alias - the repro of the Windows HIP bug that the engine's alias guard works around (tuning/LOG.md, 2026-10-03).
setlocal
call "%~dp0env.bat" || exit /b 1
set OUT=%~dp0
"%R%/lib/llvm/bin/clang++.exe" -x hip --rocm-path=%R% --rocm-device-lib-path=%R%/lib/llvm/amdgcn/bitcode ^
  -isystem "%S%/build-hip/msvc-overlay" -O2 -std=c++20 --offload-arch=%ARCH% -D_DLL -D_MT -Xclang --dependent-lib=msvcrt ^
  -D__HIP_PLATFORM_AMD__=1 -isystem %R%/include %OUT%alias_test.cu -L%R%/lib -lamdhip64 -o "%OUT%alias_test.exe" || exit /b 1
echo built %OUT%alias_test.exe
