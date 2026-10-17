@echo off
rem Shared setup for the micro-benchmark build scripts (call "%~dp0env.bat").
rem Sets: R = the engine's ROCm SDK (installed in .venv by setup), S = the Strata folder (forward slashes),
rem       ARCH = the GPU to compile for (gfx1030 = RX 6800/6900, gfx1031 = RX 6700 XT; set ARCH first to override).
rem Run setup / START-HERE.bat --backend hip once first: it installs the SDK and creates build-hip\msvc-overlay.
for %%I in ("%~dp0..\..") do set "S=%%~fI"
set "S=%S:\=/%"
set "R=%S%/.venv/Lib/site-packages/_rocm_sdk_devel"
if not defined ARCH set "ARCH=gfx1030"
set "VSW=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSW%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo Visual Studio Build Tools (C++) not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
