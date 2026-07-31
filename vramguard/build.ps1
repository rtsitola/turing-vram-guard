# build.ps1 — Build vramguard.dll for Windows
#
# PREREQUISITES:
#   - Visual Studio 2019 or 2022 with C++ workload
#   - CUDA Toolkit 12.x (tested with 12.4)
#   - CMake 3.18+
#
# NOTE: CUDA 12.4 + MSVC 14.51 has a known nvcc incompatibility.
#        Either use VS 2022 (MSVC 14.4x) or CUDA 12.6+.
#        This script auto-detects VS 2022 first, falls back to VS 2019.
#
# USAGE:
#   powershell -ExecutionPolicy Bypass -File build.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Find Visual Studio ──
$vsPath = $null
$vsVer  = ""

# Try VS 2022 first (MSVC 14.4x — works with CUDA 12.4)
$vs2022  = "K:\Microsoft Visual Studio\17\VC\Auxiliary\Build\vcvarsall.bat"
$vs2022b = "K:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$vs2022c = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$vs2022e = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
$vs2022p = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$vs2022k = "K:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

# Then try VS 2019
$vs2019 = "K:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
$vs2019b = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
$vs2019c = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
$vs2019d = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
$vs2019e = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
$vs2019f = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat"

foreach ($p in @($vs2022, $vs2022b, $vs2022c, $vs2022e, $vs2022p, $vs2022k, $vs2019, $vs2019b, $vs2019c, $vs2019d, $vs2019e, $vs2019f)) {
    if (Test-Path $p) {
        $vsPath = $p
        if ($p -like "*2022*" -or $p -like "*\17\*") { $vsVer = "2022" } else { $vsVer = "unknown" }
        Write-Host "Found VS ${vsVer}: $p"
        break
    }
}

if (-not $vsPath) {
    Write-Error "Visual Studio not found. Checked: VS 2022 (Community/Enterprise/Pro), VS 2019 (K:\, C:\)"
    exit 1
}

if ($vsVer -eq "2019") {
    Write-Warning "VS 2019 detected. CUDA 12.4 + MSVC 14.51 may fail with nvcc."
    Write-Warning "If build fails, install VS 2022 or upgrade to CUDA 12.6+."
}

# ── CUDA path ──
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
}
# Force CMake to use THIS specific CUDA installation
$env:CUDA_PATH = $cudaPath
# Remove any CUDA bin from PATH that might point to a different version
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notlike "*NVIDIA GPU Computing Toolkit\bin*" }) -join ';'
Write-Host "CUDA: $cudaPath"

# ── Build ──
$buildDir = "$PSScriptRoot\build"

# Run configure AND build in the same cmd session so vcvarsall.bat's
# environment (PATH, INCLUDE, LIB) is active for both nvcc and cl.exe.
Write-Host "Configuring + Building (one session)..."
cmd /c "`"$vsPath`" x64 && cmake -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=`"$cudaPath`" -S `"$PSScriptRoot`" && cmake --build `"$buildDir`" --config Release"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit 1
}

$dll = Get-ChildItem -Path $buildDir -Recurse -Filter "vramguard.dll" | Select-Object -First 1
if ($dll) {
    Write-Host "`nBUILD SUCCESSFUL: $($dll.FullName)" -ForegroundColor Green
    Write-Host "Copy this DLL and sitecustomize.py to your ComfyUI venv." -ForegroundColor Green
} else {
    Write-Error "DLL not found in build output."
}
