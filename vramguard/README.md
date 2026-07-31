# VRAM Guard DLL — GPU Memory Diagnostic Tool

Standalone CUDA DLL that scans GPU VRAM and identifies defective memory regions.

## Status: Diagnostic tool — does NOT protect PyTorch/ComfyUI

The cudaMalloc chunked guard successfully isolates defective pages within a single allocation sequence (e.g., llama.cpp/LM Studio). However, under WDDM, different processes (and even different allocation sequences within the same process) receive different VA→PA mappings. This means a guard installed via `vg_install()` cannot protect PyTorch tensors — PyTorch allocates through a different sequence and may receive the defective physical pages regardless.

**For PyTorch/ComfyUI protection, the only viable paths are:**
- VMM / cuMemCreate (dead end on WDDM — see [../docs/vmm-postmortem.md](../docs/vmm-postmortem.md))
- DPR / Dynamic Page Retirement (untested — see [../docs/ecc-analysis.md](../docs/ecc-analysis.md))
- Run ComfyUI on a different GPU (e.g., 3070 Ti)

**For llama.cpp/LM Studio, use the ggml-cuda.cu patch instead.** See [../ggml-cuda.patch](../ggml-cuda.patch) and the main [README](../README.md).

## Files

| File | Purpose |
|---|---|
| `vramguard.cpp` | Guard initialization + canary thread |
| `detection.cu` | GPU-side detection kernels (32-pattern suite) |
| `vramguard.h` | Public API (`vg_install`, `vg_status`) |
| `detection.h` | Detection function declarations |
| `sitecustomize.py` | CPython auto-injection (placed alongside python.exe) |
| `CMakeLists.txt` | CMake build (Windows, CUDA 12+) |
| `build.ps1` | Build script (auto-detects VS + CUDA) |

## Build

```powershell
cd vramguard
powershell -ExecutionPolicy Bypass -File build.ps1
```

Output: `build/vramguard.dll`

## Integration

```python
# sitecustomize.py — auto-loaded by CPython before any user imports
import ctypes
dll = ctypes.CDLL("vramguard.dll")
rc = dll.vg_install(0)  # 0=OK, -1=error, 1=no defect found
```

## Why It Doesn't Work for PyTorch

Under WDDM, the VA→PA mapping is deterministic **per allocation sequence**. Our guard calls `cudaMalloc` in a specific order, finds the bad chunk, and reserves it. PyTorch calls `cudaMalloc` in a DIFFERENT order — getting DIFFERENT physical pages. The bad pages our guard reserved are in OUR VA space; PyTorch sees them at different VAs and allocates there freely.

Result: SDXL produces pure black images (NaN in tensors). Same workflow produces clean output on a healthy 3070 Ti.

## Future

- VMM (cuMemCreate) would solve this but is a dead end on WDDM (see docs)
- DPR (Dynamic Page Retirement) might solve it permanently at hardware level
- Linux testing needed — non-WDDM behavior may differ
