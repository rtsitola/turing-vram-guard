# VRAM Guard DLL — PyTorch/ComfyUI Memory Guard (Experimental)

Standalone CUDA DLL that scans GPU VRAM at process startup and isolates defective memory regions.

## Status: ⚠️ EXPERIMENTAL — does NOT protect PyTorch on WDDM

The cudaMalloc chunked guard works for single-allocator scenarios (llama.cpp/LM Studio) but fails for PyTorch because WDDM assigns different VA→PA mappings to different allocation sequences. See [../docs/vmm-postmortem.md](../docs/vmm-postmortem.md) for details.

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
