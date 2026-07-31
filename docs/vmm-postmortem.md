# VMM Postmortem — 48 Hours of cuMemCreate Debugging

## The Goal

Use CUDA Virtual Memory Management (`cuMemCreate`) to quarantine defective physical pages **globally** — protecting all allocators (cudaMalloc, PyTorch, everything) from the bad GDDR6 chip, not just llama.cpp.

## The Problem

`cuMemCreate` handles never exposed the defective physical pages. Zero bad handles detected out of 11,406 tested, despite the same GPU showing 12,909 word errors through `cudaMalloc` in the same boot session.

## Timeline

### Phase 1: TLB Reuse (2 hours)
**Hypothesis:** Reusing the same scratch VA for all 11,406 handles left stale TLB entries. Fix: fresh `cuMemAddressReserve/Free` per handle.
**Result:** Still 0 bad handles. TLB not the issue.

### Phase 2: L2 Cache Masking (8 hours)
**Hypothesis:** TU102's 6 MB L2 cache hides defects when testing handles individually (2 MB < 6 MB). Fix: batch 24 handles together (48 MB) to force L2 eviction.
**Result:** Still 0. The batching approach on VMM-mapped memory doesn't expose the defect.

### Phase 3: Detection Kernel Validity (12 hours)
**Hypothesis:** Our `detect_to_flags` function (host-side cudaHostAllocMapped flags) is broken. We built `vmm_test_chunk.cu` — a character-for-character copy of the proven `vram_mapper_detailed.cu` detection code. Same compiler (VS 2022), same CUDA 12.4, same boot.
**Result:** `vram_mapper_detailed.exe` from July 29 → 12,909 errors. `vmm_test_chunk.exe` compiled Aug 1 → 0 errors. The detection code, when compiled fresh, doesn't work — even on cudaMalloc memory. Recompiling the original `vram_mapper_detailed.cu` from source today ALSO works (13,006 errors). Something about the binary layout or CRT state makes our copy fail.

### Phase 4: The Hybrid Test (4 hours)
**Approach:** Start from the WORKING `vram_mapper_detailed.cu`, add only the VMM Phase 2 code. No detection changes.
**Result:** Phase 1 (cudaMalloc) → 12,909 errors ✅. Phase 2 (cuMemCreate, same detection, same process) → 0 errors ❌. **VMM definitively does not expose the same physical pages as cudaMalloc.**

## Root Cause

On WDDM / Turing (TU102) under CUDA 12.4, `cuMemCreate` with `CU_MEM_ALLOCATION_TYPE_PINNED` allocates from a **different physical memory pool** than `cudaMalloc`. The defective GDDR6 chip #16 is present in the cudaMalloc pool but absent from the VMM pool.

## Key Evidence

| Test | Allocator | Errors | Same boot? |
|---|---|---|---|
| vram_mapper_detailed.exe | cudaMalloc | 12,909 | ✅ |
| vmm_probe hybrid Phase 1 | cudaMalloc | 12,909 | ✅ |
| vmm_probe hybrid Phase 2 | cuMemCreate | 0 | ✅ |
| vmm_validation (11406 handles) | cuMemCreate | 0 | ✅ |
| vmm_probe VMM-only (16 handles) | cuMemCreate | 0 | ✅ |

All tests run back-to-back in the same boot session. The cudaMalloc path consistently finds the defect. The cuMemCreate path never does.

## Implications

1. **Strategy B (VMM) is not viable on WDDM/Turing.** The physical pages we need to quarantine are simply not allocated through cuMemCreate.
2. **Linux behavior may differ.** Without WDDM's VidMm layer, cuMemCreate might draw from the same physical pool. Worth testing for anyone with a dual-boot setup.
3. **For PyTorch/ComfyUI on Windows:** The cudaMalloc chunked guard approach (vramguard.dll) also fails because WDDM's VA→PA mapping is deterministic per-allocation-sequence but PyTorch uses a different allocation sequence than the guard.

## Alternative Paths

| Approach | Status |
|---|---|
| cudaMalloc guard (llama.cpp) | ✅ Works |
| VMM / cuMemCreate | ❌ Different physical pool |
| DPR (Dynamic Page Retirement) | 🔄 Untested |
| vramguard.dll (PyTorch) | ❌ VA→PA mismatch |
