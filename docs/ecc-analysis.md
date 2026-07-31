# ECC Analysis — Why ECC ON Crashes, Why ECC OFF + Guard Works

## The Dilemma

On a GPU with defective GDDR6 chips:
- **ECC ON:** The driver detects multi-bit errors and triggers TDR (Timeout Detection and Recovery) → GPU reset → process crash. Useless for inference.
- **ECC OFF:** The driver ignores errors, but stuck-at-0 cells silently return zero → corrupted model output. Useless without mitigation.

## Our Approach: ECC OFF + Software Guard

ECC is kept OFF to avoid TDR crashes. The software guard in `ggml_cuda_init()` pre-allocates all VRAM, finds the bad chunk, and permanently reserves it — preventing the CUDA allocator from ever assigning those physical pages to model weights or inference buffers.

## Why ECC ON Crashes

```
cudaMalloc hits bad region (chip #16)
         ↓
GPU reads stuck-at-0 cells
         ↓
DRAM Uncorrectable Error (multi-bit, ECC can't fix)
         ↓
Driver raises Xid 48 (Double Bit Error)
         ↓
TDR kicks in → GPU reset → cudaDeviceSynchronize() returns error
         ↓
Process killed
```

The crash is correct behavior — the hardware is genuinely faulty. But it makes ECC ON unusable for any workload that touches the bad region.

## ECC Counters

From `nvidia-smi -i 1 -q -d ECC`:

```
DRAM Correctable:     3
DRAM Uncorrectable: 259
```

Important: these are **cumulative read-event counts**, not individual cell failures. Every time the GPU reads a stuck-at-0 cell, it increments. A single defective word scanned 259 times = 259 uncorrectable events. The counter does NOT indicate progressive degradation unless it keeps increasing.

## L2 Cache Interaction

TU102 has **6 MB of L2 cache**. This defines the minimum detectable chunk size:

| Chunk Size | L2 Ratio | Detection |
|---|---|---|
| 128 MB | 21× | ✅ Full detection |
| 32 MB | 5× | ✅ Partial |
| 16 MB | 2.7× | ✅ Partial |
| 8 MB | 1.3× | ⚠️ Marginal |
| 4 MB | 0.67× | ❌ L2 masks defect |
| 2 MB | 0.33× | ❌ Never touches DRAM |

At chunk sizes ≤ 6 MB, the fill→check pattern stays entirely in L2 cache, never reaching DRAM. The stuck-at-0 cells are never read, so ECC (if ON) never fires, and software detection (if OFF) never finds the defect.

**8 MB is the practical floor** — close enough to the 6 MB L2 that eviction pressure still forces some DRAM reads, revealing the defect.

## Future: Dynamic Page Retirement (DPR)

NVIDIA's driver supports retiring defective pages permanently by writing to InfoROM:

1. Enable ECC: `nvidia-smi -i N -e 1` → reboot
2. Run VRAM stress to surface errors (will TDR crash)
3. Reboot → retired pages are blacklisted from all allocations
4. Disable ECC: `nvidia-smi -i N -e 0` → blacklist persists

The documented cap is ~4 MB — may not cover our full ~8 MB defect. Untested on this card due to ECC ON TDR crashes during the surfacing step.
