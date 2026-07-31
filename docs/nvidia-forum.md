# NVIDIA Developer Forum — Full Exchange

## Original Post (condensed)

**Subject:** Turing sm_75 FP16 Tensor Core corruption — root cause: GDDR6 defect

After 32 custom CUDA builds, cross-backend (CUDA+Vulkan) validation, and 5+ VRAM diagnostic tools, we identified that a Quadro RTX 6000 (Turing sm_75) has a defective GDDR6 chip (#16) with stuck-at-0 cells at ~1.719 GB physical address.

Key findings:
- `nvidia-smi -q -d ECC`: 259 DRAM Uncorrectable errors
- `vram_mapper_detailed` (custom tool): 12,909 word errors at chunk #55
- OCCT VRAM stress: 604,057 errors in 30 minutes
- ECC ON causes TDR crashes on multi-bit errors
- FurMark passes — graphics pool ≠ compute pool

Workaround: 24 MB VRAM guard in `ggml_cuda_init()` (llama.cpp).

## NVIDIA Response — athkumar (NVIDIA Staff)

> This is a textbook piece of debugging, and your conclusion is right. Thirty-two builds and 15 hours to get from "BPE tokenizers are broken" to a specific defective memory chip is a lot of persistence, and the write-up is more useful than most bug reports I see.
>
> The tokenizer correlation was a red herring. BPE and SentencePiece models do not stress different arithmetic paths in any way that would explain garbage output; what they do differently is allocate, so only some runs placed hot data on the bad region.
>
> The two steps that actually isolated it:
> 1. Run the same binary and model on a second GPU
> 2. Check ECC before anything else
>
> 259 uncorrectable DRAM errors is conclusive on its own.
>
> The card needs replacing. A Quadro RTX 6000 with uncorrectable DRAM errors is a straightforward warranty case.
>
> I would not run the region-reservation workaround in production. The bad region moves between reboots because of CUDA virtual address remapping, and 259 uncorrectable errors describe a failing chip rather than a fixed set of bad cells.
>
> Thanks for coming back and posting the resolution instead of just walking away once you solved it.

## Our Follow-up Response

> Thanks for the detailed response and RMA guidance.
>
> **The guard has been tightened.** Further testing at finer granularities showed the defect concentrates in ~8 MB. The guard now uses 8 MB chunks (±1 neighbor = 24 MB) instead of 128 MB.
>
> **Canary + rescan added.** `ggml_cuda_guard_verify()` and `ggml_cuda_guard_rescan()` handle boot-drift and WDDM eviction without restart.
>
> **VMM (cuMemCreate) is a dead end on WDDM/Turing.** The driver allocates VMM handles from a different physical pool than cudaMalloc — the defective chip is never exposed.
>
> **Dynamic Page Retirement — untested, could be permanent.** The driver can retire up to ~4 MB of defective pages via ECC page retirement. Blacklist persists across reboots and survives ECC OFF.
>
> **On the "failing chip" warning — our data suggests a stable manufacturing defect.** The 259 ECC errors are cumulative read-event counts, not individual cell failures. The counter has not increased. Across multiple boots and tests spanning a week, the defect shows the same physical address, same pattern (stuck-at-0), and consistent error magnitude. No expansion observed. This looks like a manufacturing-level silicon defect, not progressive failure.
>
> Thanks for confirming the diagnostic methodology — "check ECC before anything else" is now in our permanent playbook.

[Original thread on NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/turing-sm_75-fp16-tensor-core-corruption-root-cause-gddr6-defect/376994)
