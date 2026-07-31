/*
 * detection.cu — GPU-side VRAM defect detection kernels.
 *
 * INSPIRED BY vram_mapper_detailed (validated on Quadro RTX 6000):
 *   - 8 static patterns per pass (0x00, 0xFF, 0xAA, 0x55, 0xCC, 0x33, 0xF0, 0x0F)
 *   - Address-dependent salt (catches addressing errors)
 *   - Walking-1 bit test (bits 0-7) — catches stuck-at-0/1 per-bit cells
 *   - 2 passes × (8 patterns + 8 walking bits) = 32 checks per chunk
 *   - Exact grid coverage (one thread per word, no stride gaps)
 *
 * CRITICAL: d_flags allocated via cudaHostAlloc (host RAM), NOT in GPU VRAM.
 *           If d_flags lands in defective VRAM, stuck-at-0 cells cause
 *           false negatives.
 */
#include "detection.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>

// ══════════════════════════════════════════════════════════════════
// GPU KERNELS — matching vram_mapper_detailed approach
// ══════════════════════════════════════════════════════════════════

// Pattern fill with address-dependent salt (catches addressing errors)
__global__ void vg_fill_pattern(uint32_t* data, size_t n, uint32_t pattern, uint32_t salt) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
}

// Pattern check: flags bad 4 KB pages via atomicOr
// FIXED: i / 1024 for 4 KB (4096 bytes / 4 bytes per uint32)
__global__ void vg_check_pattern(const uint32_t* data, size_t n, uint32_t pattern,
                                  uint32_t salt, int* bad_flags) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
        if (data[i] != expected)
            atomicOr(&bad_flags[i / 1024], 1);
    }
}

// Walking-1 fill: writes a walking-1 bit pattern with address-dependent mix
__global__ void vg_fill_walking(uint32_t* data, size_t n, int bit) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
}

// Walking-1 check
__global__ void vg_check_walking(const uint32_t* data, size_t n, int bit, int* bad_flags) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
        if (data[i] != expected)
            atomicOr(&bad_flags[i / 1024], 1);
    }
}

// Helper: exact grid (one thread per element)
static inline dim3 vg_grid(size_t n, int threads) {
    size_t blocks = (n + threads - 1) / threads;
    if (blocks > 0x7FFFFFFFULL) {
        fprintf(stderr, "[vramguard] WARNING: vg_grid clamping %zu blocks to 0x7FFFFFFF "
                "— %zu elements may be unscanned\n", blocks, n - 0x7FFFFFFFULL * threads);
        blocks = 0x7FFFFFFFULL;
    }
    return dim3((unsigned int)blocks);
}

// ══════════════════════════════════════════════════════════════════
// HOST-SIDE ORCHESTRATOR
// ══════════════════════════════════════════════════════════════════

// 8 complementary patterns — catch stuck-at-0/1 in any bit position
static const uint32_t PATTERNS[] = {
    0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
    0xCCCCCCCCu, 0x33333333u, 0xF0F0F0F0u, 0x0F0F0F0Fu,
};
static const int N_PATTERNS = 8;
static const int N_PASSES   = 2;  // 2 passes through all patterns

bool vg_detect_chunk(void* d_ptr, size_t bytes, std::vector<size_t>& bad_pages) {
    bad_pages.clear();
    if (!d_ptr || bytes == 0) return false;

    size_t count  = bytes / sizeof(uint32_t);
    size_t pages  = bytes / 4096;
    uint32_t* buf = static_cast<uint32_t*>(d_ptr);
    const int threads = 256;

    // ── Allocate bad-page flags in HOST RAM ──
    int* h_flags_host = nullptr;
    int* d_flags_dev  = nullptr;

    cudaError_t err = cudaHostAlloc(&h_flags_host, pages * sizeof(int),
                                     cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[vramguard] cudaHostAlloc failed: %s — falling back to GPU\n",
                cudaGetErrorString(err));
        if (cudaMalloc(&d_flags_dev, pages * sizeof(int)) != cudaSuccess)
            return false;
        // Zero flags in GPU memory (not via memset on NULL h_flags_host)
        cudaMemsetAsync(d_flags_dev, 0, pages * sizeof(int), 0);
        cudaDeviceSynchronize();
    } else {
        cudaHostGetDevicePointer(&d_flags_dev, h_flags_host, 0);
        memset(h_flags_host, 0, pages * sizeof(int));
    }

    cudaError_t kern_err;

    // ── Main detection loop ──
    for (int pass = 0; pass < N_PASSES; pass++) {
        // 8 static patterns
        for (int ip = 0; ip < N_PATTERNS; ip++) {
            uint32_t salt = (uint32_t)(pass * 131u + ip);

            vg_fill_pattern<<<vg_grid(count, threads), threads>>>(
                buf, count, PATTERNS[ip], salt);
            cudaDeviceSynchronize();
            kern_err = cudaGetLastError();
            if (kern_err != cudaSuccess) {
                fprintf(stderr, "[vramguard] fill_pattern pass=%d pat=%d: %s — keeping chunk\n",
                        pass, ip, cudaGetErrorString(kern_err));
                goto keep_chunk;
            }

            vg_check_pattern<<<vg_grid(count, threads), threads>>>(
                buf, count, PATTERNS[ip], salt, d_flags_dev);
            cudaDeviceSynchronize();
            kern_err = cudaGetLastError();
            if (kern_err != cudaSuccess) {
                fprintf(stderr, "[vramguard] check_pattern pass=%d pat=%d: %s — keeping chunk\n",
                        pass, ip, cudaGetErrorString(kern_err));
                goto keep_chunk;
            }
        }

        // Walking-1 bits (0-7): catches per-bit stuck-at-0/1
        for (int bit = 0; bit < 8; bit++) {
            vg_fill_walking<<<vg_grid(count, threads), threads>>>(
                buf, count, bit);
            cudaDeviceSynchronize();
            kern_err = cudaGetLastError();
            if (kern_err != cudaSuccess) {
                fprintf(stderr, "[vramguard] fill_walking bit=%d: %s — keeping chunk\n",
                        bit, cudaGetErrorString(kern_err));
                goto keep_chunk;
            }

            vg_check_walking<<<vg_grid(count, threads), threads>>>(
                buf, count, bit, d_flags_dev);
            cudaDeviceSynchronize();
            kern_err = cudaGetLastError();
            if (kern_err != cudaSuccess) {
                fprintf(stderr, "[vramguard] check_walking bit=%d: %s — keeping chunk\n",
                        bit, cudaGetErrorString(kern_err));
                goto keep_chunk;
            }
        }
    }

    // ── Compile bad page list from host-mapped flags ──
    size_t bad_count = 0;
    for (size_t p = 0; p < pages; p++) {
        if (h_flags_host[p]) {
            bad_pages.push_back(p);
            bad_count++;
        }
    }

    // DEBUG: always print detection summary for first chunk (thread-safe)
    static std::atomic<int> debug_once{1};
    if (debug_once.exchange(0) == 1) {
        fprintf(stderr, "[vramguard] DEBUG detect: %zu pages, %zu bad, h_flags[0..3]=",
                pages, bad_count);
        for (int i = 0; i < 4 && i < (int)pages; i++)
            fprintf(stderr, "%d ", h_flags_host[i]);
        fprintf(stderr, "\n");
        debug_once = 0;
    }

    cudaFreeHost(h_flags_host);
    return !bad_pages.empty();

keep_chunk:
    // Kernel error: mark all pages bad so chunk is kept
    for (size_t p = 0; p < pages; p++) bad_pages.push_back(p);
    if (h_flags_host) cudaFreeHost(h_flags_host);
    else if (d_flags_dev) cudaFree(d_flags_dev);
    return true;
}
