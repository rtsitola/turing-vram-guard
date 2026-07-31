/*
 * vmm_probe.cu — VMM-only test on a targeted region
 *
 * USAGE: vmm_probe.exe <device_id> <region_mb> [handle_count]
 * 
 * This does NOT do any cudaMalloc detection — Phase 1 is handled by
 * vram_mapper_detailed.exe. vmm_probe only does:
 *   1. Create N VMM handles at driver granularity
 *   2. Map them contiguously as a BATCH (> L2 cache)
 *   3. Test the whole batch with 32-pattern detection
 *   4. Report per-handle bad page counts
 *
 * Compile: via CMake (build.ps1)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#pragma comment(lib, "cuda.lib")

#define CUCHECK(call) do { \
    CUresult r = (call); \
    if (r != CUDA_SUCCESS) { \
        const char* name; cuGetErrorName(r, &name); \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, name); \
        exit(1); \
    } \
} while(0)

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════════════
// Detection kernels — ONE big batch > L2 to force eviction
// ═══════════════════════════════════════════════════

__global__ void fill_pattern(uint32_t* data, size_t n, uint32_t pattern, uint32_t salt) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
}

__global__ void check_pattern(const uint32_t* data, size_t n, uint32_t pattern,
                               uint32_t salt, int* bad_flags) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
        if (data[i] != expected)
            atomicOr(&bad_flags[i / 1024], 1);
    }
}

__global__ void fill_walking(uint32_t* data, size_t n, int bit) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) data[i] = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
}

__global__ void check_walking(const uint32_t* data, size_t n, int bit, int* bad_flags) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
        if (data[i] != expected)
            atomicOr(&bad_flags[i / 1024], 1);
    }
}

static const uint32_t PATTERNS[] = {
    0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
    0xCCCCCCCCu, 0x33333333u, 0xF0F0F0F0u, 0x0F0F0F0Fu,
};

// Detect on a device VA range. h_flags must be pre-allocated + zeroed.
// CRITICAL: h_flags is in HOST memory (cudaHostAllocMapped) — immune to GPU VRAM defects.
size_t detect_to_flags(uint32_t* data, size_t bytes, int* h_flags) {
    size_t count = bytes / sizeof(uint32_t);
    size_t pages = bytes / 4096;
    const int threads = 256;
    dim3 grid((unsigned)((count + threads - 1) / threads));

    int* d_flags;
    CK(cudaHostGetDevicePointer(&d_flags, h_flags, 0));

    for (int pass = 0; pass < 2; pass++) {
        for (int ip = 0; ip < 8; ip++) {
            uint32_t salt = (uint32_t)(pass * 131u + ip);
            fill_pattern<<<grid, threads>>>(data, count, PATTERNS[ip], salt);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            check_pattern<<<grid, threads>>>(data, count, PATTERNS[ip], salt, d_flags);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
        }
        for (int bit = 0; bit < 8; bit++) {
            fill_walking<<<grid, threads>>>(data, count, bit);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            check_walking<<<grid, threads>>>(data, count, bit, d_flags);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
        }
    }

    size_t bad = 0;
    for (size_t p = 0; p < pages; p++)
        if (h_flags[p]) bad++;
    return bad;
}

int main(int argc, char** argv) {
    int device_id      = (argc > 1) ? atoi(argv[1]) : 0;
    int region_mb      = (argc > 2) ? atoi(argv[2]) : 32;
    int handle_count   = (argc > 3) ? atoi(argv[3]) : 0;  // 0 = auto (region_mb / granularity)

    CK(cudaSetDevice(device_id));
    CUcontext cu_ctx;
    CUCHECK(cuCtxGetCurrent(&cu_ctx));

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, device_id));

    printf("=== VMM Probe — VMM-only targeted test ===\n");
    printf("Device %d: %s, L2: %d KB\n", device_id, prop.name, prop.l2CacheSize / 1024);
    printf("Region: %d MB\n\n", region_mb);

    // Query VMM granularity
    CUmemAllocationProp alloc_prop = {};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.location.id = device_id;

    size_t granularity = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &alloc_prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("VMM granularity: %zu MB\n", granularity / (1024*1024));

    if (handle_count == 0)
        handle_count = (int)(((size_t)region_mb * 1024 * 1024) / granularity);

    printf("Creating %d VMM handles (%d MB total)...\n", handle_count,
           (int)(handle_count * granularity / (1024*1024)));

    // Create handles
    CUmemGenericAllocationHandle* handles = (CUmemGenericAllocationHandle*)
        malloc(handle_count * sizeof(CUmemGenericAllocationHandle));

    int created = 0;
    for (int i = 0; i < handle_count; i++) {
        CUresult cr = cuMemCreate(&handles[i], granularity, &alloc_prop, 0);
        if (cr != CUDA_SUCCESS) {
            printf("cuMemCreate failed at handle %d\n", i);
            break;
        }
        created++;
    }
    printf("Created %d handles.\n", created);

    // Map as contiguous batch (L2-evicting: combined size > L2)
    size_t batch_bytes = (size_t)created * granularity;
    CUdeviceptr batch_va = 0;
    CUCHECK(cuMemAddressReserve(&batch_va, batch_bytes, 0, 0, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    for (int i = 0; i < created; i++)
        CUCHECK(cuMemMap(batch_va + i * granularity, granularity, 0, handles[i], 0));
    CUCHECK(cuMemSetAccess(batch_va, batch_bytes, &access, 1));

    // Test the whole batch — host-side flags, immune to GPU VRAM defects
    size_t batch_pages = batch_bytes / 4096;
    int* h_flags;
    CK(cudaHostAlloc(&h_flags, batch_pages * sizeof(int), cudaHostAllocMapped));
    memset(h_flags, 0, batch_pages * sizeof(int));

    printf("Testing %zu MB batch with 32-pattern detection...\n",
           batch_bytes / (1024*1024));

    size_t total_bad = detect_to_flags((uint32_t*)batch_va, batch_bytes, h_flags);
    printf("Total bad 4KB pages: %zu\n\n", total_bad);

    // Attribute to individual handles
    size_t pp_handle = granularity / 4096;
    int bad_handles = 0;
    for (int i = 0; i < created; i++) {
        size_t start = i * pp_handle;
        size_t end   = start + pp_handle;
        size_t h_bad = 0;
        for (size_t p = start; p < end; p++)
            if (h_flags[p]) h_bad++;
        if (h_bad > 0) {
            bad_handles++;
            printf("[BAD] VMM handle %d: %zu bad pages (%zu KB)\n",
                   i, h_bad, (h_bad * 4096) / 1024);
        }
    }

    printf("\n=== Result ===\n");
    printf("Handles: %d total, %d bad\n", created, bad_handles);
    if (total_bad > 0) {
        printf("VMM DETECTED defects. Strategy B is viable.\n");
    } else {
        printf("VMM found 0 defects.\n");
        printf("Either VMM doesn't expose the bad physical pages,\n");
        printf("or the batch size needs to be larger (increase region_mb).\n");
    }

    // Cleanup
    for (int i = 0; i < created; i++)
        CUCHECK(cuMemUnmap(batch_va + i * granularity, granularity));
    CUCHECK(cuMemAddressFree(batch_va, batch_bytes));
    CK(cudaFreeHost(h_flags));
    for (int i = 0; i < created; i++)
        CUCHECK(cuMemRelease(handles[i]));
    free(handles);
    return 0;
}
