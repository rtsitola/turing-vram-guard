/*
 * vramguard.cpp — Main DLL logic.
 *
 * Strategy A (chunked cudaMalloc keep-bad):
 *   1. Allocate VRAM in 128 MB chunks until OOM.
 *   2. Test each chunk in-place with GPU-side detection kernels.
 *   3. Free good chunks. Keep bad chunks (leak intentionally).
 *   4. Start canary thread to detect WDDM eviction.
 *
 * No DLL injection needed for Python targets (use sitecustomize.py).
 * For non-Python targets, inject with a CREATE_SUSPENDED launcher.
 */
#include "vramguard.h"
#include "detection.h"

#include <cuda_runtime.h>
#include <windows.h>
#include <process.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <mutex>

// ══════════════════════════════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════════════════════════════

static constexpr size_t CHUNK_SIZE          = 32ULL * 1024 * 1024;   // 32 MB
static constexpr size_t HEADROOM            = 512ULL * 1024 * 1024;  // 512 MB for WDDM
static constexpr size_t PAGE_SIZE           = 4096;
static constexpr int    CANARY_INTERVAL_SEC = 30;

// ══════════════════════════════════════════════════════════════════
// Guard chunk tracking
// ══════════════════════════════════════════════════════════════════

struct GuardChunk {
    void*  ptr;         // cudaMalloc'd pointer (never freed)
    size_t size;        // bytes
    size_t num_pages;   // 4 KB pages
    int*   h_bad_flags; // host copy: 1=bad page, 0=good page (within this chunk)
};

// ══════════════════════════════════════════════════════════════════
// Global state (protected by g_mutex)
// ══════════════════════════════════════════════════════════════════

static std::once_flag g_init_flag;
static std::mutex     g_mutex;
static int            g_device       = 0;
static bool           g_installed    = false;
static bool           g_no_defect    = false;   // true when scan found zero bad pages
static size_t         g_bad_bytes    = 0;
static size_t         g_guarded_bytes = 0;
static std::vector<GuardChunk> g_guards;

// Canary
static HANDLE         g_canary_thread = nullptr;
static volatile long  g_canary_stop   = 0;
static volatile long  g_canary_failed = 0;       // set to 1 if WDDM migration detected

// ══════════════════════════════════════════════════════════════════
// Forward declarations
// ══════════════════════════════════════════════════════════════════

static unsigned __stdcall canary_thread_proc(void* arg);
static void start_canary();
static void stop_canary();

// ══════════════════════════════════════════════════════════════════
// DLL Entry
// ══════════════════════════════════════════════════════════════════

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved) {
    (void)hModule; (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
    } else if (reason == DLL_PROCESS_DETACH) {
        stop_canary();
        // Guards are intentionally leaked — process is exiting anyway.
    }
    return TRUE;
}

// ══════════════════════════════════════════════════════════════════
// Public API
// ══════════════════════════════════════════════════════════════════

VRAMGUARD_API int vg_install(int device) {
    std::call_once(g_init_flag, [&]() {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_device = device;

        // ── Step 1: Initialize CUDA device ──
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            fprintf(stderr, "[vramguard] FATAL: no CUDA devices\n");
            return;
        }
        if (device >= count) {
            fprintf(stderr, "[vramguard] FATAL: device %d out of range (0-%d)\n",
                    device, count - 1);
            return;
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        fprintf(stderr, "[vramguard] Device %d: %s (%zu MB)\n",
                device, prop.name, prop.totalGlobalMem / (1024 * 1024));

        if (cudaSetDevice(device) != cudaSuccess) {
            fprintf(stderr, "[vramguard] FATAL: cudaSetDevice(%d) failed\n", device);
            return;
        }
        // cudaSetDevice creates/retains the primary context — PyTorch reuses it.

        // ── Step 2: Allocate chunks until OOM ──
        struct ChunkInfo { void* ptr; size_t size; };
        std::vector<ChunkInfo> chunks;
        size_t total_allocated = 0;

        while (true) {
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            if (free_mem < CHUNK_SIZE + HEADROOM) {
                fprintf(stderr, "[vramguard] Stopping alloc: free=%zu MB < needed=%zu MB\n",
                        free_mem / (1024 * 1024),
                        (CHUNK_SIZE + HEADROOM) / (1024 * 1024));
                break;
            }
            void* p = nullptr;
            cudaError_t err = cudaMalloc(&p, CHUNK_SIZE);
            if (err != cudaSuccess) {
                fprintf(stderr, "[vramguard] cudaMalloc chunk %zu failed: %s\n",
                        chunks.size(), cudaGetErrorString(err));
                break;
            }
            chunks.push_back({p, CHUNK_SIZE});
            total_allocated += CHUNK_SIZE;
        }

        fprintf(stderr, "[vramguard] Allocated %zu chunks (%zu MB total)\n",
                chunks.size(), total_allocated / (1024 * 1024));

        if (chunks.empty()) {
            fprintf(stderr, "[vramguard] WARNING: zero chunks — GPU fully reserved?\n");
            return;
        }

        // ── Step 3: Detect bad chunks ──
        size_t total_bad_pages = 0;

        for (size_t i = 0; i < chunks.size(); i++) {
            std::vector<size_t> bad_pages;
            bool has_bad = vg_detect_chunk(chunks[i].ptr, chunks[i].size, bad_pages);

            if (has_bad) {
                fprintf(stderr, "[vramguard] Chunk %zu: %zu bad 4KB pages (%zu MB) — KEEPING\n",
                        i, bad_pages.size(), (bad_pages.size() * PAGE_SIZE) / (1024 * 1024));
                total_bad_pages += bad_pages.size();

                size_t num_pages = chunks[i].size / PAGE_SIZE;
                int* h_flags = new int[num_pages]();
                for (size_t bp : bad_pages) h_flags[bp] = 1;
                g_guards.push_back({chunks[i].ptr, chunks[i].size, num_pages, h_flags});
            } else {
                fprintf(stderr, "[vramguard] Chunk %zu: clean — freeing\n", i);
                cudaFree(chunks[i].ptr);
            }
        }

        // ── Step 4: Finalize ──
        if (g_guards.empty()) {
            fprintf(stderr, "[vramguard] No bad pages detected. GPU appears healthy.\n");
            g_no_defect = true;
            g_installed = true;
            return;
        }

        g_bad_bytes     = total_bad_pages * PAGE_SIZE;
        g_guarded_bytes = g_guards.size() * CHUNK_SIZE;
        g_installed     = true;

        fprintf(stderr, "[vramguard] === SUMMARY ===\n");
        fprintf(stderr, "[vramguard] Bad pages:      %zu (%zu MB)\n",
                total_bad_pages, g_bad_bytes / (1024 * 1024));
        fprintf(stderr, "[vramguard] Guarded VRAM:   %zu MB (%zu chunks)\n",
                g_guarded_bytes / (1024 * 1024), g_guards.size());

        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "[vramguard] Usable by app:  ~%zu MB\n", free_mem / (1024 * 1024));

        start_canary();
    });

    // Read final state (call_once may have run on a previous invocation)
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_installed) return -1;
    if (g_no_defect)  return 1;
    return 0;
}

VRAMGUARD_API int vg_is_active(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_installed && !g_no_defect ? 1 : 0;
}

VRAMGUARD_API size_t vg_guarded_bytes(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_guarded_bytes;
}

VRAMGUARD_API size_t vg_bad_bytes(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_bad_bytes;
}

VRAMGUARD_API int vg_verify(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_installed || g_no_defect) return 0;

    // Must set device — caller may be on any thread.
    cudaError_t dev_err = cudaSetDevice(g_device);
    if (dev_err != cudaSuccess) {
        fprintf(stderr, "[vramguard] vg_verify: cudaSetDevice(%d) failed: %s\n",
                g_device, cudaGetErrorString(dev_err));
        return -1;
    }

    fprintf(stderr, "[vramguard] Manual verify: checking %zu guard chunks...\n", g_guards.size());

    for (size_t i = 0; i < g_guards.size(); i++) {
        auto& gc = g_guards[i];
        std::vector<size_t> bad_pages;
        bool has_bad = vg_detect_chunk(gc.ptr, gc.size, bad_pages);

        if (!has_bad) {
            fprintf(stderr, "[vramguard] *** WDDM MIGRATION DETECTED ***\n");
            fprintf(stderr, "[vramguard] Guard chunk %zu reads clean — physical pages were remapped!\n", i);
            fprintf(stderr, "[vramguard] Bad pages may have re-entered circulation. Re-scan recommended.\n");
            return -1;
        }

        // Verify bad page count didn't grow significantly
        size_t prev_bad = 0;
        for (size_t p = 0; p < gc.num_pages; p++)
            if (gc.h_bad_flags[p]) prev_bad++;

        if (bad_pages.size() > prev_bad * 2) {
            fprintf(stderr, "[vramguard] *** DEFECT GROWING ***\n");
            fprintf(stderr, "[vramguard] Chunk %zu: was %zu bad pages, now %zu\n",
                    i, prev_bad, bad_pages.size());
        }

        fprintf(stderr, "[vramguard] Chunk %zu: %zu bad pages (was %zu) — OK\n",
                i, bad_pages.size(), prev_bad);
    }

    fprintf(stderr, "[vramguard] Verify passed — guard still holds bad pages.\n");
    return 0;
}

// ══════════════════════════════════════════════════════════════════
// Canary thread — periodically re-checks guard chunks
// ══════════════════════════════════════════════════════════════════

static unsigned __stdcall canary_thread_proc(void* arg) {
    (void)arg;

    // The canary thread is a new OS thread — it must set the CUDA device
    // before making any CUDA calls.
    cudaError_t dev_err = cudaSetDevice(g_device);
    if (dev_err != cudaSuccess) {
        fprintf(stderr, "[vramguard] Canary: cudaSetDevice(%d) failed: %s\n",
                g_device, cudaGetErrorString(dev_err));
        return 1;
    }

    fprintf(stderr, "[vramguard] Canary thread started (interval=%d sec, device=%d)\n",
            CANARY_INTERVAL_SEC, g_device);

    while (InterlockedCompareExchange(&g_canary_stop, 0, 0) == 0) {
        Sleep(CANARY_INTERVAL_SEC * 1000);

        if (InterlockedCompareExchange(&g_canary_stop, 0, 0) != 0)
            break;

        // Lock and verify
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if (!g_installed || g_no_defect || g_guards.empty())
                continue;

            for (size_t i = 0; i < g_guards.size(); i++) {
                auto& gc = g_guards[i];
                std::vector<size_t> current_bad;
                bool has_bad = vg_detect_chunk(gc.ptr, gc.size, current_bad);

                if (!has_bad) {
                    fprintf(stderr, "[vramguard] CANARY: WDDM EVICTION DETECTED on chunk %zu!\n", i);
                    fprintf(stderr, "[vramguard] Guard chunk reads clean — physical pages migrated.\n");
                    InterlockedExchange(&g_canary_failed, 1);
                    // Don't break — continue checking remaining chunks
                }
            }
        }
    }

    fprintf(stderr, "[vramguard] Canary thread exiting.\n");
    return 0;
}

static void start_canary() {
    if (g_canary_thread) return;
    InterlockedExchange(&g_canary_stop, 0);
    InterlockedExchange(&g_canary_failed, 0);
    g_canary_thread = (HANDLE)_beginthreadex(
        nullptr, 0, canary_thread_proc, nullptr, 0, nullptr);
}

static void stop_canary() {
    InterlockedExchange(&g_canary_stop, 1);
    if (g_canary_thread) {
        WaitForSingleObject(g_canary_thread, 5000);
        CloseHandle(g_canary_thread);
        g_canary_thread = nullptr;
    }
}
