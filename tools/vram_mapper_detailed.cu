// vram_mapper_detailed.cu
// Mappe la VRAM chunk par chunk pour isoler les zones corrompues.
// Compile (Windows, CUDA 12+) :
//   nvcc -O2 -o vram_mapper_detailed.exe vram_mapper_detailed.cu
// Run (Quadro = device 1 chez toi) :
//   set CUDA_VISIBLE_DEVICES=1
//   vram_mapper_detailed.exe
// Options :
//   vram_mapper_detailed.exe [chunk_MB] [passes] [device_id] [start_chunk]
//   défaut : 64 MB, 3 passes, device 0 (donc mets CUDA_VISIBLE_DEVICES=1)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// ---------- kernels de test ----------

__global__ void fill_pattern(uint32_t* data, size_t n, uint32_t pattern, uint32_t salt) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        // pattern dépend de l'index pour détecter addressing errors
        data[i] = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
    }
}

__global__ void check_pattern(const uint32_t* data, size_t n, uint32_t pattern,
                              uint32_t salt, unsigned long long* err_count,
                              size_t* first_bad_idx) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = pattern ^ (uint32_t)i ^ (salt * 0x9E3779B9u);
        if (data[i] != expected) {
            atomicAdd(err_count, 1ULL);
            // garde le premier index fautif (approx)
            atomicMin((unsigned long long*)first_bad_idx, (unsigned long long)i);
        }
    }
}

__global__ void fill_walking_bit(uint32_t* data, size_t n, int bit) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
    }
}

__global__ void check_walking_bit(const uint32_t* data, size_t n, int bit,
                                  unsigned long long* err_count) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t expected = (1u << (bit & 31)) ^ (uint32_t)(i * 0x85EBCA6Bu);
        if (data[i] != expected) atomicAdd(err_count, 1ULL);
    }
}

// ---------- helpers ----------

struct ChunkResult {
    int index;
    size_t offset_bytes;
    size_t size_bytes;
    unsigned long long errors;
    size_t first_bad_word;  // index 32-bit dans le chunk
    bool bad;
};

static void launch_1d(size_t n, int threads = 256) {
    // utilisé implicitement via grid calculé dans les appels
    (void)n; (void)threads;
}

static dim3 grid_for(size_t n, int threads = 256) {
    size_t blocks = (n + threads - 1) / threads;
    if (blocks > 0x7fffffffULL) blocks = 0x7fffffffULL;
    return dim3((unsigned)blocks);
}

int main(int argc, char** argv) {
    int chunk_mb   = (argc > 1) ? atoi(argv[1]) : 64;
    int passes     = (argc > 2) ? atoi(argv[2]) : 3;
    int device_id  = (argc > 3) ? atoi(argv[3]) : 0;
    int start_c    = (argc > 4) ? atoi(argv[4]) : 0;

    if (chunk_mb < 1) chunk_mb = 64;
    if (passes < 1) passes = 1;

    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    size_t free_b = 0, total_b = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));

    printf("=== VRAM Mapper Detailed ===\n");
    printf("Device %d: %s (sm_%d%d)\n", device_id, prop.name, prop.major, prop.minor);
    printf("Total VRAM: %.2f GB | Free: %.2f GB\n",
           total_b / (1024.0 * 1024 * 1024), free_b / (1024.0 * 1024 * 1024));
    printf("Chunk size: %d MB | Passes: %d\n\n", chunk_mb, passes);

    const size_t chunk_bytes = (size_t)chunk_mb * 1024ULL * 1024ULL;
    const size_t chunk_words = chunk_bytes / sizeof(uint32_t);

    // Réserver un peu de marge pour le driver
    size_t reserve = 256ULL * 1024 * 1024; // 256 MB
    if (free_b <= reserve + chunk_bytes) {
        fprintf(stderr, "Pas assez de VRAM libre.\n");
        return 1;
    }
    size_t usable = free_b - reserve;
    int n_chunks = (int)(usable / chunk_bytes);
    if (n_chunks < 1) {
        fprintf(stderr, "Aucun chunk allouable.\n");
        return 1;
    }

    printf("Allocating %d chunks of %d MB (%.2f GB total test)...\n",
           n_chunks, chunk_mb, (n_chunks * chunk_bytes) / (1024.0 * 1024 * 1024));

    std::vector<uint32_t*> ptrs(n_chunks, nullptr);
    std::vector<ChunkResult> results(n_chunks);

    // 1) Allouer TOUT d'abord (évite le faux négatif d'un seul gros bloc réutilisé)
    for (int i = 0; i < n_chunks; ++i) {
        cudaError_t e = cudaMalloc(&ptrs[i], chunk_bytes);
        if (e != cudaSuccess) {
            printf("cudaMalloc failed at chunk %d: %s — stop allocation\n",
                   i, cudaGetErrorString(e));
            n_chunks = i;
            ptrs.resize(n_chunks);
            results.resize(n_chunks);
            break;
        }
        results[i].index = i;
        results[i].offset_bytes = (size_t)i * chunk_bytes; // offset logique de test
        results[i].size_bytes = chunk_bytes;
        results[i].errors = 0;
        results[i].first_bad_word = (size_t)-1;
        results[i].bad = false;
    }

    printf("Allocated %d chunks.\n\n", n_chunks);

    unsigned long long* d_err = nullptr;
    size_t* d_first = nullptr;
    CHECK_CUDA(cudaMalloc(&d_err, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_first, sizeof(size_t)));

    const uint32_t patterns[] = {
        0x00000000u,
        0xFFFFFFFFu,
        0xAAAAAAAAu,
        0x55555555u,
        0xCCCCCCCCu,
        0x33333333u,
        0xF0F0F0F0u,
        0x0F0F0F0Fu,
    };
    const int n_patterns = (int)(sizeof(patterns) / sizeof(patterns[0]));

    // 2) Tester chaque chunk
    for (int c = start_c; c < n_chunks; ++c) {
        unsigned long long chunk_errors = 0;
        size_t first_bad = (size_t)-1;

        for (int p = 0; p < passes; ++p) {
            for (int ip = 0; ip < n_patterns; ++ip) {
                uint32_t pat = patterns[ip];
                uint32_t salt = (uint32_t)(c * 131u + p * 17u + ip);

                CHECK_CUDA(cudaMemset(d_err, 0, sizeof(unsigned long long)));
                size_t init_first = (size_t)-1;
                CHECK_CUDA(cudaMemcpy(d_first, &init_first, sizeof(size_t), cudaMemcpyHostToDevice));

                fill_pattern<<<grid_for(chunk_words), 256>>>(ptrs[c], chunk_words, pat, salt);
                CHECK_CUDA(cudaDeviceSynchronize());

                check_pattern<<<grid_for(chunk_words), 256>>>(
                    ptrs[c], chunk_words, pat, salt, d_err, d_first);
                CHECK_CUDA(cudaDeviceSynchronize());

                unsigned long long e = 0;
                size_t fb = 0;
                CHECK_CUDA(cudaMemcpy(&e, d_err, sizeof(e), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&fb, d_first, sizeof(fb), cudaMemcpyDeviceToHost));
                chunk_errors += e;
                if (e && fb < first_bad) first_bad = fb;
            }

            // walking bit (8 bits)
            for (int bit = 0; bit < 8; ++bit) {
                CHECK_CUDA(cudaMemset(d_err, 0, sizeof(unsigned long long)));
                fill_walking_bit<<<grid_for(chunk_words), 256>>>(ptrs[c], chunk_words, bit);
                CHECK_CUDA(cudaDeviceSynchronize());
                check_walking_bit<<<grid_for(chunk_words), 256>>>(ptrs[c], chunk_words, bit, d_err);
                CHECK_CUDA(cudaDeviceSynchronize());
                unsigned long long e = 0;
                CHECK_CUDA(cudaMemcpy(&e, d_err, sizeof(e), cudaMemcpyDeviceToHost));
                chunk_errors += e;
            }
        }

        results[c].errors = chunk_errors;
        results[c].first_bad_word = first_bad;
        results[c].bad = (chunk_errors > 0);

        double off_gb = results[c].offset_bytes / (1024.0 * 1024 * 1024);
        if (results[c].bad) {
            printf("[BAD ] chunk %4d | offset ~%7.3f GB | errors=%llu | first_word=%zu\n",
                   c, off_gb, (unsigned long long)chunk_errors, first_bad);
        } else {
            printf("[OK  ] chunk %4d | offset ~%7.3f GB\n", c, off_gb);
        }
        fflush(stdout);
    }

    // 3) Résumé
    printf("\n======== SUMMARY ========\n");
    int n_bad = 0;
    unsigned long long total_err = 0;
    for (int c = 0; c < n_chunks; ++c) {
        if (results[c].bad) {
            n_bad++;
            total_err += results[c].errors;
            double off_gb = results[c].offset_bytes / (1024.0 * 1024 * 1024);
            double end_gb = (results[c].offset_bytes + results[c].size_bytes) /
                            (1024.0 * 1024 * 1024);
            printf("  BAD chunk #%d : [%.3f – %.3f] GB  errors=%llu\n",
                   c, off_gb, end_gb, (unsigned long long)results[c].errors);
        }
    }
    printf("Chunks tested : %d\n", n_chunks);
    printf("Bad chunks    : %d\n", n_bad);
    printf("Total errors  : %llu\n", (unsigned long long)total_err);
    printf("Usable (clean): %.2f GB (approx)\n",
           ((n_chunks - n_bad) * chunk_bytes) / (1024.0 * 1024 * 1024));

    if (n_bad == 0) {
        printf("\nAucune corruption détectée sur cette passe.\n");
    } else {
        printf("\nListe d'index bad (pour blacklist init) : ");
        bool first = true;
        for (int c = 0; c < n_chunks; ++c) {
            if (results[c].bad) {
                if (!first) printf(",");
                printf("%d", c);
                first = false;
            }
        }
        printf("\n");
    }

    // 4) Cleanup
    for (int i = 0; i < n_chunks; ++i) {
        if (ptrs[i]) cudaFree(ptrs[i]);
    }
    cudaFree(d_err);
    cudaFree(d_first);

    printf("\nDone.\n");
    return (n_bad > 0) ? 2 : 0;
}