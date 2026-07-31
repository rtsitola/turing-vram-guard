#pragma once
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>

// Run multi-pattern GPU-side detection on a device buffer.
//   d_ptr     : device pointer to the chunk to test
//   bytes     : size of the chunk in bytes
//   bad_pages : output — indices (0-based) of 4 KB pages that failed
// Returns true if at least one bad page was detected.
bool vg_detect_chunk(void* d_ptr, size_t bytes, std::vector<size_t>& bad_pages);

// Launch parameters (exposed so canary can reuse them).
constexpr int VG_DETECT_GRID  = 65535;  // maximum grid
constexpr int VG_DETECT_BLOCK = 256;
