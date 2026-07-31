#pragma once

#include <stddef.h>  // for size_t

#ifdef VRAMGUARD_EXPORTS
  #define VRAMGUARD_API __declspec(dllexport)
#else
  #define VRAMGUARD_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Install VRAM guard on the specified CUDA device.
// Must be called BEFORE any application CUDA allocations on that device.
//   device : CUDA device ordinal (usually 0)
// Returns:
//    0 : guard installed, bad pages were detected and reserved
//   -1 : fatal error (no CUDA device, allocation failure, etc.)
//    1 : no bad pages detected (GPU appears healthy; guard not needed)
VRAMGUARD_API int vg_install(int device);

// Returns 1 if guard is currently active, 0 otherwise.
VRAMGUARD_API int vg_is_active(void);

// Returns the total number of bytes reserved by guard allocations.
VRAMGUARD_API size_t vg_guarded_bytes(void);

// Returns the number of bytes identified as defective (the true bad region).
VRAMGUARD_API size_t vg_bad_bytes(void);

// Force a re-scan of all guard chunks. Returns 0 if guard still valid,
// -1 if WDDM migrated the guard (defect pages may be re-exposed).
VRAMGUARD_API int vg_verify(void);

#ifdef __cplusplus
}
#endif
