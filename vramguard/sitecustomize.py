# sitecustomize.py — VRAM Guard auto-loader for Python CUDA apps
#
# CPython loads this file automatically at interpreter startup, BEFORE any
# user script imports. This is the official Python customization mechanism:
#   https://docs.python.org/3/library/site.html
#
# PLACEMENT: copy to one of these locations (pick one):
#   1. <comfyui_venv>/Lib/site-packages/sitecustomize.py   (venv only)
#   2. <comfyui_venv>/Lib/sitecustomize.py                 (all Python in venv)
#   3. %USERPROFILE%/Python/sitecustomize.py               (all Python for user)
#
# ENVIRONMENT VARIABLES (optional):
#   VRAMGUARD_DLL     — path to vramguard.dll (default: vramguard.dll in PATH)
#   VRAMGUARD_DEVICE  — CUDA device ID (default: 0)
#   VRAMGUARD_SKIP    — set to "1" to bypass guard entirely
#
# RETURNS: exits with code 1 if guard fails and VRAMGUARD_SKIP != 1.
#          Normal import continues if guard succeeds or VRAMGUARD_SKIP=1.

import ctypes
import os
import sys


def _install_vram_guard():
    """Called at Python startup. Exits cleanly if guard is not needed."""

    # ── Check skip flag ──
    if os.environ.get("VRAMGUARD_SKIP") == "1":
        print("[vramguard] VRAMGUARD_SKIP=1 — bypassing guard", file=sys.stderr)
        return

    # ── Resolve DLL path ──
    dll_path = os.environ.get("VRAMGUARD_DLL", "vramguard.dll")
    device   = int(os.environ.get("VRAMGUARD_DEVICE", "0"))

    try:
        dll = ctypes.CDLL(dll_path)
    except OSError as e:
        print(f"[vramguard] Cannot load {dll_path}: {e}", file=sys.stderr)
        print("[vramguard] Set VRAMGUARD_DLL=path or VRAMGUARD_SKIP=1", file=sys.stderr)
        sys.exit(1)

    # ── Call vg_install ──
    dll.vg_install.argtypes = [ctypes.c_int]
    dll.vg_install.restype  = ctypes.c_int

    rc = dll.vg_install(device)

    if rc < 0:
        print(f"[vramguard] FATAL: vg_install({device}) returned {rc}", file=sys.stderr)
        sys.exit(1)
    elif rc == 0:
        print(f"[vramguard] Guard ACTIVE on device {device} — bad pages reserved", file=sys.stderr)
    else:  # rc == 1
        print(f"[vramguard] No bad pages detected on device {device} — guard not needed", file=sys.stderr)

    # Note: vramguard.dll allocates guard chunks and starts its canary thread.
    # The guard persists for the process lifetime via static pointers in the DLL.
    # PyTorch's CUDACachingAllocator will see ~22 GB free and work normally.


# ── Run immediately (before any other imports) ──
_install_vram_guard()
