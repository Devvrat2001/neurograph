#!/usr/bin/env python3
"""
examples/capi/example.py — P1-08 sample usage of neurograph_rt from Python.

Mirror of examples/capi/example.c, but driven through ctypes so customers
who deploy NeuroGraph behind Python services (Flask, FastAPI, Celery
workers) have a turnkey starting point. No third-party dependencies —
only the stdlib's `ctypes` and `json` so this runs on any Python 3.6+.

Usage:
    python example.py <bundle_dir> [path/to/neurograph_rt.dll]

The second argument is optional. If omitted, ctypes looks up the DLL
via the platform's normal search rules:
  - Windows: PATH and the directory containing this script.
  - Linux:   LD_LIBRARY_PATH and standard system dirs.
  - macOS:   DYLD_LIBRARY_PATH and standard system dirs.

Determinism contract from neurograph_rt.h:
  - Strings returned by ng_model_class_label / ng_model_version /
    ng_last_error point into library-owned storage. We copy them via
    ctypes' implicit char_p → bytes decode rather than hold the
    pointer past the next call.
  - The opaque ng_model* is held as c_void_p; never dereferenced.
  - Predict outputs go into caller-owned buffers we allocate as
    ctypes arrays sized by ng_model_num_classes (or 1 for regression).
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_int,
    c_void_p,
)
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Status codes — must match enum ng_status_e in neurograph_rt.h.
# ---------------------------------------------------------------------------
NG_OK = 0
NG_ERR_IO = 1
NG_ERR_CORRUPT = 2
NG_ERR_BAD_ARG = 3
NG_ERR_INTERNAL = 4

_STATUS_NAMES = {
    NG_OK: "NG_OK",
    NG_ERR_IO: "NG_ERR_IO",
    NG_ERR_CORRUPT: "NG_ERR_CORRUPT",
    NG_ERR_BAD_ARG: "NG_ERR_BAD_ARG",
    NG_ERR_INTERNAL: "NG_ERR_INTERNAL",
}


def _default_dll_name() -> str:
    """Platform-appropriate library name. Matches what CMake produces."""
    if sys.platform.startswith("win"):
        return "neurograph_rt.dll"
    if sys.platform == "darwin":
        return "libneurograph_rt.dylib"
    return "libneurograph_rt.so"


def _load_library(path: Optional[str]) -> ctypes.CDLL:
    """Load the DLL and pin every function's argtypes/restype.

    Skipping the argtypes step is the #1 source of ctypes crashes —
    without them, ctypes silently truncates pointers on 64-bit and
    you get garbage or segfaults that look unrelated. Worth the
    boilerplate.
    """
    if path is None:
        path = _default_dll_name()
    lib = ctypes.CDLL(path)

    # Lifecycle
    lib.ng_model_load.argtypes = [c_char_p, POINTER(c_void_p)]
    lib.ng_model_load.restype = c_int
    lib.ng_model_free.argtypes = [c_void_p]
    lib.ng_model_free.restype = None

    # Metadata
    lib.ng_model_num_features.argtypes = [c_void_p]
    lib.ng_model_num_features.restype = c_int
    lib.ng_model_num_classes.argtypes = [c_void_p]
    lib.ng_model_num_classes.restype = c_int
    lib.ng_model_class_label.argtypes = [c_void_p, c_int]
    lib.ng_model_class_label.restype = c_char_p
    lib.ng_model_version.argtypes = [c_void_p]
    lib.ng_model_version.restype = c_char_p

    # Inference
    lib.ng_predict_json.argtypes = [
        c_void_p, c_char_p, POINTER(c_double), POINTER(c_int),
    ]
    lib.ng_predict_json.restype = c_int
    lib.ng_predict_vec.argtypes = [
        c_void_p, POINTER(c_double), c_int,
        POINTER(c_double), POINTER(c_int),
    ]
    lib.ng_predict_vec.restype = c_int
    lib.ng_predict_batch_vec.argtypes = [
        c_void_p, POINTER(c_double), c_int, c_int,
        POINTER(c_double), POINTER(c_int),
    ]
    lib.ng_predict_batch_vec.restype = c_int

    # Diagnostics
    lib.ng_last_error.argtypes = []
    lib.ng_last_error.restype = c_char_p
    lib.ng_rt_version.argtypes = []
    lib.ng_rt_version.restype = c_char_p

    return lib


def _decode(s: Optional[bytes]) -> str:
    """ctypes hands us bytes (or None); decode for convenient printing."""
    if s is None:
        return ""
    return s.decode("utf-8", errors="replace")


def _die(lib: ctypes.CDLL, where: str, status: int) -> "NoReturn":
    name = _STATUS_NAMES.get(status, f"status={status}")
    last = _decode(lib.ng_last_error())
    print(f"example: {where}: {name} last_error={last}", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <bundle_dir> [path/to/neurograph_rt.dll]",
              file=sys.stderr)
        return 2

    bundle_dir = argv[1]
    dll_path: Optional[str] = argv[2] if len(argv) > 2 else None

    if not Path(bundle_dir).is_dir():
        print(f"example: bundle dir does not exist: {bundle_dir}",
              file=sys.stderr)
        return 1

    lib = _load_library(dll_path)
    print(f"neurograph_rt {_decode(lib.ng_rt_version())} loaded "
          f"from {dll_path or _default_dll_name()}")

    # Load the bundle. Note ng_model* is opaque — we hold it as a
    # c_void_p and never inspect its contents.
    model = c_void_p()
    bundle_bytes = os.fsencode(bundle_dir)  # avoid utf8 surprises on paths
    status = lib.ng_model_load(bundle_bytes, ctypes.byref(model))
    if status != NG_OK:
        _die(lib, "ng_model_load", status)

    try:
        n_features = lib.ng_model_num_features(model)
        n_classes = lib.ng_model_num_classes(model)
        version = _decode(lib.ng_model_version(model))
        print(f"loaded bundle: features={n_features} "
              f"classes={n_classes} version={version}")

        # Demo row matches example.c's hard-coded values so users can
        # diff the two files and see the FFI translation 1-to-1.
        row = {"f1": 1.0, "f2": 1.0, "f3": -1.0}
        row_json = json.dumps(row).encode("utf-8")

        if n_classes > 0:
            # Classification: probabilities buffer sized to num_classes,
            # plus a single int for the argmax class id.
            probs = (c_double * n_classes)()
            pred = c_int(-1)
            status = lib.ng_predict_json(
                model, row_json, probs, ctypes.byref(pred),
            )
            if status != NG_OK:
                _die(lib, "ng_predict_json", status)

            label = _decode(lib.ng_model_class_label(model, pred.value))
            print(f"predicted: id={pred.value} label={label}")
            for k in range(n_classes):
                print(f"  p[{k}]={probs[k]:.6f}")
        else:
            # Regression: probs[0] carries the scalar; no class id.
            value = c_double(0.0)
            status = lib.ng_predict_json(
                model, row_json, ctypes.byref(value), None,
            )
            if status != NG_OK:
                _die(lib, "ng_predict_json", status)
            print(f"predicted value: {value.value:.6f}")
    finally:
        # Quiesce any concurrent calls before this; here we're single-
        # threaded so it's a straight free. The header explicitly notes
        # that load/free are not thread-safe w.r.t. each other.
        lib.ng_model_free(model)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
