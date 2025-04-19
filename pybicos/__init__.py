"""
pybicos: Python bindings for libBICOS using ctypes
"""

import ctypes
import os
import platform
import numpy as np
from enum import Enum

# Load the shared library
def _load_library():
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    if platform.system() == 'Windows':
        lib_name = 'pybicos_c.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'pybicos_c.dylib'
    else:
        lib_name = 'pybicos_c.so'
    lib_path = os.path.join(lib_dir, lib_name)
    if not os.path.exists(lib_path):
        raise ImportError(f"Could not find shared library at {lib_path}")
    return ctypes.CDLL(lib_path)

_lib = _load_library()

# Define enums
class TransformMode(Enum):
    LIMITED = 0
    FULL = 1

class Precision(Enum):
    SINGLE = 0
    DOUBLE = 1

class VariantType(Enum):
    NO_DUPLICATES = 0
    CONSISTENCY = 1

# Define C struct wrappers
class BicosConfig(ctypes.Structure):
    _fields_ = [
        ("nxcorr_threshold", ctypes.c_float),
        ("subpixel_step", ctypes.c_float),
        ("min_variance", ctypes.c_float),
        ("mode", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("variant_type", ctypes.c_int),
        ("max_lr_diff", ctypes.c_int),
        ("no_dupes", ctypes.c_int)
    ]

class BicosResult(ctypes.Structure):
    _fields_ = [
        ("disparity_data", ctypes.c_void_p),
        ("disparity_rows", ctypes.c_int),
        ("disparity_cols", ctypes.c_int),
        ("disparity_type", ctypes.c_int),
        ("corrmap_data", ctypes.c_void_p),
        ("corrmap_rows", ctypes.c_int),
        ("corrmap_cols", ctypes.c_int),
        ("corrmap_type", ctypes.c_int)
    ]

# Function prototypes
_lib.BICOS_CreateDefaultConfig.restype  = ctypes.POINTER(BicosConfig)
_lib.BICOS_FreeConfig.argtypes         = [ctypes.POINTER(BicosConfig)]
_lib.BICOS_FreeResult.argtypes         = [ctypes.POINTER(BicosResult)]
_lib.BICOS_Match.argtypes              = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(BicosConfig)
]
_lib.BICOS_Match.restype               = ctypes.POINTER(BicosResult)
_lib.BICOS_InvalidDisparityFloat.restype = ctypes.c_float
_lib.BICOS_InvalidDisparityInt16.restype = ctypes.c_int16

# OpenCV type constants
CV_8U  = 0
CV_16U = 2
CV_16S = 3
CV_32F = 5
CV_64F = 6

def _get_cv_type(dtype):
    if dtype == np.uint8:
        return CV_8U
    elif dtype == np.uint16:
        return CV_16U
    elif dtype == np.int16:
        return CV_16S
    elif dtype == np.float32:
        return CV_32F
    elif dtype == np.float64:
        return CV_64F
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

def _get_np_dtype(cv_type):
    if cv_type == CV_16S or cv_type == (CV_16S | (1 << 3)):
        return np.int16
    elif cv_type == CV_32F or cv_type == (CV_32F | (1 << 3)):
        return np.float32
    elif cv_type == CV_64F or cv_type == (CV_64F | (1 << 3)):
        return np.float64
    else:
        raise ValueError(f"Unsupported OpenCV type: {cv_type}")

# Config wrapper class
class Config:
    def __init__(self):
        self._c_config = _lib.BICOS_CreateDefaultConfig()

    def __del__(self):
        if hasattr(self, '_c_config') and self._c_config:
            _lib.BICOS_FreeConfig(self._c_config)

    @property
    def nxcorr_threshold(self):
        return self._c_config.contents.nxcorr_threshold

    @nxcorr_threshold.setter
    def nxcorr_threshold(self, value):
        self._c_config.contents.nxcorr_threshold = value

    @property
    def subpixel_step(self):
        val = self._c_config.contents.subpixel_step
        return None if val < 0 else val

    @subpixel_step.setter
    def subpixel_step(self, value):
        self._c_config.contents.subpixel_step = -1.0 if value is None else value

    @property
    def min_variance(self):
        val = self._c_config.contents.min_variance
        return None if val < 0 else val

    @min_variance.setter
    def min_variance(self, value):
        self._c_config.contents.min_variance = -1.0 if value is None else value

    @property
    def mode(self):
        return TransformMode(self._c_config.contents.mode)

    @mode.setter
    def mode(self, value):
        if isinstance(value, TransformMode):
            self._c_config.contents.mode = value.value
        else:
            self._c_config.contents.mode = value

    @property
    def precision(self):
        return Precision(self._c_config.contents.precision)

    @precision.setter
    def precision(self, value):
        if isinstance(value, Precision):
            self._c_config.contents.precision = value.value
        else:
            self._c_config.contents.precision = value

    @property
    def variant(self):
        if self._c_config.contents.variant_type == VariantType.NO_DUPLICATES.value:
            return "NoDuplicates"
        else:
            return {
                "type": "Consistency",
                "max_lr_diff": self._c_config.contents.max_lr_diff,
                "no_dupes": bool(self._c_config.contents.no_dupes)
            }

    def set_no_duplicates(self):
        self._c_config.contents.variant_type = VariantType.NO_DUPLICATES.value

    def set_consistency(self, max_lr_diff=1, no_dupes=False):
        self._c_config.contents.variant_type = VariantType.CONSISTENCY.value
        self._c_config.contents.max_lr_diff = max_lr_diff
        self._c_config.contents.no_dupes = 1 if no_dupes else 0

    def __repr__(self):
        parts = [
            f"Config(",
            f"  nxcorr_threshold={self.nxcorr_threshold}",
            f"  subpixel_step={self.subpixel_step}",
            f"  min_variance={self.min_variance}",
            f"  mode={self.mode.name}",
            f"  precision={self.precision.name}",
            f"  variant={self.variant}",
            f")"
        ]
        return "\n".join(parts)

# Main function
def match(stack0, stack1, cfg=None):
    if not stack0 or not stack1:
        raise ValueError("Empty image stacks")
    # prepare pointers, arrays, call into C API
    return _lib.BICOS_Match(
        # ... fill args appropriately
    )
