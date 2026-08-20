"""加载正式 HCX 原生扩展。"""

from __future__ import annotations

try:
    from . import _hcx_native as native
except ImportError as exc:
    raise ImportError(
        "The HCX native backend could not be loaded. Copy the entire hcx_sdk "
        "directory, including _hcx_native and lib/libxoip.so, and use Linux "
        "x86_64 with the CPython version used to build the SDK."
    ) from exc
