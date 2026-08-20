"""Independent multi-camera Orbbec SDK with no collection-framework dependency."""

from .backend import OrbbecSdkUnavailableError
from .config import load_orbbec_camera_configs
from .manager import OrbbecCamera, OrbbecError, OrbbecManager, OrbbecStartupError
from .types import (
    AlignmentMode,
    CameraMetadata,
    CameraMode,
    CameraStatus,
    OrbbecCameraConfig,
    OrbbecFrame,
)

__all__ = [
    "AlignmentMode",
    "CameraMetadata",
    "CameraMode",
    "CameraStatus",
    "OrbbecCamera",
    "OrbbecCameraConfig",
    "OrbbecError",
    "OrbbecFrame",
    "OrbbecManager",
    "OrbbecSdkUnavailableError",
    "OrbbecStartupError",
    "load_orbbec_camera_configs",
]
