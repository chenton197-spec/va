"""Independent video and PNG-sequence collection helpers.

The package deliberately depends on the generic ``teleop_sdk`` interfaces,
not on a particular leader arm, follower arm, or the upstream LeRobot package.
"""

from .config import RecordingDeploymentConfig, load_recording_config
from .camera import (
    CameraAdapterConfig,
    CameraAdapterSession,
    CameraFrameSource,
    CameraSourceHealth,
    RGBDFrameSource,
)
from .camera_recorder import CameraProcessDatasetRecorder, CameraRecorderHealth
from .episode_ops import AsyncEpisodeFinalizer, EpisodeOperation, EpisodeOperationResult
from .master_triggered_camera_recorder import (
    MasterFrameRequest,
    MasterFrameSkip,
    MasterFrameSnapshot,
    MasterTriggeredCameraProcessDatasetRecorder,
)
from .recorder import DatasetRecorder, EpisodeRecorder, RecorderConfig, RecordingFollower, RecordingGripper
from .orbbec import (
    OrbbecCameraAdapterConfig,
    OrbbecCameraAdapterSession,
    OrbbecDepthSource,
    OrbbecRGBDSource,
    OrbbecRGBSource,
)
from .sources import (
    CameraFrame,
    CallableGripperFeedbackSource,
    DepthFrame,
    DepthMetadata,
    GripperFeedbackSource,
    RGBDFrame,
    RGBDMetadata,
    TimedValue,
)

__all__ = [
    "CameraFrame",
    "CameraFrameSource",
    "CameraAdapterConfig",
    "CameraAdapterSession",
    "CameraProcessDatasetRecorder",
    "CameraRecorderHealth",
    "CameraSourceHealth",
    "MasterFrameRequest",
    "MasterFrameSkip",
    "MasterFrameSnapshot",
    "MasterTriggeredCameraProcessDatasetRecorder",
    "AsyncEpisodeFinalizer",
    "CallableGripperFeedbackSource",
    "DatasetRecorder",
    "DepthFrame",
    "DepthMetadata",
    "EpisodeRecorder",
    "EpisodeOperation",
    "EpisodeOperationResult",
    "GripperFeedbackSource",
    "OrbbecDepthSource",
    "OrbbecCameraAdapterConfig",
    "OrbbecCameraAdapterSession",
    "OrbbecRGBDSource",
    "OrbbecRGBSource",
    "RecorderConfig",
    "RecordingDeploymentConfig",
    "RecordingFollower",
    "RecordingGripper",
    "RGBDFrame",
    "RGBDFrameSource",
    "RGBDMetadata",
    "TimedValue",
    "load_recording_config",
]
