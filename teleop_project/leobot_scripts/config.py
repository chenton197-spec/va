"""Deployment configuration for collection programs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .v21_writer import ImageStorage


DEFAULT_TELEOP_YAML_PATH = Path(__file__).resolve().parents[1] / "teleop.yaml"
_RECORDING_FIELDS = frozenset(
    {
        "root",
        "fps",
        "task",
        "image_storage",
        "quality",
        "numeric_sample_fps",
        "master_camera",
        "enabled_cameras",
        "min_free_disk_gb",
    }
)
_REQUIRED_RECORDING_FIELDS = frozenset({"root", "fps", "task"})


@dataclass(frozen=True)
class RecordingDeploymentConfig:
    """Static collection settings loaded from the reference deployment YAML.

    ``fps`` is the dataset output rate. ``numeric_sample_fps`` applies when a
    camera-driven recorder independently samples robot and gripper feedback.
    ``root`` is the persistent dataset directory: a compatible existing
    dataset receives new episodes, while a missing directory is initialized.
    """

    root: Path
    fps: int
    task: str
    numeric_sample_fps: int
    master_camera: str | None = None
    image_storage: ImageStorage = "video"
    quality: int = 75
    enabled_cameras: tuple[str, ...] = ()
    min_free_disk_gb: float = 10.0


def load_recording_config(
    path: str | Path = DEFAULT_TELEOP_YAML_PATH,
    *,
    section_name: str = "recording",
) -> RecordingDeploymentConfig:
    """Load one recording deployment section from ``teleop.yaml``."""

    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("section_name must be a non-empty string")
    section_name = section_name.strip()

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Recording deployment config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 teleop.yaml 的 recording 配置需要安装 PyYAML") from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("teleop.yaml 根节点必须是映射对象")
    section = data.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"teleop.yaml 必须包含 {section_name} 映射配置")
    unknown = set(section) - _RECORDING_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"teleop.yaml 的 {section_name} 包含未知字段: {names}")
    missing = _REQUIRED_RECORDING_FIELDS - set(section)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"teleop.yaml 的 {section_name} 缺少字段: {names}")

    root = _dataset_root(section["root"], config_path)
    fps = _positive_int(section["fps"], f"{section_name}.fps")
    task = _nonempty_string(section["task"], f"{section_name}.task")
    numeric_sample_fps = _positive_int(
        section.get("numeric_sample_fps", fps), f"{section_name}.numeric_sample_fps"
    )
    master_camera = (
        None
        if "master_camera" not in section
        else _nonempty_string(section["master_camera"], f"{section_name}.master_camera")
    )
    image_storage = _image_storage(
        section.get("image_storage", "video"), f"{section_name}.image_storage"
    )
    quality = _image_quality(section.get("quality", 75), f"{section_name}.quality")
    enabled_cameras = _camera_names(
        section.get("enabled_cameras", []), f"{section_name}.enabled_cameras"
    )
    min_free_disk_gb = _positive_float(
        section.get("min_free_disk_gb", 10.0), f"{section_name}.min_free_disk_gb"
    )
    return RecordingDeploymentConfig(
        root=root,
        fps=fps,
        task=task,
        numeric_sample_fps=numeric_sample_fps,
        master_camera=master_camera,
        image_storage=image_storage,
        quality=quality,
        enabled_cameras=enabled_cameras,
        min_free_disk_gb=min_free_disk_gb,
    )


def _dataset_root(value: Any, config_path: Path) -> Path:
    root_text = _nonempty_string(value, "recording.root")
    root = Path(root_text).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"teleop.yaml 的 {name} 必须是正整数")
    return int(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teleop.yaml 的 {name} 必须是非空字符串")
    return value.strip()


def _image_storage(value: Any, name: str) -> ImageStorage:
    if value == "video":
        return "video"
    if value == "png":
        return "png"
    if value == "jpg":
        return "jpg"
    raise ValueError(f"teleop.yaml 的 {name} 必须是 'video'、'png' 或 'jpg'")


def _image_quality(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError(f"teleop.yaml 的 {name} 必须是 1 到 100 的整数")
    return int(value)


def _camera_names(value: Any, name: str = "recording.enabled_cameras") -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"teleop.yaml 的 {name} 必须是字符串列表")
    names = tuple(_nonempty_string(item, name) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"teleop.yaml 的 {name} 不能包含重复名称")
    return names


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"teleop.yaml 的 {name} 必须是正数")
    return float(value)
