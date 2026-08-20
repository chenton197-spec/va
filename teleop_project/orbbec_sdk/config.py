"""Checked-in Orbbec resources and reference-deployment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import AlignmentMode, CameraMode, OrbbecCameraConfig


DEFAULT_XML_CONFIG_PATH = Path(__file__).with_name("config") / "OrbbecSDKConfig_casbot.xml"
DEFAULT_TELEOP_YAML_PATH = Path(__file__).resolve().parents[1] / "teleop.yaml"

_CAMERA_FIELDS = frozenset(
    {"name", "serial_number", "mode", "rgb_resolution", "depth_resolution", "fps", "alignment"}
)
_ORBBEC_FIELDS = frozenset({"cameras"})


def load_orbbec_camera_configs(
    path: str | Path = DEFAULT_TELEOP_YAML_PATH,
    *,
    section_name: str = "orbbec",
) -> tuple[OrbbecCameraConfig, ...]:
    """Load one ``<section_name>.cameras`` declaration from deployment YAML."""

    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("section_name must be a non-empty string")
    section_name = section_name.strip()

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Orbbec deployment config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 teleop.yaml 的 Orbbec 配置需要安装 PyYAML") from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("teleop.yaml 根节点必须是映射对象")

    section = data.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"teleop.yaml 必须包含 {section_name} 映射配置")
    unknown = set(section) - _ORBBEC_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"teleop.yaml 的 {section_name} 包含未知字段: {names}")
    cameras = section.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"teleop.yaml 的 {section_name}.cameras 必须是非空列表")

    configs = tuple(
        _camera_config(camera, index, section_name)
        for index, camera in enumerate(cameras)
    )
    names = [config.name for config in configs]
    serial_numbers = [config.serial_number for config in configs]
    if len(names) != len(set(names)) or len(serial_numbers) != len(set(serial_numbers)):
        raise ValueError(
            f"{section_name}.cameras 的 name 和 serial_number 必须唯一"
        )
    return configs


def _camera_config(
    value: Any, index: int, section_name: str = "orbbec"
) -> OrbbecCameraConfig:
    location = f"{section_name}.cameras[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"teleop.yaml 的 {location} 必须是映射对象")
    unknown = set(value) - _CAMERA_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"teleop.yaml 的 {location} 包含未知字段: {names}")
    missing = {"name", "serial_number"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"teleop.yaml 的 {location} 缺少字段: {names}")

    settings = dict(value)
    for field in ("name", "serial_number"):
        if not isinstance(settings[field], str):
            raise ValueError(f"teleop.yaml 的 {location}.{field} 必须是字符串")
    settings["mode"] = _enum_value(settings.get("mode", CameraMode.RGBD), CameraMode, location, "mode")
    if "alignment" in settings and settings["alignment"] is not None:
        settings["alignment"] = _enum_value(
            settings["alignment"], AlignmentMode, location, "alignment"
        )
    return OrbbecCameraConfig(**settings)


def _enum_value(value: Any, enum_type: type[CameraMode] | type[AlignmentMode], location: str, name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        values = ", ".join(member.value for member in enum_type)
        raise ValueError(f"teleop.yaml 的 {location}.{name} 必须是: {values}") from exc
