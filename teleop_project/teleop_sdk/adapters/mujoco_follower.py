"""MuJoCo URDF 运动学虚拟从臂适配器。"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree

import numpy as np

from ..interfaces import FollowerArm, GripperActuator


@dataclass(frozen=True)
class _MjcfEnvironment:
    """附加到 URDF 模型的 MuJoCo 场景环境参数。"""

    floor_z_m: float
    align_model_lowest_point_to_floor: bool


class MujocoSimulation:
    """一份共享的 MuJoCo URDF 场景。

    一个双臂 URDF 只加载一次。多个 ``MujocoFollower`` 可分别控制左右手臂的
    七个关节，并通过同一 ``MjData`` 和查看器展示完整机器人。
    """

    def __init__(
        self,
        urdf_path: str | Path,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"找不到 URDF 文件: {self.urdf_path}")

        self._joint_metadata = self._read_joint_metadata()
        self._lock = threading.RLock()
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._viewer: Any | None = None
        self._mjcf_environment: _MjcfEnvironment | None = None
        self._client_count = 0

    @property
    def viewer_is_running(self) -> bool:
        """查看器已打开且尚未被用户关闭时返回 ``True``。"""

        with self._lock:
            if self._viewer is None:
                return False
            try:
                return bool(self._viewer.is_running())
            except Exception:
                return False

    def joint_limits_deg(self, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """返回指定 URDF 旋转关节的角度制限位。"""

        min_angles: list[float] = []
        max_angles: list[float] = []
        for name in joint_names:
            metadata = self._joint_metadata.get(name)
            if metadata is None:
                raise ValueError(f"URDF 中找不到受控关节: {name}")
            joint_type, lower_rad, upper_rad = metadata
            if joint_type != "revolute":
                raise ValueError(
                    f"MuJoCo 虚拟从臂目前只支持 revolute 关节: {name}"
                )
            min_angles.append(float(np.rad2deg(lower_rad)))
            max_angles.append(float(np.rad2deg(upper_rad)))
        return np.asarray(min_angles, dtype=float), np.asarray(max_angles, dtype=float)

    def prismatic_joint_limits(
        self, joint_names: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回指定 URDF 平移关节的米制限位。"""

        minimum: list[float] = []
        maximum: list[float] = []
        for name in joint_names:
            metadata = self._joint_metadata.get(name)
            if metadata is None:
                raise ValueError(f"URDF 中找不到受控关节: {name}")
            joint_type, lower_m, upper_m = metadata
            if joint_type != "prismatic":
                raise ValueError(
                    f"MuJoCo 虚拟夹爪目前只支持 prismatic 关节: {name}"
                )
            minimum.append(lower_m)
            maximum.append(upper_m)
        return np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)

    def acquire(self) -> None:
        """为一个从臂适配器加载或复用共享场景。"""

        with self._lock:
            if self._model is None or self._data is None:
                self._load_locked()
            self._client_count += 1

    def release(self) -> None:
        """释放一个从臂适配器；最后一个释放者关闭共享场景。"""

        with self._lock:
            if self._client_count == 0:
                return
            self._client_count -= 1
            if self._client_count != 0:
                return
            viewer = self._detach_locked()

        self._close_viewer(viewer)

    def open_viewer(self) -> None:
        """打开 MuJoCo 被动查看器；调用方应在主线程中调用本方法。"""

        with self._lock:
            self._require_loaded_locked()
            self._open_viewer_locked()

    def set_mjcf_environment(
        self,
        *,
        floor_z_m: float = 0.0,
        align_model_lowest_point_to_floor: bool = False,
    ) -> None:
        """配置带渐变天空、棋盘格地板和定向光的 MJCF 场景环境。

        环境会在加载 URDF 后通过 ``MjSpec`` 编译进同一 MuJoCo 模型。地板使用
        纹理材质，不参与碰撞、接触或动力学计算。若启用最低点对齐，会在 URDF
        零位姿态下将唯一顶层根节点平移到地板平面。必须在任一从臂连接前调用。
        """

        try:
            floor_z = float(floor_z_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("MJCF 地板高度必须是有效数值") from exc
        if not np.isfinite(floor_z):
            raise ValueError("MJCF 地板高度必须是有限数")
        if not isinstance(align_model_lowest_point_to_floor, bool):
            raise ValueError("最低点对齐开关必须是布尔值")

        with self._lock:
            if (
                self._model is not None
                or self._data is not None
                or self._viewer is not None
            ):
                raise RuntimeError("MJCF 场景环境必须在 MuJoCo 场景加载前配置")
            self._mjcf_environment = _MjcfEnvironment(
                floor_z_m=floor_z,
                align_model_lowest_point_to_floor=align_model_lowest_point_to_floor,
            )

    def sync_viewer(self) -> None:
        """将当前共享场景状态同步到已打开的查看器。"""

        with self._lock:
            if self._viewer is None:
                return
            try:
                self._viewer.sync()
            except Exception:
                # 用户关闭窗口后不应影响控制循环；下次可重新调用 open_viewer。
                self._viewer = None

    def qpos_addresses(self, joint_names: Sequence[str]) -> np.ndarray:
        """查找指定关节在共享 MuJoCo ``qpos`` 数组中的位置。"""

        with self._lock:
            self._require_loaded_locked()
            assert self._mujoco is not None
            assert self._model is not None
            addresses: list[int] = []
            for name in joint_names:
                joint_id = self._mujoco.mj_name2id(
                    self._model,
                    self._mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                if joint_id < 0:
                    raise ValueError(f"MuJoCo 模型中找不到受控关节: {name}")
                addresses.append(int(self._model.jnt_qposadr[joint_id]))
            return np.asarray(addresses, dtype=int)

    def read_joint_angles_deg(self, qpos_addresses: np.ndarray) -> np.ndarray:
        """从共享 ``qpos`` 读取旋转关节，并转换为公共角度制。"""

        with self._lock:
            self._require_loaded_locked()
            assert self._data is not None
            return np.rad2deg(
                np.asarray(self._data.qpos[qpos_addresses], dtype=float)
            ).copy()

    def set_joint_angles_deg(self, qpos_addresses: np.ndarray, angles_deg: np.ndarray) -> None:
        """写入旋转关节角度，并刷新完整 URDF 的正向运动学状态。"""

        self.set_joint_positions(qpos_addresses, np.deg2rad(angles_deg))

    def set_joint_positions(
        self, qpos_addresses: np.ndarray, positions: np.ndarray
    ) -> None:
        """写入原始 ``qpos`` 位置，并刷新完整 URDF 的正向运动学状态。"""

        addresses = np.asarray(qpos_addresses, dtype=int)
        target = np.asarray(positions, dtype=float)
        if addresses.ndim != 1 or target.shape != addresses.shape:
            raise ValueError("MuJoCo qpos 地址与目标位置维度必须一致")
        if not np.isfinite(target).all():
            raise ValueError("MuJoCo 目标位置必须是有限数")

        with self._lock:
            self._require_loaded_locked()
            assert self._data is not None
            assert self._mujoco is not None
            assert self._model is not None
            self._data.qpos[addresses] = target
            self._mujoco.mj_forward(self._model, self._data)

    def _load_locked(self) -> None:
        """在调用方已持锁时加载 URDF 和 MuJoCo 数据。"""

        mujoco = _load_mujoco()
        try:
            if self._mjcf_environment is None:
                model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
            else:
                model = self._compile_mjcf_environment_model(
                    mujoco,
                    self._mjcf_environment,
                )
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        except Exception as exc:
            raise RuntimeError(f"加载 MuJoCo URDF 失败: {self.urdf_path}: {exc}") from exc

        self._mujoco = mujoco
        self._model = model
        self._data = data
        print(f"[SIM] MuJoCo 共享场景已加载: {self.urdf_path.name}")

    def _compile_mjcf_environment_model(
        self, mujoco: Any, environment: _MjcfEnvironment
    ) -> Any:
        """将 URDF 转为 ``MjSpec``，并追加与示例等价的纹理场景元素。"""

        if not hasattr(mujoco, "MjSpec"):
            raise RuntimeError("当前 MuJoCo 版本不支持 MjSpec 场景构建")

        spec = mujoco.MjSpec.from_file(str(self.urdf_path))
        if environment.align_model_lowest_point_to_floor:
            reference_model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
            lowest_z_m = self._lowest_geometry_z_m(mujoco, reference_model)
            root_bodies = tuple(spec.worldbody.bodies)
            if len(root_bodies) != 1:
                raise RuntimeError(
                    "最低点对齐要求 URDF 只有一个顶层根节点，请关闭对齐或拆分场景"
                )
            root_body = root_bodies[0]
            root_position = np.array(root_body.pos, dtype=float, copy=True)
            root_position[2] += environment.floor_z_m - lowest_z_m
            root_body.pos = root_position

        spec.visual.headlight.diffuse = (0.6, 0.6, 0.6)
        spec.visual.headlight.ambient = (0.1, 0.1, 0.1)
        spec.visual.headlight.specular = (0.0, 0.0, 0.0)
        spec.visual.rgba.haze = (0.15, 0.25, 0.35, 1.0)
        spec.visual.global_.azimuth = 120.0
        spec.visual.global_.elevation = -20.0

        spec.add_texture(
            name="teleop_sdk_skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=(0.3, 0.5, 0.7),
            rgb2=(0.0, 0.0, 0.0),
            width=512,
            height=3072,
        )
        ground_texture = spec.add_texture(
            name="teleop_sdk_ground_checker",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.1, 0.2, 0.3),
            markrgb=(0.8, 0.8, 0.8),
            width=300,
            height=300,
        )
        ground_material = spec.add_material(
            name="teleop_sdk_ground_material",
            texuniform=True,
            texrepeat=(5.0, 5.0),
            reflectance=0.2,
        )
        ground_material.textures[
            int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
        ] = ground_texture.name

        spec.worldbody.add_light(
            name="teleop_sdk_directional_light",
            pos=(0.0, 0.0, 1.5),
            dir=(0.0, 0.0, -1.0),
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_geom(
            name="teleop_sdk_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0.0, 0.0, environment.floor_z_m),
            size=(0.0, 0.0, 0.05),
            contype=0,
            conaffinity=0,
            material=ground_material.name,
        )
        return spec.compile()

    @staticmethod
    def _lowest_geometry_z_m(mujoco: Any, model: Any) -> float:
        """计算模型零位姿态中所有几何体的最低世界 Z 坐标。"""

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        lowest_z_m = float("inf")
        mesh_type = mujoco.mjtGeom.mjGEOM_MESH

        for geom_id in range(int(model.ngeom)):
            if model.geom_type[geom_id] == mesh_type:
                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id < 0:
                    continue
                vertex_start = int(model.mesh_vertadr[mesh_id])
                vertex_count = int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    model.mesh_vert[vertex_start : vertex_start + vertex_count],
                    dtype=float,
                )
                if vertices.size == 0:
                    continue
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                world_z = vertices @ rotation[2, :] + data.geom_xpos[geom_id, 2]
                lowest_z_m = min(lowest_z_m, float(np.min(world_z)))
            else:
                # CASBOT 使用网格；其他基本几何体采用包围球下界作为保守回退。
                lowest_z_m = min(
                    lowest_z_m,
                    float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id]),
                )

        if not np.isfinite(lowest_z_m):
            raise RuntimeError("无法计算 MuJoCo 模型的最低几何点")
        return lowest_z_m

    def _open_viewer_locked(self) -> None:
        """在调用方已持锁时创建被动查看器。"""

        if self.viewer_is_running:
            return
        try:
            viewer_module = importlib.import_module("mujoco.viewer")
            self._viewer = viewer_module.launch_passive(self._model, self._data)
            self._viewer.sync()
        except Exception as exc:
            self._viewer = None
            raise RuntimeError(f"打开 MuJoCo 查看器失败: {exc}") from exc

    def _detach_locked(self) -> Any | None:
        """在调用方已持锁时清空共享模型引用，并返回待关闭查看器。"""

        viewer = self._viewer
        self._viewer = None
        self._data = None
        self._model = None
        self._mujoco = None
        return viewer

    @staticmethod
    def _close_viewer(viewer: Any | None) -> None:
        """关闭查看器，关闭失败不应妨碍适配器资源回收。"""

        if viewer is None:
            return
        try:
            viewer.close()
        except Exception:
            pass

    def _read_joint_metadata(self) -> dict[str, tuple[str, float, float]]:
        """读取 URDF 关节类型和弧度限位，供多个从臂适配器共享。"""

        try:
            root = ElementTree.parse(self.urdf_path).getroot()
        except ElementTree.ParseError as exc:
            raise ValueError(f"URDF XML 格式无效: {self.urdf_path}: {exc}") from exc
        if root.tag != "robot":
            raise ValueError(f"URDF 根节点必须是 robot: {self.urdf_path}")

        metadata: dict[str, tuple[str, float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            joint_type = joint.get("type")
            if name is None or joint_type is None:
                continue
            limit = joint.find("limit")
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                continue
            try:
                lower_rad = float(limit.get("lower", ""))
                upper_rad = float(limit.get("upper", ""))
            except ValueError as exc:
                raise ValueError(f"URDF 关节限位不是有效数字: {name}") from exc
            if lower_rad >= upper_rad:
                raise ValueError(f"URDF 关节限位无效: {name}")
            metadata[name] = joint_type, lower_rad, upper_rad
        return metadata

    def _require_loaded_locked(self) -> None:
        """确保共享 MuJoCo 模型与数据已加载。"""

        if self._model is None or self._data is None or self._mujoco is None:
            raise RuntimeError("MuJoCo 共享场景尚未加载")


class MujocoFollower(FollowerArm):
    """绑定共享 MuJoCo 场景中一组旋转关节的虚拟从臂。

    两个实例可共享同一 ``MujocoSimulation``，分别表示完整 URDF 中的左、右手臂。
    位置命令直接写入 ``qpos`` 并调用 ``mj_forward``，用于运动学遥操验证；不模拟
    真实电机、减速器或力矩闭环。
    """

    def __init__(self, simulation: MujocoSimulation, joint_names: Sequence[str]):
        self.simulation = simulation
        self._joint_names = tuple(str(name) for name in joint_names)
        if not self._joint_names:
            raise ValueError("MuJoCo 从臂至少需要指定一个受控关节")
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("MuJoCo 从臂受控关节名称不能重复")

        self._min_angles_deg, self._max_angles_deg = simulation.joint_limits_deg(
            self._joint_names
        )
        self._lock = threading.RLock()
        self._qpos_addresses: np.ndarray | None = None
        self._connected = False
        self._servo_started = False

    @property
    def joint_count(self) -> int:
        """返回此从臂接口公开的受控关节数量。"""

        return len(self._joint_names)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """返回与公共关节数组一一对应的 URDF 关节名称。"""

        return self._joint_names

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """返回从 URDF ``limit`` 读取并转换为角度的安全范围。"""

        return self._min_angles_deg.copy(), self._max_angles_deg.copy()

    def connect(self) -> None:
        """连接到共享场景并建立本从臂关节到 ``qpos`` 的映射。"""

        with self._lock:
            if self._connected:
                return
            self.simulation.acquire()
            try:
                self._qpos_addresses = self.simulation.qpos_addresses(self._joint_names)
            except Exception:
                self.simulation.release()
                raise
            self._connected = True
            print(f"[SIM] MuJoCo 虚拟从臂已连接: {', '.join(self._joint_names)}")

    def read_joint_angles_deg(self) -> np.ndarray:
        """读取本从臂受控关节的角度制状态。"""

        with self._lock:
            self._require_connected()
            assert self._qpos_addresses is not None
            return self.simulation.read_joint_angles_deg(self._qpos_addresses)

    def start_servo(self) -> bool:
        """使能运动学位置命令接收。"""

        with self._lock:
            if not self._connected:
                return False
            self._servo_started = True
            return True

    def send_joint_angles_deg(
        self, angles_deg: np.ndarray, command_time_s: float
    ) -> bool:
        """写入一帧绝对角度目标，并刷新完整 URDF 的正向运动学。

        ``command_time_s`` 保留以满足 ``FollowerArm`` 接口；运动学模式会立即到达
        经过 URDF 限位钳制后的目标位置。
        """

        try:
            duration_s = float(command_time_s)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            return False
        try:
            target = np.asarray(angles_deg, dtype=float)
        except (TypeError, ValueError):
            return False
        if target.shape != (self.joint_count,) or not np.isfinite(target).all():
            return False

        with self._lock:
            if not self._servo_started:
                return False
            self._require_connected()
            assert self._qpos_addresses is not None
            target = np.clip(target, self._min_angles_deg, self._max_angles_deg)
            self.simulation.set_joint_angles_deg(self._qpos_addresses, target)

        self.simulation.sync_viewer()
        return True

    def recover(self) -> bool:
        """恢复虚拟伺服状态；运动学场景不需要厂商级故障恢复。"""

        with self._lock:
            if not self._connected:
                return False
            self._servo_started = True
            return True

    def stop_servo(self) -> None:
        """停止接收新的运动学位置命令。"""

        with self._lock:
            self._servo_started = False

    def disconnect(self) -> None:
        """解除本从臂与共享场景的连接；最后一个实例会关闭场景。"""

        with self._lock:
            if not self._connected:
                return
            self._servo_started = False
            self._qpos_addresses = None
            self._connected = False
        self.simulation.release()

    def _require_connected(self) -> None:
        """确保本从臂已成功关联到共享 MuJoCo 场景。"""

        if not self._connected or self._qpos_addresses is None:
            raise RuntimeError("MuJoCo 虚拟从臂尚未连接")


class MujocoGripper(GripperActuator):
    """将归一化开合量映射到一组 MuJoCo 平移指关节。"""

    def __init__(
        self,
        simulation: MujocoSimulation,
        joint_names: Sequence[str],
        closed_positions_m: Sequence[float],
        open_positions_m: Sequence[float],
    ) -> None:
        self.simulation = simulation
        self._joint_names = tuple(str(name) for name in joint_names)
        if not self._joint_names:
            raise ValueError("MuJoCo 虚拟夹爪至少需要指定一个平移关节")
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("MuJoCo 虚拟夹爪关节名称不能重复")

        self._minimum_m, self._maximum_m = simulation.prismatic_joint_limits(
            self._joint_names
        )
        self._closed_positions_m = np.asarray(closed_positions_m, dtype=float)
        self._open_positions_m = np.asarray(open_positions_m, dtype=float)
        expected_shape = (len(self._joint_names),)
        if (
            self._closed_positions_m.shape != expected_shape
            or self._open_positions_m.shape != expected_shape
        ):
            raise ValueError("夹爪闭合和张开位置数量必须与关节数量一致")
        if not (
            np.isfinite(self._closed_positions_m).all()
            and np.isfinite(self._open_positions_m).all()
        ):
            raise ValueError("夹爪闭合和张开位置必须是有限数")
        if np.any(self._closed_positions_m < self._minimum_m) or np.any(
            self._closed_positions_m > self._maximum_m
        ):
            raise ValueError("夹爪闭合位置超出 URDF 关节限位")
        if np.any(self._open_positions_m < self._minimum_m) or np.any(
            self._open_positions_m > self._maximum_m
        ):
            raise ValueError("夹爪张开位置超出 URDF 关节限位")

        self._lock = threading.RLock()
        self._qpos_addresses: np.ndarray | None = None
        self._connected = False
        self._enabled = False

    @property
    def joint_names(self) -> tuple[str, ...]:
        """返回夹爪平移关节名称及其开合位置数组顺序。"""

        return self._joint_names

    def connect(self) -> None:
        """关联共享 MuJoCo 场景并使能归一化开合命令。"""

        with self._lock:
            if self._connected:
                self._enabled = True
                return
            self.simulation.acquire()
            try:
                self._qpos_addresses = self.simulation.qpos_addresses(self._joint_names)
            except Exception:
                self.simulation.release()
                raise
            self._connected = True
            self._enabled = True
            print(f"[SIM] MuJoCo 虚拟夹爪已连接: {', '.join(self._joint_names)}")

    def send_normalized(self, opening: float) -> bool:
        """将 0（闭合）到 1（张开）线性映射为一组平移关节位置。"""

        try:
            normalized = float(opening)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(normalized):
            return False
        normalized = float(np.clip(normalized, 0.0, 1.0))
        target_m = self._closed_positions_m + normalized * (
            self._open_positions_m - self._closed_positions_m
        )

        with self._lock:
            if not self._connected or not self._enabled:
                return False
            assert self._qpos_addresses is not None
            self.simulation.set_joint_positions(self._qpos_addresses, target_m)

        self.simulation.sync_viewer()
        return True

    def disable(self) -> None:
        """停止接受新的夹爪位置命令，保留当前仿真姿态。"""

        with self._lock:
            self._enabled = False

    def disconnect(self) -> None:
        """解除夹爪对共享 MuJoCo 场景的引用。"""

        with self._lock:
            if not self._connected:
                return
            self._enabled = False
            self._qpos_addresses = None
            self._connected = False
        self.simulation.release()


def _load_mujoco() -> Any:
    """延迟导入可选依赖，避免普通硬件部署强制安装 MuJoCo。"""

    try:
        return importlib.import_module("mujoco")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuJoCo 是可选依赖，请执行: "
            "python -m pip install -r requirements-mojoco.txt"
        ) from exc
