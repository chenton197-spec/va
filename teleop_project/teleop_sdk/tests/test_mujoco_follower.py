"""MuJoCo 虚拟从臂的无图形、无 MuJoCo 安装测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np

from teleop_sdk.adapters.mujoco_follower import MujocoFollower, MujocoSimulation


class _FakeModel:
    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        self.jnt_qposadr = np.arange(len(joint_names), dtype=int)


class _FakeModelFactory:
    joint_names = ("left_a", "left_b", "right_a", "right_b")
    xml_paths: list[str] = []

    @classmethod
    def from_xml_path(cls, xml_path: str) -> _FakeModel:
        cls.xml_paths.append(xml_path)
        return _FakeModel(cls.joint_names)


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        self.qpos = np.zeros(len(model.jnt_qposadr), dtype=float)


class _FakeMjtObj:
    mjOBJ_JOINT = object()


class _FakeMjtGeom:
    mjGEOM_PLANE = "plane"


class _FakeMjtTexture:
    mjTEXTURE_2D = "2d"
    mjTEXTURE_SKYBOX = "skybox"


class _FakeMjtBuiltin:
    mjBUILTIN_GRADIENT = "gradient"
    mjBUILTIN_CHECKER = "checker"


class _FakeMjtMark:
    mjMARK_EDGE = "edge"


class _FakeMjtTextureRole:
    mjTEXROLE_RGB = 1


class _FakeMjtLightType:
    mjLIGHT_DIRECTIONAL = "directional"


class _FakeSetting:
    pass


class _FakeVisual:
    def __init__(self) -> None:
        self.headlight = _FakeSetting()
        self.rgba = _FakeSetting()
        self.global_ = _FakeSetting()


class _FakeTexture:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeMaterial:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.textures = [""] * 10


class _FakeBody:
    def __init__(self, name: str) -> None:
        self.name = name
        self.pos = np.zeros(3, dtype=float)


class _FakeWorldBody:
    def __init__(self) -> None:
        self.bodies = [_FakeBody("robot_root")]
        self.geoms: list[_FakeTexture] = []
        self.lights: list[_FakeTexture] = []

    def add_geom(self, **kwargs: object) -> _FakeTexture:
        geom = _FakeTexture(**kwargs)
        self.geoms.append(geom)
        return geom

    def add_light(self, **kwargs: object) -> _FakeTexture:
        light = _FakeTexture(**kwargs)
        self.lights.append(light)
        return light


class _FakeSpec:
    xml_paths: list[str] = []
    instances: list["_FakeSpec"] = []

    def __init__(self) -> None:
        self.visual = _FakeVisual()
        self.textures: list[_FakeTexture] = []
        self.materials: list[_FakeMaterial] = []
        self.worldbody = _FakeWorldBody()
        self.compile_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.xml_paths = []
        cls.instances = []

    @classmethod
    def from_file(cls, xml_path: str) -> "_FakeSpec":
        spec = cls()
        cls.xml_paths.append(xml_path)
        cls.instances.append(spec)
        return spec

    def add_texture(self, **kwargs: object) -> _FakeTexture:
        texture = _FakeTexture(**kwargs)
        self.textures.append(texture)
        return texture

    def add_material(self, **kwargs: object) -> _FakeMaterial:
        material = _FakeMaterial(**kwargs)
        self.materials.append(material)
        return material

    def compile(self) -> _FakeModel:
        self.compile_calls += 1
        return _FakeModel(_FakeModelFactory.joint_names)


class _FakeMujoco:
    MjModel = _FakeModelFactory
    MjSpec = _FakeSpec
    MjData = _FakeData
    mjtObj = _FakeMjtObj
    mjtGeom = _FakeMjtGeom
    mjtTexture = _FakeMjtTexture
    mjtBuiltin = _FakeMjtBuiltin
    mjtMark = _FakeMjtMark
    mjtTextureRole = _FakeMjtTextureRole
    mjtLightType = _FakeMjtLightType
    forward_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.forward_calls = 0
        _FakeModelFactory.xml_paths = []
        _FakeSpec.reset()

    @staticmethod
    def mj_name2id(model: _FakeModel, _object_type: object, name: str) -> int:
        try:
            return model.joint_names.index(name)
        except ValueError:
            return -1

    @classmethod
    def mj_forward(cls, _model: _FakeModel, _data: _FakeData) -> None:
        cls.forward_calls += 1


class MujocoFollowerTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeMujoco.reset()
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.urdf_path = self.root / "urdfs" / "test.urdf"
        self.urdf_path.parent.mkdir()
        self.urdf_path.write_text(
            """<robot name="test">
  <mujoco><compiler meshdir="../meshes" strippath="false" discardvisual="false"/></mujoco>
  <link name="base"/>
  <link name="mesh_link"><visual><geometry><mesh filename="link.stl"/></geometry></visual></link>
  <link name="left_a_link"/>
  <link name="left_b_link"/>
  <link name="right_a_link"/>
  <link name="right_b_link"/>
  <link name="slider_link"/>
  <joint name="left_a" type="revolute"><parent link="base"/><child link="left_a_link"/><limit lower="-1.0" upper="1.0"/></joint>
  <joint name="left_b" type="revolute"><parent link="base"/><child link="left_b_link"/><limit lower="-0.5" upper="0.5"/></joint>
  <joint name="right_a" type="revolute"><parent link="base"/><child link="right_a_link"/><limit lower="-1.5" upper="1.5"/></joint>
  <joint name="right_b" type="revolute"><parent link="base"/><child link="right_b_link"/><limit lower="-2.0" upper="2.0"/></joint>
  <joint name="slider" type="fixed"><parent link="base"/><child link="slider_link"/><limit lower="0.0" upper="0.2"/></joint>
</robot>""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _simulation(self) -> MujocoSimulation:
        return MujocoSimulation(self.urdf_path)

    def test_reads_degree_limits_from_a_native_mujoco_urdf(self) -> None:
        simulation = self._simulation()

        minimum, maximum = simulation.joint_limits_deg(("left_a", "left_b"))
        np.testing.assert_allclose(minimum, np.rad2deg([-1.0, -0.5]))
        np.testing.assert_allclose(maximum, np.rad2deg([1.0, 0.5]))

        root = ElementTree.parse(self.urdf_path).getroot()
        mesh = root.find(".//mesh")
        self.assertIsNotNone(mesh)
        assert mesh is not None
        self.assertEqual(mesh.get("filename"), "link.stl")
        compiler = root.find("mujoco/compiler")
        self.assertIsNotNone(compiler)
        assert compiler is not None
        self.assertEqual(compiler.get("meshdir"), "../meshes")
        self.assertEqual(compiler.get("strippath"), "false")
        self.assertEqual(compiler.get("discardvisual"), "false")

    def test_two_followers_share_one_scene_and_keep_sides_isolated(self) -> None:
        simulation = self._simulation()
        left = MujocoFollower(simulation, ("left_a", "left_b"))
        right = MujocoFollower(simulation, ("right_a", "right_b"))

        with patch(
            "teleop_sdk.adapters.mujoco_follower._load_mujoco", return_value=_FakeMujoco
        ):
            left.connect()
            right.connect()
            self.assertEqual(_FakeModelFactory.xml_paths, [str(self.urdf_path.resolve())])
            self.assertTrue(left.start_servo())
            self.assertTrue(right.start_servo())
            self.assertTrue(left.send_joint_angles_deg(np.array([15.0, -20.0]), 0.008))
            self.assertTrue(right.send_joint_angles_deg(np.array([-30.0, 45.0]), 0.008))

            np.testing.assert_allclose(left.read_joint_angles_deg(), [15.0, -20.0])
            np.testing.assert_allclose(right.read_joint_angles_deg(), [-30.0, 45.0])
            assert simulation._data is not None
            np.testing.assert_allclose(
                simulation._data.qpos,
                np.deg2rad([15.0, -20.0, -30.0, 45.0]),
            )

            left.disconnect()
            self.assertIsNotNone(simulation._data)
            right.disconnect()
            self.assertIsNone(simulation._data)
            self.assertEqual(simulation._client_count, 0)

    def test_rejects_invalid_commands_and_clips_to_urdf_limits(self) -> None:
        simulation = self._simulation()
        follower = MujocoFollower(simulation, ("left_a", "left_b"))

        with patch(
            "teleop_sdk.adapters.mujoco_follower._load_mujoco", return_value=_FakeMujoco
        ):
            follower.connect()
            self.assertTrue(follower.start_servo())
            initial = follower.read_joint_angles_deg()
            self.assertFalse(follower.send_joint_angles_deg(np.array([1.0]), 0.008))
            self.assertFalse(follower.send_joint_angles_deg(np.array([np.nan, 1.0]), 0.008))
            self.assertFalse(follower.send_joint_angles_deg(np.array([1.0, 1.0]), 0.0))
            self.assertFalse(follower.send_joint_angles_deg(np.array([1.0, 1.0]), np.nan))
            self.assertFalse(follower.send_joint_angles_deg(np.array([1.0, 1.0]), "bad"))
            np.testing.assert_allclose(follower.read_joint_angles_deg(), initial)

            self.assertTrue(follower.send_joint_angles_deg(np.array([100.0, -100.0]), 0.008))
            lower, upper = follower.joint_limits_deg
            np.testing.assert_allclose(follower.read_joint_angles_deg(), [upper[0], lower[1]])
            follower.disconnect()

    def test_rejects_missing_duplicate_and_non_revolute_joint_names(self) -> None:
        simulation = self._simulation()

        with self.assertRaisesRegex(ValueError, "找不到受控关节"):
            MujocoFollower(simulation, ("missing",))
        with self.assertRaisesRegex(ValueError, "只支持 revolute"):
            MujocoFollower(simulation, ("slider",))
        with self.assertRaisesRegex(ValueError, "不能重复"):
            MujocoFollower(simulation, ("left_a", "left_a"))

    def test_compiles_textured_mjcf_environment_from_urdf(self) -> None:
        simulation = self._simulation()
        simulation.set_mjcf_environment(floor_z_m=-0.20)

        with patch(
            "teleop_sdk.adapters.mujoco_follower._load_mujoco",
            return_value=_FakeMujoco,
        ):
            simulation.acquire()
            self.assertEqual(_FakeSpec.xml_paths, [str(self.urdf_path.resolve())])
            self.assertEqual(_FakeModelFactory.xml_paths, [])
            self.assertEqual(len(_FakeSpec.instances), 1)

        spec = _FakeSpec.instances[0]
        self.assertEqual(spec.compile_calls, 1)
        self.assertEqual(spec.visual.headlight.diffuse, (0.6, 0.6, 0.6))
        self.assertEqual(spec.visual.headlight.ambient, (0.1, 0.1, 0.1))
        self.assertEqual(spec.visual.rgba.haze, (0.15, 0.25, 0.35, 1.0))
        self.assertEqual(spec.visual.global_.azimuth, 120.0)
        self.assertEqual(spec.visual.global_.elevation, -20.0)

        self.assertEqual(len(spec.textures), 2)
        skybox, ground_texture = spec.textures
        self.assertEqual(skybox.type, _FakeMjtTexture.mjTEXTURE_SKYBOX)
        self.assertEqual(skybox.builtin, _FakeMjtBuiltin.mjBUILTIN_GRADIENT)
        self.assertEqual(skybox.rgb1, (0.3, 0.5, 0.7))
        self.assertEqual(ground_texture.type, _FakeMjtTexture.mjTEXTURE_2D)
        self.assertEqual(ground_texture.builtin, _FakeMjtBuiltin.mjBUILTIN_CHECKER)
        self.assertEqual(ground_texture.mark, _FakeMjtMark.mjMARK_EDGE)

        self.assertEqual(len(spec.materials), 1)
        material = spec.materials[0]
        self.assertEqual(material.textures[_FakeMjtTextureRole.mjTEXROLE_RGB], ground_texture.name)
        self.assertEqual(material.texrepeat, (5.0, 5.0))
        self.assertTrue(material.texuniform)
        self.assertEqual(material.reflectance, 0.2)

        self.assertEqual(len(spec.worldbody.lights), 1)
        light = spec.worldbody.lights[0]
        self.assertEqual(light.type, _FakeMjtLightType.mjLIGHT_DIRECTIONAL)
        self.assertEqual(light.dir, (0.0, 0.0, -1.0))
        self.assertEqual(len(spec.worldbody.geoms), 1)
        floor = spec.worldbody.geoms[0]
        self.assertEqual(floor.type, _FakeMjtGeom.mjGEOM_PLANE)
        self.assertEqual(floor.pos, (0.0, 0.0, -0.20))
        self.assertEqual(floor.size, (0.0, 0.0, 0.05))
        self.assertEqual(floor.material, material.name)
        self.assertEqual((floor.contype, floor.conaffinity), (0, 0))

        with self.assertRaisesRegex(RuntimeError, "加载前"):
            simulation.set_mjcf_environment(floor_z_m=-0.25)
        simulation.release()

    def test_aligns_unique_model_root_lowest_point_to_floor(self) -> None:
        simulation = self._simulation()
        simulation.set_mjcf_environment(
            floor_z_m=-0.20,
            align_model_lowest_point_to_floor=True,
        )

        with (
            patch(
                "teleop_sdk.adapters.mujoco_follower._load_mujoco",
                return_value=_FakeMujoco,
            ),
            patch.object(MujocoSimulation, "_lowest_geometry_z_m", return_value=-0.30),
        ):
            simulation.acquire()

        spec = _FakeSpec.instances[0]
        np.testing.assert_allclose(spec.worldbody.bodies[0].pos, [0.0, 0.0, 0.10])
        simulation.release()

    def test_rejects_invalid_mjcf_environment_configuration(self) -> None:
        simulation = self._simulation()

        with self.assertRaisesRegex(ValueError, "有限"):
            simulation.set_mjcf_environment(floor_z_m=np.nan)
        with self.assertRaisesRegex(ValueError, "数值"):
            simulation.set_mjcf_environment(floor_z_m="bad")
        with self.assertRaisesRegex(ValueError, "布尔"):
            simulation.set_mjcf_environment(align_model_lowest_point_to_floor=1)
