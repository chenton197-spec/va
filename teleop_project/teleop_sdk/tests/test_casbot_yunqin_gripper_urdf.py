"""CASBOT 二指夹爪 URDF 的静态结构测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from xml.etree import ElementTree


class CasbotYunqinGripperUrdfTest(unittest.TestCase):
    _URDF_FILENAMES = (
        "CASBOTWL12_WL12P1.urdf",
        "CASBOTWL12_WL12P2.urdf",
    )
    _MOUNT_ORIGINS = {
        "left": {
            "xyz": "0.0249775224791579 0.0000192745744934109 -0.0695833186467645",
            "rpy": "1.5707963267948966 0 1.5707963267948966",
        },
        "right": {
            "xyz": "-0.0249775224791579 -0.0000192745744934109 -0.0695833186467645",
            "rpy": "1.5707963267948966 0 4.71238898038469",
        },
    }
    _FINGER_LIMITS_M = {
        "finger_a": ("0", "0.04947"),
        "finger_b": ("-0.04947", "0"),
    }
    _FINGER_AXES = {
        "finger_a": "0 0 1",
        "finger_b": "0 0 -1",
    }
    _FINGER_VISUAL_RGBA = "0.05 0.05 0.05 1"

    def _root_for(self, filename: str) -> ElementTree.Element:
        project_root = Path(__file__).resolve().parents[2]
        urdf_path = project_root / "simulation" / "urdfs" / filename
        return ElementTree.parse(urdf_path).getroot()

    def test_replaces_dexterous_hands_with_yunqin_grippers(self) -> None:
        for filename in self._URDF_FILENAMES:
            with self.subTest(filename=filename):
                root = self._root_for(filename)
                links = {link.get("name") for link in root.findall("link")}
                joints = {
                    joint.get("name"): joint
                    for joint in root.findall("joint")
                    if joint.get("name") is not None
                }
                mesh_filenames = {
                    mesh.get("filename") for mesh in root.findall(".//mesh")
                }

                self.assertFalse(any("ROH_LiteS001" in name for name in links if name))
                self.assertFalse(
                    any("ROH_LiteS001" in filename for filename in mesh_filenames if filename)
                )
                self.assertNotIn("left_hand_frame", links)
                self.assertNotIn("right_hand_frame", links)

                for side in ("left", "right"):
                    prefix = f"{side}_yunqin_gripper"
                    base_link = f"{prefix}_base_link"
                    finger_a_link = f"{prefix}_finger_a_link"
                    finger_b_link = f"{prefix}_finger_b_link"
                    self.assertTrue(
                        {base_link, finger_a_link, finger_b_link}.issubset(links)
                    )

                    mount = joints[f"{prefix}_mount_joint"]
                    self.assertEqual(mount.get("type"), "fixed")
                    self.assertEqual(mount.find("parent").get("link"), f"{side}_flange_frame")
                    self.assertEqual(mount.find("child").get("link"), base_link)
                    mount_origin = mount.find("origin")
                    self.assertIsNotNone(mount_origin)
                    assert mount_origin is not None
                    self.assertEqual(
                        mount_origin.get("xyz"), self._MOUNT_ORIGINS[side]["xyz"]
                    )
                    self.assertEqual(
                        mount_origin.get("rpy"), self._MOUNT_ORIGINS[side]["rpy"]
                    )

                    for finger_name in ("finger_a", "finger_b"):
                        finger_link = root.find(f"link[@name='{prefix}_{finger_name}_link']")
                        self.assertIsNotNone(finger_link)
                        assert finger_link is not None
                        finger_color = finger_link.find("visual/material/color")
                        self.assertIsNotNone(finger_color)
                        assert finger_color is not None
                        self.assertEqual(
                            finger_color.get("rgba"), self._FINGER_VISUAL_RGBA
                        )

                        finger_joint = joints[f"{prefix}_{finger_name}_joint"]
                        self.assertEqual(finger_joint.get("type"), "prismatic")
                        self.assertEqual(finger_joint.find("parent").get("link"), base_link)
                        self.assertEqual(
                            finger_joint.find("axis").get("xyz"),
                            self._FINGER_AXES[finger_name],
                        )
                        limit = finger_joint.find("limit")
                        self.assertIsNotNone(limit)
                        assert limit is not None
                        expected_lower, expected_upper = self._FINGER_LIMITS_M[finger_name]
                        self.assertEqual(limit.get("lower"), expected_lower)
                        self.assertEqual(limit.get("upper"), expected_upper)

                self.assertIn("yunqin1.1.SLDASM/L_Link7.STL", mesh_filenames)
                self.assertIn("yunqin1.1.SLDASM/L_Link8.STL", mesh_filenames)
                self.assertIn("yunqin1.1.SLDASM/L_Link9.STL", mesh_filenames)

    def test_keeps_fourteen_arm_joints_and_adds_four_gripper_sliders(self) -> None:
        for filename in self._URDF_FILENAMES:
            with self.subTest(filename=filename):
                root = self._root_for(filename)
                arm_joint_names = {
                    joint.get("name")
                    for joint in root.findall("joint")
                    if joint.get("type") == "revolute"
                }
                gripper_joint_names = {
                    joint.get("name")
                    for joint in root.findall("joint")
                    if joint.get("type") == "prismatic"
                }

                self.assertEqual(len(arm_joint_names), 14)
                self.assertTrue(
                    all(
                        name.startswith(("left_", "right_"))
                        and "yunqin_gripper" not in name
                        for name in arm_joint_names
                        if name is not None
                    )
                )
                self.assertEqual(
                    gripper_joint_names,
                    {
                        "left_yunqin_gripper_finger_a_joint",
                        "left_yunqin_gripper_finger_b_joint",
                        "right_yunqin_gripper_finger_a_joint",
                        "right_yunqin_gripper_finger_b_joint",
                    },
                )
