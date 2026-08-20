"""代码默认配置与 YAML 覆盖的无硬件测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teleop_sdk.config import load_runtime_config


class RuntimeConfigTest(unittest.TestCase):
    def test_missing_file_uses_code_defaults(self) -> None:
        runtime = load_runtime_config("/tmp/nonexistent-teleop-config.yaml")

        self.assertIsNone(runtime.teleop.axis_order)
        self.assertEqual(runtime.fr3.robot_ip, "192.168.57.3")

    def test_yaml_overrides_only_specified_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                "fr3:\n  robot_ip: 10.0.0.8\nteleop:\n  axis_sign: [1, -1, 1, -1, 1, -1]\n",
                encoding="utf-8",
            )
            runtime = load_runtime_config(path)

        self.assertEqual(runtime.fr3.robot_ip, "10.0.0.8")
        self.assertEqual(runtime.fr3.axis_sign, (1, -1, 1, -1, 1, -1))
        self.assertIsNone(runtime.teleop.axis_sign)
        self.assertEqual(runtime.alicia.gripper_type, "50mm")

    def test_fr3_axis_sign_rejects_a_conflicting_legacy_teleop_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                "fr3:\n  axis_sign: [1, 1, 1, 1, 1, 1]\n"
                "teleop:\n  axis_sign: [-1, -1, -1, -1, -1, -1]\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不能同时配置"):
                load_runtime_config(path)

    def test_ignores_orbbec_section_owned_by_the_hardware_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """fr3:
  robot_ip: 10.0.0.8
orbbec:
  cameras: []
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.fr3.robot_ip, "10.0.0.8")

    def test_ignores_recording_section_owned_by_the_collection_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """fr3:
  robot_ip: 10.0.0.8
recording:
  root: datasets/demo
  fps: 30
  task: move cube
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.fr3.robot_ip, "10.0.0.8")

    def test_ignores_hcx_collection_sections_owned_by_the_collection_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """hcx_orbbec:
  cameras: []
hcx_recording:
  root: datasets/hcx
  fps: 30
  task: dual arm task
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.hcx.left_robot_id, 2)

    def test_gloria_contact_settings_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """gloria_m:
  contact_torque_nm: 0.4
  contact_stall_duration_s: 0.2
  hold_torque_nm: 0.15
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.gloria_m.contact_torque_nm, 0.4)
        self.assertEqual(runtime.gloria_m.contact_stall_duration_s, 0.2)
        self.assertEqual(runtime.gloria_m.hold_torque_nm, 0.15)

    def test_gloria_dual_settings_merge_each_side_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """gloria_m_dual:
  rate_hz: 40.0
  leader_read_timeout_s: 0.02
  stiffness_nm_per_rad: 1.5
  damping_nm_s_per_rad: 0.4
  contact_detection_enabled: false
  contact_torque_nm: 0.4
  contact_stall_duration_s: 0.2
  contact_position_tolerance_rad: 0.02
  hold_torque_nm: 0.15
  contact_release_hysteresis_rad: 0.06
  left:
    enabled: true
    port: /dev/ttyUSB2
  right:
    enabled: true
    port: /dev/ttyUSB3
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.gloria_m_dual.rate_hz, 40.0)
        self.assertEqual(runtime.gloria_m_dual.leader_read_timeout_s, 0.02)
        self.assertEqual(runtime.gloria_m_dual.stiffness_nm_per_rad, 1.5)
        self.assertEqual(runtime.gloria_m_dual.damping_nm_s_per_rad, 0.4)
        self.assertFalse(runtime.gloria_m_dual.contact_detection_enabled)
        self.assertEqual(runtime.gloria_m_dual.contact_torque_nm, 0.4)
        self.assertEqual(runtime.gloria_m_dual.contact_stall_duration_s, 0.2)
        self.assertEqual(runtime.gloria_m_dual.contact_position_tolerance_rad, 0.02)
        self.assertEqual(runtime.gloria_m_dual.hold_torque_nm, 0.15)
        self.assertEqual(runtime.gloria_m_dual.contact_release_hysteresis_rad, 0.06)
        self.assertTrue(runtime.gloria_m_dual.left.enabled)
        self.assertEqual(runtime.gloria_m_dual.left.port, "/dev/ttyUSB2")
        self.assertTrue(runtime.gloria_m_dual.right.enabled)
        self.assertEqual(runtime.gloria_m_dual.right.port, "/dev/ttyUSB3")
        for side in ("left", "right"):
            gripper_config = runtime.gloria_m_dual.side_config(side)
            self.assertEqual(gripper_config.stiffness_nm_per_rad, 1.5)
            self.assertEqual(gripper_config.damping_nm_s_per_rad, 0.4)
            self.assertFalse(gripper_config.contact_detection_enabled)
            self.assertEqual(gripper_config.contact_torque_nm, 0.4)
            self.assertEqual(gripper_config.contact_stall_duration_s, 0.2)
            self.assertEqual(gripper_config.contact_position_tolerance_rad, 0.02)
            self.assertEqual(gripper_config.hold_torque_nm, 0.15)
            self.assertEqual(gripper_config.contact_release_hysteresis_rad, 0.06)

    def test_gloria_dual_rejects_unknown_nested_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """gloria_m_dual:
  left:
    unknown_option: true
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gloria_m_dual.left"):
                load_runtime_config(path)

    def test_gloria_dual_rejects_side_specific_shared_control_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """gloria_m_dual:
  left:
    stiffness_nm_per_rad: 2.0
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "公共夹爪控制参数"):
                load_runtime_config(path)

    def test_openarm_mini_settings_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """openarm_mini:
  port_left: /dev/ttyUSB1
  port_right: /dev/ttyUSB0
  calibration_path: /tmp/openarm.json
  baudrate: 500000
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.openarm_mini.port_left, "/dev/ttyUSB1")
        self.assertEqual(runtime.openarm_mini.port_right, "/dev/ttyUSB0")
        self.assertEqual(runtime.openarm_mini.calibration_path, "/tmp/openarm.json")
        self.assertEqual(runtime.openarm_mini.baudrate, 500000)

    def test_hcx_settings_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """hcx:
  local_ip: 192.0.2.10
  remote_ip: 192.0.2.20
  port: 12345
  connect_timeout_s: 5.0
  left_robot_id: 4
  right_robot_id: 5
  left_axis_sign: [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
  right_axis_sign: [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
  direct_servo_rate_hz: 250
  direct_servo_watchdog_s: 0.3
  direct_servo_confirm_unsafe: false
  direct_servo_interpolation: linear
  auto_detach_hmi: true
  auto_clear_alarms: true
  auto_enable: true
  controller_initialization_wait_s: 3.0
  ethercat_master_indices: [0, 1]
  ethercat_op_timeout_s: 12.0
  alarm_clear_retry_count: 3
  alarm_clear_retry_interval_s: 0.5
  global_enable_retry_count: 4
  global_enable_retry_interval_s: 0.25
  single_arm_enable_timeout_s: 6.0
  enable_status_poll_interval_s: 0.2
""",
                encoding="utf-8",
            )

            runtime = load_runtime_config(path)

        self.assertEqual(runtime.hcx.local_ip, "192.0.2.10")
        self.assertEqual(runtime.hcx.remote_ip, "192.0.2.20")
        self.assertEqual(runtime.hcx.port, 12345)
        self.assertEqual(runtime.hcx.connect_timeout_s, 5.0)
        self.assertEqual((runtime.hcx.left_robot_id, runtime.hcx.right_robot_id), (4, 5))
        self.assertEqual(
            runtime.hcx.left_axis_sign,
            (1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
        )
        self.assertEqual(
            runtime.hcx.right_axis_sign,
            (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0),
        )
        self.assertEqual(runtime.hcx.direct_servo_rate_hz, 250)
        self.assertEqual(runtime.hcx.direct_servo_watchdog_s, 0.3)
        self.assertFalse(runtime.hcx.direct_servo_confirm_unsafe)
        self.assertEqual(runtime.hcx.direct_servo_interpolation, "linear")
        self.assertTrue(runtime.hcx.auto_detach_hmi)
        self.assertTrue(runtime.hcx.auto_clear_alarms)
        self.assertTrue(runtime.hcx.auto_enable)
        self.assertEqual(runtime.hcx.controller_initialization_wait_s, 3.0)
        self.assertEqual(runtime.hcx.ethercat_master_indices, (0, 1))
        self.assertEqual(runtime.hcx.ethercat_op_timeout_s, 12.0)
        self.assertEqual(runtime.hcx.alarm_clear_retry_count, 3)
        self.assertEqual(runtime.hcx.alarm_clear_retry_interval_s, 0.5)
        self.assertEqual(runtime.hcx.global_enable_retry_count, 4)
        self.assertEqual(runtime.hcx.global_enable_retry_interval_s, 0.25)
        self.assertEqual(runtime.hcx.single_arm_enable_timeout_s, 6.0)
        self.assertEqual(runtime.hcx.enable_status_poll_interval_s, 0.2)

    def test_hcx_limited_direct_servo_settings_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """hcx:
  direct_servo_interpolation: limited
  direct_servo_limited_max_vel_deg_s: 25.0
  direct_servo_limited_max_accel_deg_s2: 70.0
  direct_servo_limited_lowpass_alpha: 0.3
""",
                encoding="utf-8",
            )
            runtime = load_runtime_config(path)

        self.assertEqual(runtime.hcx.direct_servo_interpolation, "limited")
        self.assertEqual(runtime.hcx.direct_servo_limited_max_vel_deg_s, 25.0)
        self.assertEqual(runtime.hcx.direct_servo_limited_max_accel_deg_s2, 70.0)
        self.assertEqual(runtime.hcx.direct_servo_limited_lowpass_alpha, 0.3)

    def test_joint_count_is_not_a_yaml_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text("fr3:\n  joint_count: 7\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "禁止覆盖字段"):
                load_runtime_config(path)
