"""HCX 简易上位机后端的无硬件测试；不创建 Tk 窗口。"""

from __future__ import annotations

import unittest

from examples import hcx_follower_gui as example
from teleop_sdk.config import HcxConfig


class _FakeMotion:
    def __init__(self, sequence: int) -> None:
        self.sequence = sequence


class _FakeArm:
    def __init__(self, robot_id: int, sign: float) -> None:
        self.robot_id = robot_id
        self.enabled = True
        self.protection_enabled = True
        self._angles = tuple(sign * float(index) for index in range(1, 8))
        self._torques = tuple(range(robot_id, robot_id + 7))
        self.joint_limits_deg = tuple((-170.0, 170.0) for _ in range(7))
        self.move_calls: list[tuple[tuple[float, ...], dict[str, object]]] = []
        self.enabled_calls: list[bool] = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.clear_route_calls: list[bool] = []

    def joint_angles(self) -> tuple[float, ...]:
        return self._angles

    def joint_torque_feedback(self) -> tuple[int, ...]:
        return self._torques

    def move_joints(self, angles_deg: tuple[float, ...], **kwargs: object) -> _FakeMotion:
        self.move_calls.append((angles_deg, kwargs))
        return _FakeMotion(len(self.move_calls))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled_calls.append(enabled)
        self.enabled = enabled

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def clear_route(self, *, emergency_stop: bool) -> None:
        self.clear_route_calls.append(emergency_stop)


class _FakeRobotClient:
    instances: list["_FakeRobotClient"] = []

    def __init__(self, local_ip: str, remote_ip: str, port: int) -> None:
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self.connected = False
        self.link_status = True
        self.connect_timeouts: list[float | None] = []
        self.close_calls = 0
        self.global_enabled = False
        self.active_alarms = ("test alarm",)
        self.hmi_detached = False
        self.soft_emergency_stop_normal = True
        self.clear_alarm_calls = 0
        self.detach_hmi_calls = 0
        self.global_enable_calls: list[bool] = []
        self.ethercat_operational = {0: True, 1: True}
        self.arms = {2: _FakeArm(2, 1.0), 1: _FakeArm(1, -1.0)}
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def connect(self, *, timeout_s: float | None) -> "_FakeRobotClient":
        self.connect_timeouts.append(timeout_s)
        self.connected = True
        return self

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False

    def arm(self, robot_id: int) -> _FakeArm:
        return self.arms[robot_id]

    def ethercat_master_operational(self, master_index: int) -> bool:
        return self.ethercat_operational[master_index]

    def clear_alarms(self) -> None:
        self.clear_alarm_calls += 1
        self.active_alarms = ()

    def detach_hmi(self) -> None:
        self.detach_hmi_calls += 1
        self.hmi_detached = True

    def set_global_enable(self, enabled: bool) -> None:
        self.global_enable_calls.append(enabled)
        self.global_enabled = enabled


class HcxFollowerGuiBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRobotClient.reset()
        self.config = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            port=12345,
            connect_timeout_s=5.0,
            left_robot_id=2,
            right_robot_id=1,
            ethercat_master_indices=(0,),
        )

    def _backend(self) -> example.HcxFollowerBackend:
        return example.HcxFollowerBackend(
            self.config, robot_client_factory=_FakeRobotClient
        )

    def test_connect_reads_two_seven_axis_snapshots_without_gui_or_motion(self) -> None:
        backend = self._backend()

        snapshot = backend.connect()

        client = _FakeRobotClient.instances[0]
        self.assertEqual(client.connect_timeouts, [5.0])
        self.assertEqual(snapshot.arms[0].side, "left")
        self.assertEqual(snapshot.arms[0].angles_deg, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        self.assertEqual(snapshot.arms[1].angles_deg, (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0))
        self.assertEqual(snapshot.ethercat_operational, ((0, True),))
        self.assertEqual(client.arms[2].move_calls, [])
        self.assertEqual(client.arms[1].move_calls, [])

    def test_snapshot_rejects_a_controller_link_that_is_down(self) -> None:
        backend = self._backend()
        backend.connect()
        _FakeRobotClient.instances[0].link_status = False

        self.assertFalse(backend.link_healthy())
        with self.assertRaisesRegex(RuntimeError, "链路状态"):
            backend.snapshot()

    def test_enable_and_motion_prerequisites_follow_hcx_startup_order(self) -> None:
        backend = self._backend()
        initial = backend.connect()
        self.assertEqual(
            example._controller_enable_block_reason(initial),
            "示教器未脱离；存在报警",
        )
        self.assertEqual(
            example._arm_motion_block_reason(initial, initial.arms[0]),
            "示教器未脱离；存在报警；全局未使能",
        )

        client = _FakeRobotClient.instances[0]
        client.hmi_detached = True
        client.active_alarms = ()
        client.global_enabled = True
        client.arms[2].enabled = True
        ready = backend.snapshot()
        self.assertIsNone(example._controller_enable_block_reason(ready))
        self.assertIsNone(
            example._arm_motion_block_reason(ready, ready.arms[0])
        )

        client.ethercat_operational[0] = False
        ethercat_not_ready = backend.snapshot()
        self.assertEqual(
            example._controller_enable_block_reason(ethercat_not_ready),
            "EtherCAT 未进入 OP",
        )

    def test_planned_move_uses_explicit_motion_options_and_target(self) -> None:
        backend = self._backend()
        backend.connect()
        options = example.MotionOptions(
            speed_ratio=0.1,
            acceleration_seconds=0.5,
            deceleration_seconds=0.4,
            smooth=2,
        )

        motion = backend.move_arm("left", (10, 20, 30, 40, 50, 60, 70), options)

        left = _FakeRobotClient.instances[0].arms[2]
        self.assertEqual(motion.sequence, 1)
        self.assertEqual(
            left.move_calls,
            [
                (
                    (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
                    {
                        "interrupt": False,
                        "acceleration_seconds": 0.5,
                        "deceleration_seconds": 0.4,
                        "speed_ratio": 0.1,
                        "smooth": 2,
                        "wait": False,
                    },
                )
            ],
        )

    def test_positive_jog_moves_only_selected_axis_to_limit_margin(self) -> None:
        backend = self._backend()
        backend.connect()
        options = example.MotionOptions(
            speed_ratio=0.1,
            acceleration_seconds=0.5,
            deceleration_seconds=0.4,
            smooth=2,
        )

        motion, target = backend.jog_axis("left", 2, 1, options)

        left = _FakeRobotClient.instances[0].arms[2]
        self.assertEqual(motion.sequence, 1)
        self.assertEqual(target, (1.0, 2.0, 169.0, 4.0, 5.0, 6.0, 7.0))
        self.assertEqual(
            left.move_calls,
            [
                (
                    target,
                    {
                        "interrupt": False,
                        "acceleration_seconds": 0.5,
                        "deceleration_seconds": 0.4,
                        "speed_ratio": 0.1,
                        "smooth": 2,
                        "wait": False,
                    },
                )
            ],
        )

    def test_negative_jog_moves_only_selected_axis_to_limit_margin(self) -> None:
        backend = self._backend()
        backend.connect()
        options = example.MotionOptions(0.1, 0.5, 0.4, 2)

        _, target = backend.jog_axis("right", 4, -1, options)

        self.assertEqual(
            target, (-1.0, -2.0, -3.0, -4.0, -169.0, -6.0, -7.0)
        )

    def test_jog_rejects_invalid_axis_direction_and_boundary(self) -> None:
        options = example.MotionOptions(0.1, 0.5, 0.4, 2)

        with self.assertRaisesRegex(ValueError, "关节索引"):
            example._build_jog_target(
                (0.0,) * 7,
                ((-170.0, 170.0),) * 7,
                7,
                1,
            )
        with self.assertRaisesRegex(ValueError, "点动方向"):
            example._build_jog_target(
                (0.0,) * 7,
                ((-170.0, 170.0),) * 7,
                0,
                0,
            )

        backend = self._backend()
        backend.connect()
        left = _FakeRobotClient.instances[0].arms[2]
        left._angles = (169.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        with self.assertRaisesRegex(ValueError, "正向点动边界"):
            backend.jog_axis("left", 0, 1, options)
        self.assertEqual(left.move_calls, [])

    def test_control_buttons_map_to_the_matching_sdk_methods(self) -> None:
        backend = self._backend()
        backend.connect()

        backend.clear_alarms()
        backend.detach_hmi()
        backend.set_global_enabled(True)
        backend.set_arm_enabled("right", False)
        backend.pause_arm("right")
        backend.resume_arm("right")
        backend.clear_arm_route("right")

        client = _FakeRobotClient.instances[0]
        right = client.arms[1]
        self.assertEqual(client.clear_alarm_calls, 1)
        self.assertEqual(client.detach_hmi_calls, 1)
        self.assertEqual(client.global_enable_calls, [True])
        self.assertEqual(right.enabled_calls, [False])
        self.assertEqual(right.pause_calls, 1)
        self.assertEqual(right.resume_calls, 1)
        self.assertEqual(right.clear_route_calls, [True])

    def test_parse_helpers_reject_invalid_values_before_sdk_control(self) -> None:
        with self.assertRaisesRegex(ValueError, "七个"):
            example._parse_target_angles((0, 1, 2))
        with self.assertRaisesRegex(ValueError, "速度比例"):
            example._parse_motion_options("0", "0.5", "0.5", "1")
        with self.assertRaisesRegex(ValueError, "加速时间"):
            example._parse_motion_options("0.1", "0.01", "0.5", "1")

    def test_duplicate_robot_ids_fail_before_creating_a_client(self) -> None:
        invalid = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=2,
        )

        with self.assertRaisesRegex(ValueError, "必须不同"):
            example.HcxFollowerBackend(invalid, robot_client_factory=_FakeRobotClient)

        self.assertEqual(_FakeRobotClient.instances, [])


if __name__ == "__main__":
    unittest.main()
