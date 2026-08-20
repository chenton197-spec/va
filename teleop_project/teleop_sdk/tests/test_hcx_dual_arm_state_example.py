"""HCX 双臂只读状态示例的无硬件测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from examples import test_hcx_dual_arm_state as state_example
from teleop_sdk.config import HcxConfig, RuntimeConfig


class _FakeArm:
    def __init__(self, robot_id: int, angles_deg: tuple[float, ...]) -> None:
        self.robot_id = robot_id
        self._angles_deg = angles_deg
        self._torque_feedback = tuple(range(robot_id, robot_id + len(angles_deg)))
        self.enabled = True
        self.protection_enabled = True
        self.fail_joint_read = False

    @property
    def axis_count(self) -> int:
        return len(self._angles_deg)

    def joint_angles(self) -> tuple[float, ...]:
        if self.fail_joint_read:
            raise RuntimeError("joint feedback unavailable")
        return self._angles_deg

    def joint_torque_feedback(self) -> tuple[int, ...]:
        return self._torque_feedback


class _FakeRobotClient:
    instances: list["_FakeRobotClient"] = []
    fail_joint_read_for: int | None = None

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.fail_joint_read_for = None

    def __init__(self, local_ip: str, remote_ip: str, port: int) -> None:
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self.connected = False
        self.connect_timeouts: list[float | None] = []
        self.close_calls = 0
        self.global_enabled = True
        self.active_alarms = ("test alarm",)
        self.soft_emergency_stop_normal = True
        self.hmi_detached = False
        self.ethercat_calls: list[int] = []
        self.arms = {
            2: _FakeArm(2, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
            1: _FakeArm(1, (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0)),
        }
        if self.fail_joint_read_for is not None:
            self.arms[self.fail_joint_read_for].fail_joint_read = True
        self.instances.append(self)

    def connect(self, *, timeout_s: float | None = None) -> "_FakeRobotClient":
        self.connect_timeouts.append(timeout_s)
        self.connected = True
        return self

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False

    def arm(self, robot_id: int) -> _FakeArm:
        return self.arms[robot_id]

    def ethercat_master_operational(self, master_index: int) -> bool:
        self.ethercat_calls.append(master_index)
        return master_index == 0


class HcxDualArmStateExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRobotClient.reset()
        self.runtime = RuntimeConfig(
            hcx=HcxConfig(
                local_ip="192.0.2.10",
                remote_ip="192.0.2.20",
                port=12345,
                connect_timeout_s=5.0,
                left_robot_id=2,
                right_robot_id=1,
                ethercat_master_indices=(0,),
            )
        )

    def _run(self) -> tuple[int, str]:
        output = StringIO()
        with (
            patch.object(state_example, "load_runtime_config", return_value=self.runtime),
            patch.object(
                state_example, "_load_robot_client", return_value=_FakeRobotClient
            ),
            redirect_stdout(output),
        ):
            return state_example.main(), output.getvalue()

    def test_reads_both_arms_without_sending_control_commands(self) -> None:
        result, output = self._run()

        self.assertEqual(result, 0)
        client = _FakeRobotClient.instances[0]
        self.assertEqual(client.connect_timeouts, [5.0])
        self.assertEqual(client.ethercat_calls, [0])
        self.assertEqual(client.close_calls, 1)
        payload = json.loads(output.split("\n", 1)[1])
        self.assertEqual(payload["机械臂"]["左臂"]["关节角度（度）"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(payload["机械臂"]["右臂"]["关节角度（度）"], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
        self.assertEqual(payload["机械臂"]["左臂"]["关节力矩反馈（原始值）"], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(payload["EtherCAT 主站 OP 状态"], {"主站 0": True})

    def test_invalid_configuration_returns_before_loading_hcx_sdk(self) -> None:
        self.runtime = RuntimeConfig(hcx=HcxConfig())
        output = StringIO()

        with (
            patch.object(state_example, "load_runtime_config", return_value=self.runtime),
            patch.object(state_example, "_load_robot_client") as load_robot_client,
            redirect_stdout(output),
        ):
            result = state_example.main()

        self.assertEqual(result, 2)
        load_robot_client.assert_not_called()
        self.assertEqual(_FakeRobotClient.instances, [])

    def test_read_failure_closes_the_connection(self) -> None:
        _FakeRobotClient.fail_joint_read_for = 1

        result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(_FakeRobotClient.instances[0].close_calls, 1)

    def test_duplicate_robot_ids_are_rejected_before_connecting(self) -> None:
        self.runtime = RuntimeConfig(
            hcx=HcxConfig(
                local_ip="192.0.2.10",
                remote_ip="192.0.2.20",
                left_robot_id=2,
                right_robot_id=2,
            )
        )
        output = StringIO()

        with (
            patch.object(state_example, "load_runtime_config", return_value=self.runtime),
            patch.object(state_example, "_load_robot_client") as load_robot_client,
            redirect_stdout(output),
        ):
            result = state_example.main()

        self.assertEqual(result, 2)
        load_robot_client.assert_not_called()
