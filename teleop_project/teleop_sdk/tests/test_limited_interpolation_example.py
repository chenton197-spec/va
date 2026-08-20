"""真实 HCX LimitedInterpolator 示例的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
from queue import SimpleQueue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_limited_interpolation as example
from teleop_sdk.algorithms import LimitedInterpolator as _RealLimitedInterpolator
from teleop_sdk.config import HcxConfig


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration_s: float) -> None:
        self.value += max(0.0, duration_s)


class _FakeStopEvent:
    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, duration_s: float) -> bool:
        self._clock.sleep(duration_s)
        return self._set


class _FakeSession:
    def __init__(self) -> None:
        self.targets: list[list[float]] = []
        self.stop_calls = 0
        self.state = SimpleNamespace(
            running=True,
            faulted=False,
            sent_count=11,
            error=None,
        )

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def set_target(self, target: list[float]) -> None:
        self.targets.append(target)

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeArm:
    def __init__(self) -> None:
        self.axis_count = 7
        self.joint_limits_deg = tuple(((-170.0, 170.0) for _ in range(7)))
        self.protection_enabled = False
        self.session = _FakeSession()
        self.start_calls: list[dict[str, object]] = []
        self.protection_calls: list[bool] = []

    def joint_angles(self) -> tuple[float, ...]:
        return (0.0,) * 7

    def start_direct_servo(self, **kwargs: object) -> _FakeSession:
        self.start_calls.append(kwargs)
        return self.session

    def set_protection(self, enabled: bool, **kwargs: object) -> None:
        self.protection_calls.append(enabled)
        self.protection_enabled = enabled


class _FakeConnection:
    instances: list["_FakeConnection"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.arm = _FakeArm()
        self.acquired: list[int] = []
        self.prepared: list[int] = []
        self.released: list[int] = []
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def acquire(self, robot_id: int) -> _FakeArm:
        self.acquired.append(robot_id)
        return self.arm

    def prepare_for_motion(self, robot_id: int) -> bool:
        self.prepared.append(robot_id)
        return True

    def release(self, robot_id: int) -> None:
        self.released.append(robot_id)


class _CountingLimitedInterpolator:
    """保留真实算法，同时记录每次生成整批轨迹的调用。"""

    instances: list["_CountingLimitedInterpolator"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._delegate = _RealLimitedInterpolator(*args, **kwargs)
        self.interpolate_call_count = 0
        type(self).instances.append(self)

    @classmethod
    def reset_instances(cls) -> None:
        cls.instances = []

    def reset(self, initial_angles_deg: np.ndarray) -> None:
        self._delegate.reset(initial_angles_deg)

    def interpolate(self, target_angles_deg: np.ndarray) -> np.ndarray:
        self.interpolate_call_count += 1
        return self._delegate.interpolate(target_angles_deg)


class LimitedInterpolationExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeConnection.reset()
        _CountingLimitedInterpolator.reset_instances()
        self.hcx_config = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=1,
            direct_servo_rate_hz=500,
            direct_servo_watchdog_s=0.2,
            direct_servo_confirm_unsafe=True,
            direct_servo_interpolation="limited",
            direct_servo_limited_max_vel_deg_s=20.0,
            direct_servo_limited_max_accel_deg_s2=80.0,
            direct_servo_limited_lowpass_alpha=0.25,
        )

    def test_500_hz_sender_only_sends_precomputed_points_or_holds_last(self) -> None:
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        initial = np.zeros(example.JOINT_COUNT)
        raw_target = np.zeros(example.JOINT_COUNT)
        command_batch = np.zeros((5, example.JOINT_COUNT))
        command_batch[:, 3] = np.arange(1.0, 6.0)
        mailbox = example._LatestTrajectoryBatch()
        mailbox.publish(raw_target, command_batch)
        recorder = example._SenderRecorder.create(initial)
        session = _FakeSession()
        failures: SimpleQueue[BaseException] = SimpleQueue()

        # 发送函数不接收 LimitedInterpolator；补丁会在未来错误地把算法放入
        # 500 Hz 热循环时立即使测试失败。
        with patch.object(
            example,
            "LimitedInterpolator",
            side_effect=AssertionError("500 Hz sender must not run the algorithm"),
        ):
            example._run_fixed_rate_sender(
                session,
                mailbox,
                initial,
                500,
                stop_event,
                failures,
                recorder,
                monotonic=clock.monotonic,
                max_ticks=10,
            )

        trace = recorder.trace()
        self.assertEqual(len(session.targets), 10)
        self.assertTrue(failures.empty())
        np.testing.assert_allclose(np.diff(trace.timestamps_s), 0.002)
        self.assertEqual(recorder.deadline_miss_count, 0)
        np.testing.assert_array_equal(
            trace.precomputed_target_mask,
            [True, True, True, True, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trace.limited_batch_ids,
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        )
        np.testing.assert_allclose(
            trace.command_targets_deg[:, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        )

    def test_generates_limited_batches_outside_the_500_hz_sender(self) -> None:
        config = example.DemoConfig(
            side="right",
            joint_number=4,
            step_deg=1.0,
            step_hold_seconds=0.05,
            return_settle_seconds=0.05,
            disable_protection=False,
        )
        with (
            patch.object(example, "HcxConnection", _FakeConnection),
            patch.object(
                example,
                "LimitedInterpolator",
                _CountingLimitedInterpolator,
            ),
        ):
            report = example.run_demo(self.hcx_config, config)

        connection = _FakeConnection.instances[0]
        self.assertEqual(connection.acquired, [1])
        self.assertEqual(connection.prepared, [1])
        self.assertEqual(connection.released, [1])
        self.assertEqual(connection.arm.start_calls, [
            {"rate_hz": 500, "watchdog_s": 0.2, "confirm_unsafe": True}
        ])
        self.assertEqual(report.rate_hz, 500)
        self.assertEqual(report.limited_update_rate_hz, 100)
        self.assertGreater(report.submitted_count, 1)
        self.assertEqual(report.planned_command_count, 50)
        self.assertEqual(report.submitted_count, report.planned_command_count)
        self.assertEqual(
            report.submitted_count,
            len(connection.arm.session.targets),
        )
        self.assertEqual(len(_CountingLimitedInterpolator.instances), 1)
        self.assertEqual(
            _CountingLimitedInterpolator.instances[0].interpolate_call_count,
            report.limited_update_count,
        )
        self.assertLess(report.limited_update_count, report.submitted_count)
        self.assertTrue(np.any(report.trace.precomputed_target_mask))
        np.testing.assert_allclose(
            np.asarray(connection.arm.session.targets),
            report.trace.command_targets_deg,
        )
        self.assertEqual(report.trace.command_targets_deg.shape[1], 7)
        tested_joint_index = config.joint_index
        non_tested_axes = tuple(
            axis for axis in range(example.JOINT_COUNT) if axis != tested_joint_index
        )
        np.testing.assert_allclose(
            report.trace.raw_targets_deg[:, non_tested_axes], 0.0
        )
        np.testing.assert_allclose(
            report.trace.command_targets_deg[:, non_tested_axes], 0.0
        )
        self.assertTrue(
            np.any(report.trace.raw_targets_deg[:, tested_joint_index] == 1.0)
        )
        self.assertGreater(
            np.max(report.trace.command_targets_deg[:, tested_joint_index]), 0.0
        )
        self.assertGreater(len(connection.arm.session.targets), 1)
        self.assertGreaterEqual(connection.arm.session.stop_calls, 1)

    def test_rejects_non_500_hz_before_connecting(self) -> None:
        invalid = replace(self.hcx_config, direct_servo_rate_hz=250)
        with patch.object(example, "HcxConnection", _FakeConnection):
            with self.assertRaisesRegex(ValueError, "固定以 500 Hz"):
                example.run_demo(
                    invalid,
                    example.DemoConfig(
                        side="right",
                        joint_number=3,
                        step_deg=1.0,
                        step_hold_seconds=1.0,
                        return_settle_seconds=0.0,
                        disable_protection=False,
                    ),
                )

        self.assertEqual(_FakeConnection.instances, [])

    def test_rejects_non_limited_hcx_mode_before_connecting(self) -> None:
        invalid = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            direct_servo_confirm_unsafe=True,
            direct_servo_interpolation="direct",
        )
        with patch.object(example, "HcxConnection", _FakeConnection):
            with self.assertRaisesRegex(ValueError, "LimitedInterpolator"):
                example.run_demo(
                    invalid,
                    example.DemoConfig(
                        side="right",
                        joint_number=3,
                        step_deg=1.0,
                        step_hold_seconds=1.0,
                        return_settle_seconds=0.0,
                        disable_protection=False,
                    ),
                )

        self.assertEqual(_FakeConnection.instances, [])

    def test_rejects_zero_step_before_connecting(self) -> None:
        with patch.object(example, "HcxConnection", _FakeConnection):
            with self.assertRaisesRegex(ValueError, "step_deg"):
                example.run_demo(
                    self.hcx_config,
                    example.DemoConfig(
                        side="right",
                        joint_number=3,
                        step_deg=0.0,
                        step_hold_seconds=1.0,
                        return_settle_seconds=1.0,
                        disable_protection=False,
                    ),
                )

        self.assertEqual(_FakeConnection.instances, [])

    def test_rejects_joint_number_outside_j1_to_j7_before_connecting(self) -> None:
        with patch.object(example, "HcxConnection", _FakeConnection):
            with self.assertRaisesRegex(ValueError, "joint_number"):
                example.run_demo(
                    self.hcx_config,
                    example.DemoConfig(
                        side="right",
                        joint_number=8,
                        step_deg=1.0,
                        step_hold_seconds=1.0,
                        return_settle_seconds=1.0,
                        disable_protection=False,
                    ),
                )

        self.assertEqual(_FakeConnection.instances, [])

    def test_main_uses_top_level_constants_without_yaml_or_cli_arguments(self) -> None:
        report = object()
        with (
            patch.object(example, "run_demo", return_value=report) as run_demo,
            patch.object(example, "_print_report") as print_report,
        ):
            result = example.main()

        self.assertEqual(result, 0)
        self.assertFalse(hasattr(example, "load_runtime_config"))
        run_demo.assert_called_once()
        hcx_config, config = run_demo.call_args.args
        self.assertEqual(hcx_config.local_ip, example.LOCAL_IP)
        self.assertEqual(hcx_config.remote_ip, example.REMOTE_IP)
        self.assertEqual(hcx_config.port, example.PORT)
        self.assertEqual(hcx_config.left_robot_id, example.LEFT_ARM_ID)
        self.assertEqual(hcx_config.right_robot_id, example.RIGHT_ARM_ID)
        self.assertEqual(hcx_config.direct_servo_rate_hz, example.DIRECT_SERVO_RATE_HZ)
        self.assertEqual(example.DIRECT_SERVO_RATE_HZ, 500)
        self.assertEqual(example.LIMITED_UPDATE_RATE_HZ, 100)
        self.assertEqual(
            hcx_config.direct_servo_interpolation,
            example.DIRECT_SERVO_INTERPOLATION,
        )
        self.assertEqual(config.side, example.TEST_SIDE)
        self.assertEqual(config.joint_number, example.TEST_JOINT_NUMBER)
        self.assertEqual(config.step_deg, example.TEST_STEP_DEG)
        self.assertEqual(config.step_hold_seconds, example.TEST_STEP_HOLD_SECONDS)
        self.assertEqual(
            config.return_settle_seconds,
            example.TEST_RETURN_SETTLE_SECONDS,
        )
        self.assertEqual(config.disable_protection, example.TEST_DISABLE_PROTECTION)
        print_report.assert_called_once_with(report)
