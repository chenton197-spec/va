"""双 HCX RobotClient 只读反馈探测示例的无硬件测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import threading
import unittest
from unittest.mock import patch

from examples import test_hcx_two_clients_feedback as example


class _FakeRobotClient:
    instances: list["_FakeRobotClient"] = []
    owner: "_FakeRobotClient | None" = None
    allow_multiple_connections = False
    fail_construction_at: int | None = None

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.owner = None
        cls.allow_multiple_connections = False
        cls.fail_construction_at = None

    def __init__(self, local_ip: str, remote_ip: str, port: int) -> None:
        instance_index = len(self.__class__.instances) + 1
        if self.__class__.fail_construction_at == instance_index:
            raise RuntimeError("client construction failed")
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self.connected = False
        self.connect_timeouts: list[float | None] = []
        self.close_calls = 0
        self.__class__.instances.append(self)

    def connect(self, *, timeout_s: float | None = None) -> "_FakeRobotClient":
        self.connect_timeouts.append(timeout_s)
        owner = self.__class__.owner
        if (
            not self.__class__.allow_multiple_connections
            and owner is not None
            and owner is not self
        ):
            raise RuntimeError(
                "only one RobotClient may own the static HCX SDK in this process"
            )
        self.__class__.owner = self
        self.connected = True
        return self

    def close(self) -> None:
        self.close_calls += 1
        if self.__class__.owner is self:
            self.__class__.owner = None
        self.connected = False


class HcxTwoClientsFeedbackExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRobotClient.reset()
        self.config = replace(
            example.PROBE_CONFIG,
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            feedback_duration_s=0.1,
        )

    @staticmethod
    def _summary(label: str, robot_id: int) -> example.FeedbackSummary:
        return example.FeedbackSummary(
            label=label,
            robot_id=robot_id,
            sample_count=3,
            observed_rate_hz=30.0,
            maximum_gap_s=1.0 / 30.0,
            mean_read_duration_s=0.001,
            maximum_read_duration_s=0.002,
            latest_angles_deg=(float(robot_id),) * example.JOINT_COUNT,
        )

    def test_second_client_is_rejected_then_both_sides_read_sequentially(self) -> None:
        reads: list[tuple[str, int, bool]] = []

        def feedback_reader(
            client: _FakeRobotClient,
            robot_id: int,
            label: str,
            rate_hz: float,
            duration_s: float,
            report_interval_s: float,
        ) -> example.FeedbackSummary:
            del rate_hz, duration_s, report_interval_s
            reads.append((label, robot_id, client.connected))
            return self._summary(label, robot_id)

        result = example.run_probe(
            _FakeRobotClient,
            self.config,
            feedback_reader=feedback_reader,
        )

        self.assertFalse(result.concurrent_clients_connected)
        self.assertIn("only one RobotClient", result.second_client_rejection or "")
        self.assertEqual(
            reads,
            [
                ("LEFT", self.config.left_robot_id, True),
                ("RIGHT", self.config.right_robot_id, True),
            ],
        )
        left_client, right_client = _FakeRobotClient.instances
        self.assertEqual(left_client.port, self.config.left_port)
        self.assertEqual(right_client.port, self.config.right_port)
        self.assertEqual(left_client.connect_timeouts, [self.config.connect_timeout_s])
        self.assertEqual(
            right_client.connect_timeouts,
            [self.config.connect_timeout_s, self.config.connect_timeout_s],
        )
        self.assertEqual(left_client.close_calls, 1)
        self.assertEqual(right_client.close_calls, 1)

    def test_unexpected_second_connection_reads_both_sides_concurrently(self) -> None:
        _FakeRobotClient.allow_multiple_connections = True
        barrier = threading.Barrier(2)
        reads: list[str] = []

        def feedback_reader(
            client: _FakeRobotClient,
            robot_id: int,
            label: str,
            rate_hz: float,
            duration_s: float,
            report_interval_s: float,
        ) -> example.FeedbackSummary:
            del client, rate_hz, duration_s, report_interval_s
            barrier.wait(timeout=1.0)
            reads.append(label)
            return self._summary(label, robot_id)

        result = example.run_probe(
            _FakeRobotClient,
            self.config,
            feedback_reader=feedback_reader,
        )

        self.assertTrue(result.concurrent_clients_connected)
        self.assertIsNone(result.second_client_rejection)
        self.assertCountEqual(reads, ["LEFT", "RIGHT"])
        for client in _FakeRobotClient.instances:
            self.assertEqual(client.close_calls, 1)

    def test_feedback_failure_closes_all_created_clients(self) -> None:
        def failing_feedback_reader(
            client: _FakeRobotClient,
            robot_id: int,
            label: str,
            rate_hz: float,
            duration_s: float,
            report_interval_s: float,
        ) -> example.FeedbackSummary:
            del client, robot_id, label, rate_hz, duration_s, report_interval_s
            raise RuntimeError("joint feedback unavailable")

        with self.assertRaisesRegex(RuntimeError, "joint feedback unavailable"):
            example.run_probe(
                _FakeRobotClient,
                self.config,
                feedback_reader=failing_feedback_reader,
            )

        self.assertEqual(len(_FakeRobotClient.instances), 2)
        for client in _FakeRobotClient.instances:
            self.assertEqual(client.close_calls, 1)

    def test_second_client_construction_failure_closes_first_client(self) -> None:
        _FakeRobotClient.fail_construction_at = 2

        with self.assertRaisesRegex(RuntimeError, "client construction failed"):
            example.run_probe(_FakeRobotClient, self.config)

        self.assertEqual(len(_FakeRobotClient.instances), 1)
        self.assertEqual(_FakeRobotClient.instances[0].close_calls, 1)

    def test_config_rejects_duplicate_robot_ids(self) -> None:
        invalid = replace(
            self.config,
            right_robot_id=self.config.left_robot_id,
        )

        with self.assertRaisesRegex(ValueError, "必须不同"):
            invalid.validate()

    def test_config_rejects_duplicate_ports(self) -> None:
        invalid = replace(
            self.config,
            right_port=self.config.left_port,
        )

        with self.assertRaisesRegex(ValueError, "left_port 与 right_port 必须不同"):
            invalid.validate()

    def test_main_uses_top_level_constants_without_yaml_or_cli(self) -> None:
        result = example.ProbeResult(
            second_client_rejection="RuntimeError: only one RobotClient",
            left_feedback=self._summary("LEFT", self.config.left_robot_id),
            right_feedback=self._summary("RIGHT", self.config.right_robot_id),
        )
        output = StringIO()
        with (
            patch.object(example, "_load_robot_client", return_value=_FakeRobotClient),
            patch.object(example, "run_probe", return_value=result) as run_probe,
            redirect_stdout(output),
        ):
            exit_code = example.main()

        self.assertEqual(exit_code, 0)
        run_probe.assert_called_once_with(_FakeRobotClient, example.PROBE_CONFIG)
        self.assertIn("按预期拒绝", output.getvalue())
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
