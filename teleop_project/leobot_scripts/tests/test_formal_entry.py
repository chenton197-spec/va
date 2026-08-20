"""Hardware-free checks for the formal Alicia-D to FR3 recording entry."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from alicia_fr3_record import (
    AliciaFR3RecordingSession,
    SessionState,
    _check_disk_space,
    _resolve_dataset_root,
    _write_run_manifest,
)
from leobot_scripts.config import RecordingDeploymentConfig
from teleop_sdk.config import RuntimeConfig


def _recording_config(
    root: Path,
) -> RecordingDeploymentConfig:
    return RecordingDeploymentConfig(
        root=root,
        fps=30,
        task="move object",
        numeric_sample_fps=60,
        min_free_disk_gb=1.0,
    )


class FormalRecordingEntryTest(unittest.TestCase):
    def test_dataset_root_is_persistent_across_program_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configured_root = Path(directory) / "datasets" / "alicia_fr3"
            root = _resolve_dataset_root(_recording_config(configured_root))

            self.assertEqual(root, configured_root.resolve())
            self.assertFalse(root.exists())
            root.mkdir(parents=True)
            self.assertEqual(_resolve_dataset_root(_recording_config(configured_root)), root)

    def test_dataset_root_rejects_an_existing_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset-file"
            root.write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(NotADirectoryError, "不是目录"):
                _resolve_dataset_root(_recording_config(root))

    def test_disk_check_uses_nearest_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created" / "run_001"
            with patch(
                "alicia_fr3_record.shutil.disk_usage",
                return_value=SimpleNamespace(free=2 * 1024**3),
            ) as disk_usage:
                _check_disk_space(target, 1.0)

            self.assertEqual(disk_usage.call_args.args[0], Path(directory))

    def test_run_manifest_preserves_the_initial_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            original = root / "meta" / "run_config.json"
            original.parent.mkdir()
            original.write_text('{"original": true}\n', encoding="utf-8")

            path = _write_run_manifest(
                root,
                RuntimeConfig(),
                _recording_config(root),
                (),
            )

            self.assertEqual(json.loads(original.read_text(encoding="utf-8")), {"original": True})
            self.assertEqual(path.parent, root / "meta" / "runs")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording"]["root"], str(root))
            self.assertEqual(manifest["orbbec_cameras"], [])

    def test_q_requests_control_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = AliciaFR3RecordingSession(
                RuntimeConfig(),
                _recording_config(Path(directory) / "dataset"),
                (),
                Path(directory) / "dataset",
            )
            session._state = SessionState.TELEOP

            @contextmanager
            def fake_terminal():
                yield 42

            stop_requested = threading.Event()
            with (
                patch("alicia_fr3_record._single_key_terminal", fake_terminal),
                patch("alicia_fr3_record.select.select", return_value=([42], [], [])),
                patch("alicia_fr3_record.os.read", return_value=b"q"),
            ):
                session.run_interactive(stop_requested)

            self.assertTrue(stop_requested.is_set())
            self.assertTrue(session._stop_event.is_set())

    def test_shutdown_closes_recorder_before_hardware(self) -> None:
        events: list[str] = []

        class FakeRecorder:
            active = True

            def close(self) -> None:
                events.append("recorder.close")

        class FakeFinalizer:
            busy = False

            def close(self) -> None:
                events.append("finalizer.close")

        class FakeController:
            def shutdown(self) -> None:
                events.append("controller.shutdown")

        with tempfile.TemporaryDirectory() as directory:
            session = AliciaFR3RecordingSession(
                RuntimeConfig(),
                _recording_config(Path(directory) / "dataset"),
                (),
                Path(directory) / "dataset",
            )
            session._recorder = FakeRecorder()  # type: ignore[assignment]
            session._finalizer = FakeFinalizer()  # type: ignore[assignment]
            session._controller = FakeController()  # type: ignore[assignment]

            session.shutdown()

        self.assertEqual(events, ["recorder.close", "finalizer.close", "controller.shutdown"])
        self.assertEqual(session.state, SessionState.SHUTDOWN)


if __name__ == "__main__":
    unittest.main()
