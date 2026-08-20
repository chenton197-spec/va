from __future__ import annotations

import threading
import unittest

from leobot_scripts.episode_ops import AsyncEpisodeFinalizer, EpisodeOperation


class _BlockingRecorder:
    def __init__(self) -> None:
        self.active = True
        self.finish_started = threading.Event()
        self.release_finish = threading.Event()
        self.discarded = False

    def stop_episode(self) -> int:
        self.finish_started.set()
        self.release_finish.wait(timeout=2.0)
        self.active = False
        return 7

    def discard_episode(self) -> None:
        self.discarded = True
        self.active = False


class AsyncEpisodeFinalizerTest(unittest.TestCase):
    def test_finish_does_not_block_and_reports_completion_once(self) -> None:
        recorder = _BlockingRecorder()
        finalizer = AsyncEpisodeFinalizer(recorder)  # type: ignore[arg-type]
        try:
            finalizer.finish()
            self.assertTrue(recorder.finish_started.wait(timeout=1.0))
            self.assertTrue(finalizer.busy)
            self.assertIsNone(finalizer.poll())
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                finalizer.discard()

            recorder.release_finish.set()
            result = finalizer.wait(timeout_s=1.0)

            assert result is not None
            self.assertTrue(result.succeeded)
            self.assertEqual(result.operation, EpisodeOperation.FINISH)
            self.assertEqual(result.episode_index, 7)
            self.assertFalse(finalizer.busy)
        finally:
            finalizer.close()

    def test_discard_runs_asynchronously(self) -> None:
        recorder = _BlockingRecorder()
        finalizer = AsyncEpisodeFinalizer(recorder)  # type: ignore[arg-type]
        try:
            finalizer.discard()
            result = finalizer.wait(timeout_s=1.0)

            assert result is not None
            self.assertTrue(result.succeeded)
            self.assertEqual(result.operation, EpisodeOperation.DISCARD)
            self.assertTrue(recorder.discarded)
        finally:
            finalizer.close()


if __name__ == "__main__":
    unittest.main()
