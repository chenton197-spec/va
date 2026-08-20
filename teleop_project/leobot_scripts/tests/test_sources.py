"""Tests for device-neutral collection sources."""

from __future__ import annotations

import unittest

import numpy as np

from leobot_scripts.sources import CallableGripperFeedbackSource


class CallableGripperFeedbackSourceTest(unittest.TestCase):
    def test_wraps_normalized_feedback_with_a_host_timestamp(self) -> None:
        source = CallableGripperFeedbackSource(lambda: 0.375)

        sample = source.read_gripper_opening()

        assert sample is not None
        np.testing.assert_array_equal(sample.value, np.array([0.375]))
        self.assertIsNotNone(sample.capture_monotonic_ns)

    def test_preserves_unavailable_feedback(self) -> None:
        source = CallableGripperFeedbackSource(lambda: None)

        self.assertIsNone(source.read_gripper_opening())


if __name__ == "__main__":
    unittest.main()
