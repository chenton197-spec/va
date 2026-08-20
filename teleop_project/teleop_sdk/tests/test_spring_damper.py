"""SpringDamper 的无硬件算法测试。"""

from __future__ import annotations

import unittest

import numpy as np

from teleop_sdk.algorithms import SpringDamper


def create_spring(n_joints: int) -> SpringDamper:
    """创建使用原默认参数的指定维度弹簧阻尼器。"""
    return SpringDamper(
        rate_hz=125.0,
        omega=12.0,
        jump_threshold_deg=180.0,
        max_accel_deg_s2=500.0,
        max_vel_deg_s=60.0,
        min_angles_deg=np.full(n_joints, -170.0),
        max_angles_deg=np.full(n_joints, 170.0),
    )


class SpringDamperTest(unittest.TestCase):
    """验证初始状态、重置和不同关节维度。"""

    def test_first_step_returns_initial_angles(self) -> None:
        spring = create_spring(6)
        initial = np.arange(6, dtype=float)

        result = spring.step(initial + 10.0, initial)

        np.testing.assert_array_equal(result, initial)

    def test_reset_uses_new_initial_angles(self) -> None:
        spring = create_spring(6)
        initial = np.zeros(6)
        spring.step(np.full(6, 10.0), initial)
        spring.step(np.full(6, 10.0), initial)
        spring.reset()

        result = spring.step(np.full(6, 20.0), np.full(6, 3.0))

        np.testing.assert_array_equal(result, np.full(6, 3.0))

    def test_supports_six_and_seven_joint_arrays(self) -> None:
        for n_joints in (6, 7):
            with self.subTest(n_joints=n_joints):
                spring = create_spring(n_joints)
                initial = np.zeros(n_joints)
                spring.step(np.ones(n_joints), initial)
                result = spring.step(np.ones(n_joints), initial)

                self.assertEqual(result.shape, (n_joints,))
                self.assertTrue(np.all(result > 0.0))

    def test_output_is_clipped_to_joint_limits(self) -> None:
        spring = SpringDamper(
            rate_hz=100.0,
            omega=12.0,
            jump_threshold_deg=180.0,
            max_accel_deg_s2=500.0,
            max_vel_deg_s=60.0,
            min_angles_deg=np.full(6, -1.0),
            max_angles_deg=np.full(6, 1.0),
        )
        initial = np.zeros(6)
        spring.step(np.full(6, 100.0), initial)

        result = initial
        for _ in range(100):
            result = spring.step(np.full(6, 100.0), initial)

        self.assertTrue(np.all(result <= 1.0))

    def test_glitch_freezes_only_affected_joint(self) -> None:
        spring = create_spring(2)
        initial = np.zeros(2)
        spring.step(np.zeros(2), initial)
        result = spring.step(np.array([999.0, 20.0]), initial)

        self.assertEqual(result[0], 0.0)
        self.assertGreater(result[1], 0.0)

    def test_prediction_is_limited(self) -> None:
        spring = SpringDamper(
            rate_hz=100.0,
            omega=20.0,
            jump_threshold_deg=180.0,
            max_accel_deg_s2=800.0,
            max_vel_deg_s=150.0,
            min_angles_deg=np.full(2, -1.0),
            max_angles_deg=np.full(2, 1.0),
        )
        spring.step(np.zeros(2), np.zeros(2))
        spring.step(np.ones(2), np.zeros(2))
        self.assertTrue(np.all(spring.predict(10.0) <= 1.0))


if __name__ == "__main__":
    unittest.main()
