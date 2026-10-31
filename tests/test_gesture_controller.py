"""Unit tests for the gesture interaction layer.

State-machine and arithmetic only: no webcam, no model bundle and no OpenCV
context are needed. ``GestureController`` owns model pose, so the tests drive
it through ``update_one_hand`` and ``update_two_hands`` and inspect the
resulting pose and mode.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.interaction.gesture_controller import GestureController, Mode

DT = 1.0 / 30.0
PALM = (0.5, 0.5)
TWO_HAND_STATS = {"midpoint": PALM, "span": 0.1, "num_hands": 2}


def two_hand_frames(controller: GestureController, count: int) -> None:
    """Feed ``count`` consecutive two-hand frames."""
    for _ in range(count):
        controller.update_two_hands(TWO_HAND_STATS, DT)


def absent_frames(controller: GestureController, count: int) -> None:
    """Feed ``count`` frames with no two-hand configuration."""
    for _ in range(count):
        controller.update_two_hands(None, DT)


class TestInitialState:
    def test_starts_idle_and_centred(self) -> None:
        controller = GestureController()
        assert controller.mode == Mode.IDLE
        assert controller.rot_x == 0.0
        assert controller.rot_y == 0.0
        assert controller.scale == 1.0
        assert np.allclose(controller.offset, 0.0)
        assert np.allclose(controller.angular_velocity, 0.0)


class TestSpinDecay:
    def test_idle_velocity_decays_geometrically(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        controller.angular_velocity = np.array([0.5, -0.3])
        dt = 0.02

        expected = controller.angular_velocity * math.exp(-dt / GestureController.TAU)
        controller.update_one_hand(PALM, False, dt)

        assert np.allclose(controller.angular_velocity, expected, atol=1e-12)

    def test_rotation_advances_while_spinning(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        controller.angular_velocity = np.array([0.4, 0.0])

        controller.update_one_hand(PALM, False, DT)
        controller.update_one_hand(PALM, False, DT)

        decay = math.exp(-DT / GestureController.TAU)
        expected_rot_x = 0.4 * DT * (decay + decay ** 2)
        assert np.isclose(controller.rot_x, expected_rot_x, atol=1e-12)

    def test_velocity_eventually_comes_to_rest(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        controller.angular_velocity = np.array([1.0, 1.0])
        for _ in range(200):
            controller.update_one_hand(PALM, False, DT)
        assert np.linalg.norm(controller.angular_velocity) < 1e-6


class TestGrabRotate:
    def test_grab_integrates_rotation(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        delta = np.array([0.02, -0.01])

        controller.update_one_hand((0.52, 0.49), True, DT)

        expected_velocity = delta * GestureController.ROTATION_SENSITIVITY
        assert np.allclose(controller.angular_velocity, expected_velocity, atol=1e-12)
        assert np.allclose(controller.rot_x, expected_velocity[0] * DT, atol=1e-12)
        assert np.allclose(controller.rot_y, expected_velocity[1] * DT, atol=1e-12)
        assert controller.mode == Mode.ROTATE

    def test_release_holds_last_rotation(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        controller.update_one_hand((0.52, 0.49), True, DT)
        before = controller.rot_x
        velocity = controller.angular_velocity.copy()

        controller.update_one_hand((0.52, 0.49), False, DT)

        assert controller.mode == Mode.IDLE
        # The first frame after release still advances the pose by the
        # friction-decayed velocity; the velocity is decayed afterwards.
        decay = math.exp(-DT / GestureController.TAU)
        expected = before + velocity[0] * decay * DT
        assert np.isclose(controller.rot_x, expected, atol=1e-12)

    def test_catch_re_sources_velocity(self) -> None:
        controller = GestureController()
        controller._last_palm = np.asarray(PALM)
        controller.angular_velocity = np.array([10.0, 10.0])
        for _ in range(60):
            controller.update_one_hand(PALM, False, DT)
        assert np.linalg.norm(controller.angular_velocity) < 0.1

        delta = np.array([0.01, -0.02])
        controller.update_one_hand((0.51, 0.48), True, DT)

        expected = delta * GestureController.ROTATION_SENSITIVITY
        assert np.allclose(controller.angular_velocity, expected, atol=1e-12)

    def test_first_grab_frame_has_no_prior_delta(self) -> None:
        # With no previous palm to compare against, the first grabbed frame
        # must not inject a spurious rotation.
        controller = GestureController()
        controller.update_one_hand(PALM, True, DT)
        assert np.allclose(controller.angular_velocity, 0.0, atol=1e-12)


class TestTwoHandHysteresis:
    def test_enter_requires_three_consecutive_frames(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 1)
        assert controller.mode != Mode.TRANSFORM
        two_hand_frames(controller, 1)
        assert controller.mode != Mode.TRANSFORM
        two_hand_frames(controller, 1)
        assert controller.mode == Mode.TRANSFORM

    def test_intermittent_two_hand_drops_the_count(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 2)
        absent_frames(controller, 1)
        two_hand_frames(controller, 1)
        assert controller.mode != Mode.TRANSFORM

    def test_exit_requires_five_consecutive_absent_frames(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        assert controller.mode == Mode.TRANSFORM
        for _ in range(4):
            absent_frames(controller, 1)
            assert controller.mode == Mode.TRANSFORM
        absent_frames(controller, 1)
        assert controller.mode != Mode.TRANSFORM

    def test_two_hand_returns_after_leaving(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        absent_frames(controller, 5)
        assert controller.mode != Mode.TRANSFORM
        two_hand_frames(controller, 3)
        assert controller.mode == Mode.TRANSFORM


class TestTwoHandTransform:
    def test_scale_clamps_at_the_upper_bound(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        for _ in range(5):
            controller.update_two_hands(
                {"midpoint": PALM, "span": 10.0, "num_hands": 2}, DT
            )
        assert np.isclose(controller.scale, controller.SCALE_MAX)

    def test_scale_clamps_at_the_lower_bound(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        controller.update_two_hands({"midpoint": PALM, "span": 10.0, "num_hands": 2}, DT)
        for _ in range(5):
            controller.update_two_hands(
                {"midpoint": PALM, "span": 1e-9, "num_hands": 2}, DT
            )
        assert np.isclose(controller.scale, controller.SCALE_MIN)

    def test_pan_moves_offset_by_midpoint_delta(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        previous = controller.offset.copy()
        controller.update_two_hands(
            {"midpoint": (0.55, 0.5), "span": 0.1, "num_hands": 2}, DT
        )
        expected = previous + np.array([0.05, 0.0]) * controller.PAN_SENSITIVITY
        assert np.allclose(controller.offset, expected, atol=1e-12)

    def test_steady_hands_hold_the_scale(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        controller.update_two_hands(TWO_HAND_STATS, DT)
        controller.update_two_hands(TWO_HAND_STATS, DT)
        assert np.isclose(controller.scale, 1.0, atol=1e-12)

    def test_one_hand_rotation_is_suspended_while_transforming(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        assert controller.mode == Mode.TRANSFORM
        previous_rot_x = controller.rot_x

        controller.update_one_hand((0.6, 0.5), True, DT)

        assert controller.rot_x == previous_rot_x
        assert controller.mode == Mode.TRANSFORM

    def test_transform_survives_a_one_hand_frame(self) -> None:
        controller = GestureController()
        two_hand_frames(controller, 3)
        controller.update_one_hand(PALM, True, DT)
        assert controller.mode == Mode.TRANSFORM


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        controller = GestureController()
        controller.rot_x = 1.0
        controller.rot_y = -1.0
        controller.scale = 2.0
        controller.offset = np.array([10.0, 20.0])
        controller.angular_velocity = np.array([3.0, -3.0])
        controller.mode = Mode.TRANSFORM
        controller._last_palm = np.asarray(PALM)

        controller.reset()

        assert controller.rot_x == 0.0
        assert controller.rot_y == 0.0
        assert controller.scale == 1.0
        assert np.allclose(controller.offset, 0.0)
        assert np.allclose(controller.angular_velocity, 0.0)
        assert controller.mode == Mode.IDLE
        assert controller._last_palm is None