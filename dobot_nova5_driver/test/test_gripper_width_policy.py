import pytest

from dobot_nova5_driver.gripper_width_policy import select_gripper_width


DEFAULTS = {
    "clearance_m": 0.020,
    "max_opening_m": 0.095,
    "full_open_width_threshold_m": 0.060,
    "near_square_aspect_ratio": 1.2,
}


def decide(length_m, width_m):
    return select_gripper_width(length_m, width_m, **DEFAULTS)


def test_narrow_rectangular_box_uses_measured_width_plus_clearance():
    decision = decide(0.120, 0.050)

    assert decision.command_m == pytest.approx(0.070)
    assert not decision.full_opening
    assert decision.reason == "measured_plus_clearance"


def test_width_exactly_sixty_millimeters_keeps_clearance_policy():
    decision = decide(0.120, 0.060)

    assert decision.command_m == pytest.approx(0.080)
    assert not decision.full_opening


def test_width_above_sixty_millimeters_uses_full_opening():
    decision = decide(0.120, 0.061)

    assert decision.command_m == pytest.approx(0.095)
    assert decision.full_opening
    assert decision.reason == "wide"


def test_near_square_narrow_box_uses_full_opening():
    decision = decide(0.055, 0.050)

    assert decision.command_m == pytest.approx(0.095)
    assert decision.full_opening
    assert decision.aspect_ratio == pytest.approx(1.1)
    assert decision.reason == "near_square"


def test_aspect_ratio_exactly_one_point_two_does_not_trigger_full_opening():
    decision = decide(0.060, 0.050)

    assert decision.command_m == pytest.approx(0.070)
    assert not decision.full_opening
