import pytest
from scipy.spatial.transform import Rotation as SciPyRot

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v4 import (
    select_nearest_long_edge_grasp_orientation,
)


@pytest.mark.parametrize(
    ("current_yaw_deg", "expected_offset_deg", "expected_travel_deg"),
    [(4.0, 0.0, 4.0), (176.0, 180.0, 4.0)],
)
def test_rectangular_grasp_uses_nearest_equivalent_long_edge(
    current_yaw_deg, expected_offset_deg, expected_travel_deg
):
    target = TcpPose(0.52, -0.18, 0.19, 180.0, 0.0, 0.0)
    base = SciPyRot.from_euler("xyz", [target.rx, target.ry, target.rz], degrees=True)
    current_rotation = base * SciPyRot.from_euler(
        "z", current_yaw_deg, degrees=True
    )
    current_rpy = current_rotation.as_euler("xyz", degrees=True)
    current = TcpPose(0.3, 0.1, 0.4, *map(float, current_rpy))

    chosen, travel_deg, offset_deg = select_nearest_long_edge_grasp_orientation(
        target, current
    )

    chosen_rotation = SciPyRot.from_euler(
        "xyz", [chosen.rx, chosen.ry, chosen.rz], degrees=True
    )
    assert offset_deg == expected_offset_deg
    assert travel_deg == pytest.approx(expected_travel_deg)
    assert (chosen.x, chosen.y, chosen.z) == (target.x, target.y, target.z)
    assert abs(float(base.apply([0, 1, 0]) @ chosen_rotation.apply([0, 1, 0]))) == pytest.approx(1.0)
    assert chosen_rotation.apply([0, 0, 1]) == pytest.approx(base.apply([0, 0, 1]))
