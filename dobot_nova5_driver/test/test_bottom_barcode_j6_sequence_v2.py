import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).parents[1]
    / "dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v2.py"
)


def load_method(name):
    module = ast.parse(SOURCE.read_text())
    node_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CosmeticBoxSingleArmNode"
    )
    method = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[name]


RETURN_ONE_FACE = load_method("_return_j6_before_bottom_recovery")
REVERSE_REMAINDER = load_method("_reverse_barcode_search_j6")
TRANSFER_BARCODE_GRACE = load_method("_barcode_after_transfer_grace")


@dataclass
class Pose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


def load_table_pose_method():
    module = ast.parse(SOURCE.read_text())
    node_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CosmeticBoxSingleArmNode"
    )
    method = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_bottom_flip_table_pose"
    )
    namespace = {"TcpPose": Pose}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_bottom_flip_table_pose"]


TABLE_POSE = load_table_pose_method()


class FakeController:
    def __init__(self):
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, -255.7]
        self.commands = []

    def current_joint(self):
        return list(self.joints)

    def move_joint(self, joints, **_kwargs):
        self.joints = list(joints)
        self.commands.append(list(joints))


def make_node():
    parameters = {
        "bottom_flip_j6_pre_return_deg": 90.0,
        "barcode_flip_watch_joint_index": 5,
        "barcode_flip_jog_tolerance_deg": 1.0,
    }
    controller = FakeController()
    node = SimpleNamespace(
        barcode_search_net_delta_deg=-270.0,
        controller=controller,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        _is_barcode_flip_joint_safe=lambda _joints, _delta: True,
        _motion_profile=lambda: {
            "barcode_alignment_speed": 100,
            "barcode_alignment_acc": 100,
        },
        _publish_status=lambda _message: None,
        _require_cycle_active=lambda _stage: None,
    )
    node._record_barcode_search_travel = lambda delta: setattr(
        node,
        "barcode_search_net_delta_deg",
        node.barcode_search_net_delta_deg + delta,
    )
    return node


def test_failed_270_sweep_returns_90_then_reverses_remaining_180():
    node = make_node()

    RETURN_ONE_FACE(node)
    assert node.controller.commands[0][5] == pytest.approx(-165.7)
    assert node.barcode_search_net_delta_deg == pytest.approx(-180.0)

    REVERSE_REMAINDER(node)
    assert node.controller.commands[1][5] == pytest.approx(14.3)


def test_pre_return_is_before_scanner_retreat_in_cycle():
    source = SOURCE.read_text()
    start = source.index('with self._timed_stage("barcode_acquisition")')
    end = source.index('if bottom_recovery:', source.index('"scanner_retreat"', start))
    section = source[start:end]
    assert section.index('"bottom_flip_j6_pre_return"') < section.index(
        '"scanner_retreat"'
    )


def test_bottom_regrasp_uses_dedicated_160mm_lift_parameter():
    source = SOURCE.read_text()
    start = source.index('with self._timed_stage("bottom_flip_lift")')
    end = source.index('with self._timed_stage("bottom_flip_j6_reverse")', start)
    assert 'get_parameter("bottom_flip_lift_m")' in source[start:end]


def test_table_return_reverses_60mm_lift_but_retains_10mm_clearance():
    values = {
        "grasp_lift_m": 0.060,
        "bottom_flip_table_z_offset_m": 0.010,
        "minimum_safe_tcp_z_m": 0.010,
    }
    current = Pose(0.5, 0.2, 0.200, 180.0, 0.0, -90.0)
    node = SimpleNamespace(
        _current_command_pose=lambda: current,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        get_logger=lambda: SimpleNamespace(warning=lambda _message: None),
    )

    target = TABLE_POSE(node)

    assert target.z == pytest.approx(0.150)
    assert (target.x, target.y, target.rx, target.ry, target.rz) == (
        current.x,
        current.y,
        current.rx,
        current.ry,
        current.rz,
    )


def test_table_return_never_commands_below_minimum_tcp_z():
    values = {
        "grasp_lift_m": 0.060,
        "bottom_flip_table_z_offset_m": 0.010,
        "minimum_safe_tcp_z_m": 0.010,
    }
    node = SimpleNamespace(
        _current_command_pose=lambda: Pose(0.5, 0.2, 0.040, 180.0, 0.0, -90.0),
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        get_logger=lambda: SimpleNamespace(warning=lambda _message: None),
    )

    assert TABLE_POSE(node).z == pytest.approx(0.010)


def test_table_ry_flip_uses_linear_tcp_to_hold_xyz():
    source = SOURCE.read_text()
    start = source.index("    def _try_bottom_flip_ry_at_table")
    end = source.index("    def _lower_to_dynamic_placement_z", start)
    section = source[start:end]
    assert "self.controller.move_linear_tcp(" in section
    assert "self.controller.move_joint_tcp(" not in section


def test_bottom_fixed_place_applies_additional_user_ry_rotation():
    source = SOURCE.read_text()
    start = source.index('with self._timed_stage("bottom_barcode_fixed_place_ptp")')
    end = source.index('with self._timed_stage("bottom_barcode_gripper_open_place")', start)
    section = source[start:end]
    assert 'get_parameter("bottom_barcode_place_ry_delta_deg")' in source
    assert "ry_delta_deg=place_ry_delta_deg" in section


def test_side_barcode_path_combines_safe_fixed_place_and_rx_tilt():
    source = SOURCE.read_text()
    start = source.index('with self._timed_stage("post_scan_safe_height_ptp")')
    end = source.index('with self._timed_stage("placement_grasp_check")', start)
    section = source[start:end]
    assert 'get_parameter("side_barcode_place_rx_delta_deg")' in section
    assert '"side_barcode_rx_tilt"' not in section
    assert '"side_barcode_fixed_place_ptp"' in section
    assert "_side_barcode_place_xyz()" in section
    assert '"placement_vertical_descent"' not in section
    assert "rx_delta_deg=side_rx_delta_deg" in section
    assert "linear_tcp=True" not in section


def test_transfer_barcode_grace_catches_late_hid_callback_before_approach():
    waited = []
    statuses = []
    parameters = {
        "scanner_transfer_barcode_grace_s": 0.06,
        "barcode_stable_hits": 1,
    }
    node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        _current_stable_barcode=lambda: "",
        _publish_status=statuses.append,
        _wait_for_current_barcode=lambda hits, timeout, stage: (
            waited.append((hits, timeout, stage)) or "6941594520755"
        ),
    )

    assert TRANSFER_BARCODE_GRACE(node) == "6941594520755"
    assert waited == [(1, 0.06, "waiting for transfer-joint barcode callback")]
    assert any("transfer grace window" in status for status in statuses)


def test_transfer_barcode_grace_does_not_wait_when_barcode_is_already_stable():
    node = SimpleNamespace(
        _current_stable_barcode=lambda: "already-read",
        _wait_for_current_barcode=lambda *_args: pytest.fail("unexpected wait"),
    )

    assert TRANSFER_BARCODE_GRACE(node) == "already-read"
