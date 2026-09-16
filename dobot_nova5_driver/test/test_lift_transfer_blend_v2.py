from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from dobot_nova5_driver.controller_v2 import (
    DobotNova5Controller,
    MotionCommandHandle,
    TcpPose,
)
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v2 import (
    CosmeticBoxSingleArmNode,
)


class QueueDashboard:
    def __init__(self):
        self.calls = []
        self.next_id = 40

    def MovJ(self, *args, **kwargs):
        self.calls.append(("MovJ", args, kwargs))
        self.next_id += 1
        return f"0,{{{self.next_id}}},MovJ;"

    def SpeedFactor(self, value):
        self.calls.append(("SpeedFactor", (value,), {}))
        return "0,{},ok;"

    def AccJ(self, value):
        self.calls.append(("AccJ", (value,), {}))
        return "0,{},ok;"

    def VelJ(self, value):
        self.calls.append(("VelJ", (value,), {}))
        return "0,{},ok;"

    def AccL(self, value):
        self.calls.append(("AccL", (value,), {}))
        return "0,{},ok;"

    def VelL(self, value):
        self.calls.append(("VelL", (value,), {}))
        return "0,{},ok;"

    def RelMovJUser(self, *args, **kwargs):
        self.calls.append(("RelMovJUser", args, kwargs))
        self.next_id += 1
        return f"0,{{{self.next_id}}},RelMovJUser;"

    def MovL(self, *args, **kwargs):
        self.calls.append(("MovL", args, kwargs))
        self.next_id += 1
        return f"0,{{{self.next_id}}},MovL;"


def test_controller_submit_helpers_preserve_cp_and_return_handles():
    controller = DobotNova5Controller("192.168.111.101")
    dashboard = QueueDashboard()
    controller.dashboard = dashboard
    controller._single_command_motion_scaling = True

    lift = controller.submit_rel_move_user_joint(
        TcpPose(0.0, 0.0, 0.060, 0.0, 0.0, 0.0),
        speed=100,
        accel=90,
        cp=20,
        user_index=0,
        tool_index=1,
    )
    transfer = controller.submit_move_joint(
        [14.0, -29.0, -99.0, 39.0, 88.0, 15.0],
        speed=100,
        accel=100,
        cp=20,
    )
    safe_height = controller.submit_move_joint_tcp(
        TcpPose(0.560, 0.375, 0.320, -88.0, 1.0, -41.7),
        speed=100,
        accel=100,
        cp=20,
        user_index=0,
        tool_index=1,
    )
    descent = controller.submit_move_linear_tcp(
        TcpPose(0.560, 0.375, 0.080, -88.0, 1.0, -41.7),
        speed=100,
        accel=100,
        cp=20,
        user_index=0,
        tool_index=1,
    )

    assert isinstance(lift, MotionCommandHandle)
    assert isinstance(transfer, MotionCommandHandle)
    assert isinstance(safe_height, MotionCommandHandle)
    assert isinstance(descent, MotionCommandHandle)
    assert lift.command_epoch == transfer.command_epoch == safe_height.command_epoch == descent.command_epoch == 0
    assert dashboard.calls[0][0] == "RelMovJUser"
    assert dashboard.calls[0][2]["cp"] == 20
    assert dashboard.calls[1][0] == "MovJ"
    assert dashboard.calls[1][2]["cp"] == 20
    assert dashboard.calls[2][0] == "MovJ"
    assert dashboard.calls[2][2]["cp"] == 20
    assert dashboard.calls[2][2]["user"] == 0
    assert dashboard.calls[2][2]["tool"] == 1
    assert dashboard.calls[3][0] == "MovL"
    assert dashboard.calls[3][2]["cp"] == 20
    assert dashboard.calls[3][2]["user"] == 0
    assert dashboard.calls[3][2]["tool"] == 1


def test_single_command_scaling_is_not_rewritten_on_every_gui_apply():
    controller = DobotNova5Controller("192.168.111.101")
    dashboard = QueueDashboard()
    controller.dashboard = dashboard

    controller.enable_single_command_motion_scaling()
    dashboard.calls.clear()
    controller.enable_single_command_motion_scaling()

    assert dashboard.calls == []


def _blend_test_node():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "grasp_lift_m": 0.060,
        "grasp_lift_transfer_queue_lead_m": 0.010,
        "grasp_lift_transfer_blend_cp": 20,
        "grasp_lift_transfer_command_start_grace_s": 0.30,
        "jog_axis_timeout_s": 2.0,
        "user_index": 0,
        "command_tool_index": 1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._require_cycle_active = MagicMock()
    node._publish_status = MagicMock()
    node._reset_barcode_window = MagicMock()
    node._validate_grasp_feedback = MagicMock()
    node._six_values = lambda _name: [14.0, -29.0, -99.0, 39.0, 88.0, 15.0]
    node.get_logger = lambda: MagicMock()

    poses = iter(
        [
            TcpPose(0.0, 0.0, 0.010, 0.0, 0.0, 0.0),
            TcpPose(0.0, 0.0, 0.060, 0.0, 0.0, 0.0),
        ]
    )
    node._current_command_pose = lambda: next(poses)

    class FakeController:
        def __init__(self):
            self.submitted = []
            self.waited = []

        def submit_rel_move_user_joint(self, pose, **kwargs):
            command = MotionCommandHandle(101, 0)
            self.submitted.append(("lift", pose, kwargs, command))
            return command

        def submit_move_joint(self, joints, **kwargs):
            command = MotionCommandHandle(102, 0)
            self.submitted.append(("transfer", joints, kwargs, command))
            return command

        def motion_command_is_active(self, _command):
            return True

        def wait_for_command(self, command, **_kwargs):
            self.waited.append(command)

    node.controller = FakeController()
    return node


def test_blend_helper_queues_transfer_at_safe_lift_gate():
    node = _blend_test_node()
    completed = node._execute_grasp_lift_transfer(
        {
            "grasp_lift_speed": 100,
            "grasp_lift_acc": 100,
            "transfer_speed": 100,
            "transfer_acc": 100,
        },
        max_opening=0.095,
    )

    assert completed is True
    assert [entry[0] for entry in node.controller.submitted] == ["lift", "transfer"]
    assert node.controller.submitted[0][2]["cp"] == 20
    assert node.controller.submitted[1][2]["cp"] == 20
    assert node.controller.submitted[0][2]["user_index"] == 0
    assert node._validate_grasp_feedback.call_count == 1
    node._reset_barcode_window.assert_called_once_with()
    assert node.controller.waited == [MotionCommandHandle(102, 0)]


def test_blend_helper_allows_feedback_startup_grace_before_fallback():
    node = _blend_test_node()
    active_checks = {"count": 0}

    def delayed_active(_command):
        active_checks["count"] += 1
        return active_checks["count"] >= 4

    node.controller.motion_command_is_active = delayed_active
    poses = iter(
        [
            TcpPose(0.0, 0.0, 0.010, 0.0, 0.0, 0.0),
            TcpPose(0.0, 0.0, 0.010, 0.0, 0.0, 0.0),
            TcpPose(0.0, 0.0, 0.010, 0.0, 0.0, 0.0),
            TcpPose(0.0, 0.0, 0.010, 0.0, 0.0, 0.0),
            TcpPose(0.0, 0.0, 0.060, 0.0, 0.0, 0.0),
        ]
    )
    node._current_command_pose = lambda: next(poses)

    completed = node._execute_grasp_lift_transfer(
        {
            "grasp_lift_speed": 100,
            "grasp_lift_acc": 100,
            "transfer_speed": 100,
            "transfer_acc": 100,
        },
        max_opening=0.095,
    )

    assert completed is True
    assert [entry[0] for entry in node.controller.submitted] == ["lift", "transfer"]
    assert node.controller.waited == [MotionCommandHandle(102, 0)]


def test_post_scan_place_helper_queues_fixed_pose_at_safe_height_gate():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "post_scan_place_queue_lead_m": 0.020,
        "post_scan_place_blend_cp": 20,
        "post_scan_place_command_start_grace_s": 0.30,
        "face_up_user_ry_deg": 0.0,
        "post_scan_user_rz_deg": 0.0,
        "user_index": 0,
        "command_tool_index": 1,
        "jog_axis_timeout_s": 2.0,
        "jog_tolerance_m": 0.001,
        "face_up_jog_tolerance_deg": 5.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._require_cycle_active = MagicMock()
    node._publish_status = MagicMock()
    node.get_logger = lambda: MagicMock()
    node._stop_active_motion_for_queue_failure = MagicMock()

    start_pose = TcpPose(0.500, 0.100, 0.200, 0.0, 0.0, 0.0)
    safe_pose = TcpPose(0.560, 0.375, 0.320, 0.0, 0.0, 0.0)
    place_pose = TcpPose(0.531, 0.328, 0.180, 0.0, 0.0, 0.0)
    poses = iter([start_pose, safe_pose, place_pose])
    node._current_command_pose = lambda: next(poses)

    class FakeController:
        def __init__(self):
            self.submitted = []
            self.waited = []

        def current_joint(self):
            return [0.0] * 6

        def inverse_kinematics(self, pose, **_kwargs):
            return [pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz]

        def submit_move_joint_tcp(self, pose, **kwargs):
            command = MotionCommandHandle(201 + len(self.submitted), 0)
            self.submitted.append((pose, kwargs, command))
            return command

        def motion_command_is_active(self, _command):
            return True

        def wait_for_command(self, command, **_kwargs):
            self.waited.append(command)

    node.controller = FakeController()

    completed = node._execute_post_scan_safe_place_blend(
        [safe_pose.x, safe_pose.y, safe_pose.z],
        [place_pose.x, place_pose.y, place_pose.z],
        0.0,
        {"post_scan_speed": 100, "post_scan_acc": 100},
    )

    assert completed is True
    assert len(node.controller.submitted) == 2
    assert node.controller.submitted[0][0] == safe_pose
    assert node.controller.submitted[1][0] == place_pose
    assert node.controller.submitted[0][1]["cp"] == 20
    assert node.controller.submitted[1][1]["cp"] == 20
    assert node.controller.waited == [MotionCommandHandle(202, 0)]


def test_scanner_retreat_queues_post_scan_path_before_retreat_endpoint():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "scanner_retreat_extra_m": 0.030,
        "scanner_retreat_post_scan_queue_lead_m": 0.010,
        "scanner_retreat_post_scan_blend_cp": 20,
        "scanner_retreat_post_scan_command_start_grace_s": 0.30,
        "post_scan_place_queue_lead_m": 0.020,
        "post_scan_place_blend_cp": 20,
        "post_scan_place_command_start_grace_s": 0.30,
        "face_up_user_ry_deg": 0.0,
        "post_scan_user_rz_deg": 0.0,
        "user_index": 0,
        "command_tool_index": 1,
        "jog_axis_timeout_s": 2.0,
        "jog_tolerance_m": 0.001,
        "face_up_jog_tolerance_deg": 5.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._require_cycle_active = MagicMock()
    node._publish_status = MagicMock()
    node._stop_active_motion_for_queue_failure = MagicMock()
    node.get_logger = lambda: MagicMock()
    node._motion_profile = lambda: {
        "scanner_retreat_speed": 100,
        "scanner_retreat_acc": 100,
    }
    safe_pose = TcpPose(0.560, 0.375, 0.320, 0.0, 0.0, 0.0)
    place_pose = TcpPose(0.531, 0.328, 0.180, 0.0, 0.0, 0.0)
    poses = iter(
        [
            TcpPose(0.500, 0.100, 0.200, 0.0, 0.0, 0.0),
            TcpPose(0.480, 0.100, 0.200, 0.0, 0.0, 0.0),
            TcpPose(0.480, 0.100, 0.200, 0.0, 0.0, 0.0),
            safe_pose,
            place_pose,
        ]
    )
    node._current_command_pose = lambda: next(poses)
    composed = iter([safe_pose, place_pose])
    node._compose_user_target_pose = lambda *_args, **_kwargs: next(composed)

    class FakeController:
        def __init__(self):
            self.submitted = []
            self.waited = []

        @property
        def robot_mode(self):
            return 7

        def current_joint(self):
            return [0.0] * 6

        def inverse_kinematics(self, pose, **_kwargs):
            return [pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz]

        def submit_rel_move_user_joint(self, pose, **kwargs):
            command = MotionCommandHandle(401, 0)
            self.submitted.append(("retreat", pose, kwargs, command))
            return command

        def submit_move_joint_tcp(self, pose, **kwargs):
            command_id = 402 + len(self.submitted)
            command = MotionCommandHandle(command_id, 0)
            self.submitted.append(("ptp", pose, kwargs, command))
            return command

        def motion_command_is_active(self, command):
            # The scanner retreat remains the active predecessor while the
            # safe-height command is queued.  The pending safe command is
            # accepted by the helper through the allow_pending_queue path.
            return command.command_id == 401

        def wait_for_command(self, command, **_kwargs):
            self.waited.append(command)

    node.controller = FakeController()

    completed = node._execute_scanner_retreat_post_scan_blend(
        0.0,
        [0.560, 0.375, 0.320],
        [0.531, 0.328, 0.180],
        0.0,
        {"post_scan_speed": 100, "post_scan_acc": 100},
    )

    assert completed is True
    assert [entry[0] for entry in node.controller.submitted] == [
        "retreat",
        "ptp",
        "ptp",
    ]
    assert node.controller.submitted[0][1] == TcpPose(
        -0.030, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    assert node.controller.submitted[0][2]["cp"] == 20
    assert node.controller.submitted[1][1] == safe_pose
    assert node.controller.submitted[2][1] == place_pose
    assert node.controller.submitted[1][2]["cp"] == 20
    assert node.controller.submitted[2][2]["cp"] == 20
    assert node.controller.waited == [
        MotionCommandHandle(404, 0),
        MotionCommandHandle(401, 0),
    ]


def test_offset_high_descent_helper_queues_movl_before_high_endpoint():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "offset_high_descent_queue_lead_m": 0.030,
        "offset_high_descent_blend_cp": 20,
        "offset_high_descent_command_start_grace_s": 0.30,
        "user_index": 0,
        "command_tool_index": 1,
        "jog_axis_timeout_s": 2.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._require_cycle_active = MagicMock()
    node._publish_status = MagicMock()
    node._timed_stage = lambda _name: nullcontext()
    node._stop_active_motion_for_queue_failure = MagicMock()

    high_pose = TcpPose(0.388, 0.200, 0.133, 0.0, 0.0, 0.0)
    low_pose = TcpPose(0.388, 0.200, 0.013, 0.0, 0.0, 0.0)
    # The second sample is inside the 30 mm gate, while the high command is
    # still active, so the next MovL can be submitted before the PTP stops.
    poses = iter(
        [
            TcpPose(0.388, 0.200, 0.080, 0.0, 0.0, 0.0),
            TcpPose(0.388, 0.200, 0.110, 0.0, 0.0, 0.0),
        ]
    )
    node._current_command_pose = lambda: next(poses)
    events = []

    class FakeController:
        def __init__(self):
            self.submitted = []
            self.waited = []

        def submit_move_joint_tcp(self, pose, **kwargs):
            command = MotionCommandHandle(301, 0)
            events.append("high_submit")
            self.submitted.append(("high", pose, kwargs, command))
            return command

        def submit_move_linear_tcp(self, pose, **kwargs):
            command = MotionCommandHandle(302, 0)
            events.append("low_submit")
            self.submitted.append(("low", pose, kwargs, command))
            return command

        def motion_command_is_active(self, _command):
            return True

        def wait_for_command(self, command, **_kwargs):
            self.waited.append(command)

    node.controller = FakeController()

    def after_high():
        events.append("validated")

    completed = node._execute_offset_high_descent_blend(
        high_pose,
        low_pose,
        {
            "joint_speed": 100,
            "joint_pose_acc": 100,
            "linear_speed": 100,
            "linear_acc": 100,
        },
        after_offset_high=after_high,
    )

    assert completed is True
    assert [entry[0] for entry in node.controller.submitted] == ["high", "low"]
    assert events.index("validated") < events.index("low_submit")
    assert node.controller.submitted[0][2]["cp"] == 20
    assert node.controller.submitted[1][2]["cp"] == 20
    assert node.controller.submitted[1][2]["user_index"] == 0
    assert node.controller.submitted[1][2]["tool_index"] == 1
    assert node.controller.waited == [MotionCommandHandle(302, 0)]
