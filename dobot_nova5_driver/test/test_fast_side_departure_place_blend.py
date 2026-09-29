import unittest
from types import SimpleNamespace

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3_fast import (
    GRIP_GRIPPED,
    CosmeticBoxSingleArmNode,
)


class FastSideDeparturePlaceBlendTest(unittest.TestCase):
    def test_holds_xy_until_camera_clearance_then_queues_transfer(self):
        node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
        params = {
            "grasp_lift_m": 0.060,
            "scan_exit_user_xyz": [0.560, 0.375, 0.320],
            "side_barcode_place_xyz": [0.531, 0.328, 0.180],
            "side_barcode_place_rx_delta_deg": -20.0,
            "user_index": 0,
            "command_tool_index": 1,
            "jog_tolerance_m": 0.002,
            "jog_axis_timeout_s": 2.0,
            "barcode_flip_watch_joint_index": 5,
            "d435_side_face_reference_joint_deg": 0.0,
            "barcode_flip_step_deg": -90.0,
            "barcode_flip_safe_joint_limit_deg": 355.0,
            "barcode_flip_jog_tolerance_deg": 1.0,
            "fast_side_departure_place_queue_lead_m": 0.030,
            "fast_side_departure_place_blend_cp": 20,
            "fast_side_camera_clearance_z_m": 0.260,
            "fast_side_rise_after_camera_clearance_m": 0.005,
            "fast_side_departure_place_command_start_grace_s": 0.30,
            "dh_max_opening_m": 0.095,
            "grasp_success_min_opening_m": 0.004,
            "grasp_feedback_required": True,
        }
        node.get_parameter = lambda name: SimpleNamespace(value=params[name])
        node._require_cycle_active = lambda _where: None
        node._is_barcode_flip_joint_safe = lambda _joints, _delta: True
        statuses = []
        node._publish_status = statuses.append
        node.get_logger = lambda: SimpleNamespace(warning=statuses.append)

        def unexpected_stop(reason):
            raise AssertionError(reason)

        node._stop_active_motion_for_queue_failure = unexpected_stop
        node._validate_grasp_feedback = (
            lambda label, _opening: statuses.append(label)
        )

        class FakeController:
            def __init__(self):
                self.pose = TcpPose(0.550, -0.150, 0.180, 0.0, 0.0, 20.0)
                self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
                self.submitted = []
                self.active = False
                self.start_feedback_polls = 2

            def current_joint(self):
                return list(self.joints)

            def forward_kinematics(self, joints, **_kwargs):
                return TcpPose(
                    self.pose.x,
                    self.pose.y,
                    self.pose.z,
                    0.0,
                    0.0,
                    float(joints[5]),
                )

            def inverse_kinematics(self, pose, **_kwargs):
                return [0.0, 0.0, 0.0, 0.0, 0.0, float(pose.rz)]

            def submit_move_linear_tcp(self, target, **kwargs):
                self.submitted.append((target, kwargs))
                self.pose = TcpPose(
                    self.pose.x,
                    self.pose.y,
                    0.261,
                    target.rx,
                    target.ry,
                    target.rz,
                )
                self.joints[5] = target.rz
                self.active = True
                return object()

            def motion_command_is_active(self, _command):
                if self.start_feedback_polls > 0:
                    self.start_feedback_polls -= 1
                    return False
                return self.active

            def wait_for_command(self, _command, timeout_s):
                del timeout_s
                self.active = False

        controller = FakeController()
        node.controller = controller
        node._current_command_pose = lambda: controller.pose
        node.gripper = SimpleNamespace(
            read_position=lambda: 0.40,
            read_grip_state=lambda: GRIP_GRIPPED,
        )
        removed = []
        node._mark_turntable_material_removed = lambda: removed.append(True)
        queued = []

        def execute_place(
            approach,
            fixed,
            rx,
            motion,
            *,
            allow_pending_queue=False,
            start_pose_override=None,
            queue_lead_override_m=None,
            cp_override=None,
            command_start_grace_override_s=None,
            motion_description="",
            queue_gate_callback=None,
        ):
            queued.append(
                (
                    approach,
                    fixed,
                    rx,
                    motion,
                    allow_pending_queue,
                    start_pose_override,
                    queue_lead_override_m,
                    cp_override,
                    command_start_grace_override_s,
                    motion_description,
                    queue_gate_callback,
                )
            )
            if queue_gate_callback is not None:
                queue_gate_callback()
            controller.pose = TcpPose(
                fixed[0], fixed[1], fixed[2], -20.0, 0.0, 0.0
            )
            controller.active = False
            return True

        node._execute_post_scan_safe_place_blend = execute_place
        motion = {
            "grasp_lift_speed": 20,
            "grasp_lift_acc": 20,
            "barcode_alignment_speed": 20,
            "barcode_alignment_acc": 20,
        }

        self.assertTrue(
            node._execute_turntable_lift_face_snap(
                motion,
                continuous_place=True,
            )
        )
        self.assertEqual(len(controller.submitted), 1)
        self.assertAlmostEqual(controller.submitted[0][0].x, 0.550)
        self.assertAlmostEqual(controller.submitted[0][0].y, -0.150)
        self.assertAlmostEqual(controller.submitted[0][0].z, 0.265)
        self.assertEqual(controller.submitted[0][1]["cp"], 20)
        self.assertEqual(queued[0][0][:2], [0.560, 0.375])
        self.assertAlmostEqual(queued[0][0][2], 0.265)
        self.assertEqual(queued[0][1], [0.531, 0.328, 0.180])
        self.assertTrue(queued[0][4])
        self.assertAlmostEqual(queued[0][5].z, 0.265)
        self.assertEqual(queued[0][6], 0.030)
        self.assertEqual(queued[0][7], 20)
        self.assertEqual(queued[0][8], 0.30)
        self.assertEqual(queued[0][9], "fast camera-safe rising transfer")
        self.assertTrue(callable(queued[0][10]))
        self.assertEqual(removed, [True])
        self.assertTrue(node._fast_turntable_removed_precompleted)
        self.assertTrue(node._fast_side_place_precompleted)
        self.assertTrue(
            any("fast camera-safe rising placement reached" in text for text in statuses)
        )


if __name__ == "__main__":
    unittest.main()
