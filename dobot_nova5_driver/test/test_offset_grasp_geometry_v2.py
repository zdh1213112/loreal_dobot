import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dobot_nova5_driver.offset_grasp_geometry_v2 import (
    camera_rotation_at_target_tcp,
    plan_offset,
)


def test_image_down_maps_to_horizontal_user_direction_and_scales_with_box():
    camera = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    offset = plan_offset([.5, .2, .013], camera, .124, .060)
    np.testing.assert_allclose(offset, [.388, .2, .013])


def test_unreliable_camera_projection_is_rejected():
    camera = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    with pytest.raises(ValueError, match='projection'):
        plan_offset([0, 0, .02], camera, .12, .060)


def test_nonfinite_geometry_is_rejected():
    with pytest.raises(ValueError):
        plan_offset([0, 0, .02], np.eye(3), float('nan'), .060)


def test_camera_rotation_is_predicted_at_target_tcp_attitude():
    current_tcp = Rotation.from_euler('z', 10, degrees=True).as_matrix()
    current_camera = Rotation.from_euler('z', 100, degrees=True).as_matrix()
    target_tcp = Rotation.from_euler('z', 40, degrees=True).as_matrix()

    predicted = camera_rotation_at_target_tcp(
        current_tcp, current_camera, target_tcp
    )

    expected = Rotation.from_euler('z', 130, degrees=True).as_matrix()
    np.testing.assert_allclose(predicted, expected, atol=1e-12)


def make_motion_node(cancel_after_low=False):
    """Exercise production orchestration without importing ROS or contacting hardware."""
    import ast
    import math
    import threading
    import time
    from contextlib import nullcontext
    from dataclasses import dataclass
    from pathlib import Path
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    source = Path(__file__).parents[1] / 'dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v2.py'
    module = ast.parse(source.read_text())
    node_class = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'CosmeticBoxSingleArmNode')
    method = next(n for n in node_class.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_offset_entry')
    @dataclass
    class Pose:
        x: float; y: float; z: float; rx: float; ry: float; rz: float
    def transform(p):
        t = np.eye(4); t[:3,:3] = Rotation.from_euler('xyz',[p.rx,p.ry,p.rz],degrees=True).as_matrix()
        t[:3,3] = [p.x,p.y,p.z]; return t
    ns = dict(np=np, math=math, time=time, TcpPose=Pose, SciPyRot=Rotation,
              pose_to_transform=transform, plan_offset=plan_offset,
              camera_rotation_at_target_tcp=camera_rotation_at_target_tcp,
              String=SimpleNamespace)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    pose = Pose(.5,.2,.3,0,0,0)
    moves=[]; events=[]
    params = dict(user_index=0, command_tool_index=1, flange_tool_index=0,
                  offset_finger_span_m=.06, offset_grasp_clearance_m=.02,
                  offset_high_clearance_m=.12,
                  # Keep these orchestration tests on the original blocking
                  # path; the asynchronous CP helper has dedicated tests.
                  offset_high_descent_blend_enabled=False,
                  offset_high_descent_blend_cp=20,
                  offset_high_descent_queue_lead_m=.030,
                  offset_high_descent_command_start_grace_s=.30,
                  pregrasp_min_hover_clearance_m=.03,
                  grasp_z_offset_m=.010)
    def move(p, **kw):
        nonlocal pose
        pose=p; moves.append(p)
    def active(stage):
        if cancel_after_low and len(moves)>=2:
            raise RuntimeError('cancelled')
    node=SimpleNamespace(
        get_parameter=lambda k:SimpleNamespace(value=params[k]),
        _current_command_pose=lambda:pose,
        _six_values_from_rotation=lambda k:[0,0,0],
        handeye_flange_to_cam=transform(Pose(0,0,0,0,0,90)),
        data_lock=threading.Lock(), rgb_to_ir_rotation=np.eye(3),
        _require_cycle_active=active,_publish_status=events.append,
        _motion_profile=lambda:{'joint_speed':100,'joint_pose_acc':100,
                                'linear_speed':100,'linear_acc':100},
        _timed_stage=lambda s:nullcontext(),_wait_for_top_surface_barcode=lambda:events.append('wait'),
        controller=SimpleNamespace(current_tcp_pose=lambda **kw:Pose(0,0,0,0,0,0),
                                   inverse_kinematics=lambda *a,**kw:None,current_joint=lambda:[0]*6,
                                   move_joint_tcp=move,
                                   move_linear_tcp=move))
    def run(after_high=None):
        return ns['_execute_offset_entry'](
            node, Pose(.5,.2,.013,0,0,0), .124, after_high
        )
    run.params = params
    return run, moves, events


def test_offset_path_order_and_low_observation():
    run,moves,events=make_motion_node()
    run()
    np.testing.assert_allclose([[p.x,p.y,p.z] for p in moves],
                               [[.388,.2,.133],[.388,.2,.013],[.5,.2,.013]])
    assert events.index('offset_descent') < events.index('wait') < events.index('offset_insert')


def test_cancellation_after_descent_prevents_insertion():
    run,moves,_=make_motion_node(cancel_after_low=True)
    with pytest.raises(RuntimeError,match='cancelled'):
        run()
    assert len(moves)==2


def test_offset_high_callback_runs_before_descent():
    run,moves,events=make_motion_node()

    def after_high():
        events.append(('validated', len(moves)))

    run(after_high)

    assert ('validated', 1) in events
    assert events.index(('validated', 1)) < events.index('offset_descent')


def test_offset_descent_and_insert_use_normalized_linear_speed():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / 'dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v2.py'
    module = ast.parse(source.read_text())
    node_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == 'CosmeticBoxSingleArmNode'
    )
    method = next(
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '_execute_offset_entry'
    )
    section = ast.get_source_segment(source.read_text(), method)

    assert 'speed=motion["linear_speed"]' in section
    assert 'accel=motion["linear_acc"]' in section
    assert 'offset_grasp_speed_percent' not in section


def test_offset_high_clearance_cannot_undercut_pregrasp_minimum():
    run, moves, _ = make_motion_node()
    run.params['offset_high_clearance_m'] = .02

    with pytest.raises(RuntimeError, match='must be at least'):
        run()

    assert moves == []


def test_redundant_startup_move_is_skipped_but_gripper_still_opens():
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace

    source = Path(__file__).parents[1] / 'dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v2.py'
    module = ast.parse(source.read_text())
    node_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == 'CosmeticBoxSingleArmNode'
    )
    method = next(
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '_move_startup_and_open'
    )
    namespace = {'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

    startup = [14.0, 7.0, -130.0, 47.0, 83.0, 10.0]
    joint_moves = []
    gripper_calls = []
    statuses = []
    node = SimpleNamespace(
        _motion_profile=lambda: {'return_startup_speed': 100, 'return_startup_acc': 100},
        _six_values=lambda name: startup,
        get_parameter=lambda name: SimpleNamespace(
            value={'startup_joint_skip_tolerance_deg': 1.0, 'dh_force': 50}[name]
        ),
        _publish_status=statuses.append,
        _require_cycle_active=lambda stage: None,
        _cycle_cancel_requested=lambda: False,
        get_logger=lambda: SimpleNamespace(warning=lambda message: None),
        controller=SimpleNamespace(
            current_joint=lambda: [14.2, 7.0, -130.1, 47.0, 83.0, 10.0],
            move_joint=lambda *args, **kwargs: joint_moves.append((args, kwargs)),
        ),
        gripper=SimpleNamespace(
            set_force=lambda force: gripper_calls.append(('force', force)),
            open=lambda **kwargs: gripper_calls.append(('open', kwargs)),
        ),
    )

    namespace['_move_startup_and_open'](node)

    assert joint_moves == []
    assert [call[0] for call in gripper_calls] == ['force', 'open']
    assert any('skipping redundant MovJ' in status for status in statuses)


def test_enabled_offset_branch_combines_orientation_with_offset_high():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / 'dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v2.py'
    module = ast.parse(source.read_text())
    node_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == 'CosmeticBoxSingleArmNode'
    )
    method = next(
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '_execute_one_cycle'
    )

    def called_attributes(nodes):
        wrapper = ast.Module(body=nodes, type_ignores=[])
        return {
            call.func.attr
            for call in ast.walk(wrapper)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }

    offset_branch = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and '_execute_offset_entry' in called_attributes(node.body)
    )

    assert '_execute_offset_entry' in called_attributes(offset_branch.body)
    assert 'move_joint_tcp' not in called_attributes(offset_branch.body)
    assert 'move_joint_tcp' in called_attributes(offset_branch.orelse)
    assert all(
        not isinstance(node, ast.Constant) or node.value != 'grasp_orientation_align'
        for node in ast.walk(method)
    )

    arm_call = next(
        call for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == '_set_top_surface_barcode_window'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value is True
    )
    assert arm_call.lineno < offset_branch.lineno
