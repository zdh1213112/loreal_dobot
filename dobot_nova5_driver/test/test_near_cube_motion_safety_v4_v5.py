"""Offline kinematic-risk and held-material fault regressions; no robot sockets."""
import importlib
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.dobot_dh_api_v3 import GRIP_GRIPPED, GRIP_REACHED
from dobot_nova5_driver.near_cube_motion_safety import (
    NEAR_CUBE_SAFETY_DEFAULTS, NearCubeHeldMotionFailure, NearCubePreflightError,
    PreflightSettings, SampledCartesianPreflight, with_near_cube_preflight_session,
)


def settings(**changes):
    defaults = {key.removeprefix("bottom_near_cube_preflight_"): value
                for key, value in NEAR_CUBE_SAFETY_DEFAULTS.items()}
    defaults['joint_limits_deg'] = tuple(defaults['joint_limits_deg'])
    return replace(PreflightSettings(**defaults), **changes)


class PlanarArm:
    """Independent 2-link arm + Z axis + Euler wrist, with analytic IK/FK.

    The straight elbow and the pitch=90 wrist have known singularities.
    A reachable endpoint alone therefore cannot certify a motion.
    """
    def forward(self, joints):
        shoulder, elbow, z_axis = np.radians(joints[:3])
        return TcpPose(
            .45 * math.cos(shoulder) + .35 * math.cos(shoulder + elbow),
            .45 * math.sin(shoulder) + .35 * math.sin(shoulder + elbow),
            .5 * z_axis, *map(float, joints[3:]),
        )

    def inverse(self, pose, nearby):
        c = (pose.x**2 + pose.y**2 - .45**2 - .35**2) / (2 * .45 * .35)
        if not -1.00000001 <= c <= 1.00000001:
            raise NearCubePreflightError('unreachable analytic arm')
        elbow = math.acos(max(-1., min(1., c)))
        candidates = []
        for angle in (elbow, -elbow):
            shoulder = math.atan2(pose.y, pose.x) - math.atan2(
                .35 * math.sin(angle), .45 + .35 * math.cos(angle))
            candidates.append([math.degrees(shoulder), math.degrees(angle),
                               math.degrees(pose.z / .5), pose.rx, pose.ry, pose.rz])
        for q in candidates:
            for axis in range(3, 6):
                q[axis] += 360. * round((nearby[axis] - q[axis]) / 360.)
        return min(candidates, key=lambda q: np.linalg.norm(np.array(q) - nearby))


def make_checker(arm=None, **changes):
    arm = arm or PlanarArm()
    return SampledCartesianPreflight(arm.inverse, arm.forward, lambda: None,
                                     settings(**changes))


def test_sampled_regular_path_passes_without_motion_commands():
    arm = PlanarArm()
    start = TcpPose(.52, -.17, .19, 135., 0., -108.)
    checker = make_checker(arm)
    end = replace(start, z=.22)
    final = checker.check(start, arm.inverse(start, [0.] * 6), [('lift', end)])
    assert arm.forward(final).z == pytest.approx(.22)
    assert checker.sample_count >= 6
    assert checker.minimum_sigma > .025
    assert checker.maximum_joint_step_deg < 20.


def test_reachable_straight_elbow_is_rejected():
    arm = PlanarArm()
    start = TcpPose(.8, 0., .19, 135., 0., 0.)
    # Exact IK exists, but the Jacobian loses rank at this position.
    joints = arm.inverse(start, [0.] * 6)
    with pytest.raises(NearCubePreflightError, match='singularity margin'):
        make_checker(arm).check(start, joints, [('lift', replace(start, z=.22))])


def test_wrist_singularity_inside_path_is_rejected_with_reachable_endpoints():
    arm = PlanarArm()
    start = TcpPose(.52, -.17, .19, 0., 80., 0.)
    end = replace(start, ry=100.)
    # Choosing the equivalent Euler solution near the previous wrist angles
    # removes representation jumps so the numerical singularity test is used.
    analytic_inverse = arm.inverse

    def continuous_inverse(pose, nearby):
        q = analytic_inverse(pose, nearby)
        alternatives = [q, q[:3] + [q[3] + 180., 180. - q[4], q[5] + 180.],
                        q[:3] + [q[3] - 180., 180. - q[4], q[5] - 180.]]
        return min(alternatives, key=lambda item: np.linalg.norm(np.array(item) - nearby))

    checker = make_checker(arm, rotation_step_deg=5.)
    checker.inverse = continuous_inverse
    assert make_checker(arm).check(start, analytic_inverse(start, [0.] * 6), [])
    assert make_checker(arm).check(end, analytic_inverse(end, [0.] * 6), [])
    with pytest.raises(NearCubePreflightError, match='singularity margin'):
        checker.check(start, analytic_inverse(start, [0.] * 6), [('wrist', end)])


@pytest.mark.parametrize('problem', ['branch', 'limit', 'nan', 'mismatch'])
def test_bad_ik_does_not_pass_preflight(problem):
    arm = PlanarArm()
    start = TcpPose(.52, -.17, .19, 135., 0., 0.)
    checker = make_checker(arm)

    def bad_inverse(pose, nearby):
        q = arm.inverse(pose, nearby)
        if problem == 'branch':
            q[3] -= 360.  # Same FK pose, genuinely discontinuous joint branch.
        elif problem == 'limit':
            q[2] = 160.
        elif problem == 'nan':
            q[0] = float('nan')
        else:
            q[2] += .2  # 1.75 mm FK/IK error, without a branch discontinuity.
        return q

    checker.inverse = bad_inverse
    expected = {'branch': 'unresolved joint step', 'limit': 'joint-limit margin',
                'nan': 'six finite joints', 'mismatch': 'IK/FK mismatch'}[problem]
    with pytest.raises(NearCubePreflightError, match=expected):
        checker.check(start, arm.inverse(start, [0.] * 6), [('lift', replace(start, z=.22))])


class GearedWristArm(PlanarArm):
    """A continuous wrist needing 2–3x the Tool angle, without singularity."""
    def __init__(self, ratio, origin_deg=0.):
        self.ratio = ratio
        self.origin_deg = origin_deg

    def forward(self, joints):
        q = list(joints)
        q[3] = q[3] / self.ratio + self.origin_deg
        return super().forward(q)

    def inverse(self, pose, nearby):
        seed = list(nearby)
        seed[3] = seed[3] / self.ratio + self.origin_deg
        q = super().inverse(pose, seed)
        q[3] = (q[3] - self.origin_deg) * self.ratio
        return q


@pytest.mark.parametrize('coarse_change', [20.16, 22.84, 24.23, 25.15])
def test_large_continuous_joint_change_passes_after_refinement(coarse_change):
    # These are the oversized steps in the real log. A step size alone
    # cannot tell a fast continuous solution from an IK branch switch.
    arm = GearedWristArm(coarse_change / 10.)
    start = TcpPose(.52, -.17, .19, 0., 0., 0.)
    target = replace(start, rx=10.)
    checker = make_checker(arm)
    final = checker.check(start, arm.inverse(start, [0.] * 6), [('flip', target)])
    assert arm.forward(final).rx == pytest.approx(10.)
    assert checker.maximum_trial_joint_step_deg == pytest.approx(coarse_change)
    assert checker.refinement_count >= 1
    assert checker.sample_count >= 2
    assert checker.maximum_joint_step_deg <= 20.
    assert checker.minimum_sigma >= .025


def test_refinement_does_not_bypass_singularity_margin():
    arm = GearedWristArm(2.5)
    start = TcpPose(.52, -.17, .19, 0., 80., 0.)
    target = replace(start, rx=10., ry=90.)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        with pytest.raises(NearCubePreflightError, match='singularity margin'):
            make_checker(arm).check(start, arm.inverse(start, [0.] * 6), [('flip', target)])


def test_inverse_branch_can_be_recovered_only_by_a_new_midpoint_seed():
    arm = PlanarArm()
    start = TcpPose(.52, -.17, .19, 135., 0., 0.)
    target = replace(start, rx=144.)
    checker = make_checker(arm)
    tried = []

    def seed_sensitive_inverse(pose, nearby):
        q = arm.inverse(pose, nearby)
        if abs(q[3] - nearby[3]) > 6.:
            q[3] -= 360.
        tried.append((pose.rx, nearby[3], q[3]))
        return q

    checker.inverse = seed_sensitive_inverse
    final = checker.check(start, arm.inverse(start, [0.] * 6), [('flip', target)])
    assert final[3] == pytest.approx(144.)
    assert checker.refinement_count >= 1
    assert tried[0][2] < 0.
    assert tried[-1][1] >= 139.5
    assert tried[-1][2] == pytest.approx(144.)


def test_adaptive_query_budget_is_bounded():
    arm = PlanarArm()
    checker = make_checker(arm)
    checker.ik_query_count = 256
    checker.inverse = MagicMock(side_effect=arm.inverse)
    start = TcpPose(.52, -.17, .19, 135., 0., 0.)
    with pytest.raises(NearCubePreflightError, match='256 IK queries'):
        checker.check(start, arm.inverse(start, [0.] * 6), [('lift', replace(start, z=.20))])
    checker.inverse.assert_not_called()


def test_cancellation_stops_querying_immediately():
    arm = PlanarArm()
    checker = make_checker(arm)
    calls = []

    def active():
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError('operator stop')

    checker.check_active = active
    start = TcpPose(.52, -.17, .19, 135., 0., 0.)
    with pytest.raises(RuntimeError, match='operator stop'):
        checker.check(start, arm.inverse(start, [0.] * 6), [('lift', replace(start, z=.22))])
    assert len(calls) == 5


def test_query_timing_accounts_for_each_read_only_controller_call(monkeypatch):
    clock = [0.]
    arm = PlanarArm()
    checker = make_checker(arm, fast_jacobian=False)

    def forward(q):
        clock[0] += .007
        return arm.forward(q)

    def inverse(pose, nearby):
        clock[0] += .011
        return arm.inverse(pose, nearby)

    monkeypatch.setattr('dobot_nova5_driver.near_cube_motion_safety.time.perf_counter', lambda: clock[0])
    checker.forward, checker.inverse = forward, inverse
    start = TcpPose(.52, -.17, .19, 135., 0., 0.)
    checker.check(start, arm.inverse(start, [0.] * 6), [('lift', replace(start, z=.22))])
    assert checker.ik_query_count > 0 and checker.fk_query_count > 0
    assert checker.ik_query_s == pytest.approx(checker.ik_query_count * .011)
    assert checker.fk_query_s == pytest.approx(checker.fk_query_count * .007)


@pytest.mark.parametrize('changes', [dict(linear_step_m=0.), dict(min_sigma=0.),
                                   dict(joint_limits_deg=(0.,) * 12)])
def test_invalid_safety_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        settings(**changes)


@pytest.fixture(params=[4, 5], ids=['V4', 'V5'])
def rig(request):
    mod = importlib.import_module(
        f'dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v{request.param}')
    node = mod.CosmeticBoxSingleArmNode.__new__(mod.CosmeticBoxSingleArmNode)
    params = dict(NEAR_CUBE_SAFETY_DEFAULTS, user_index=0, command_tool_index=1,
                  bottom_near_cube_initial_tool_rx_deg=-45.,
                  bottom_near_cube_flip_tool_rx_deg=90.,
                  bottom_near_cube_restore_tool_rx_deg=-45.,
                  bottom_near_cube_flip_lift_m=.030,
                  bottom_near_cube_preflight_recovery_lift_m=.020,
                  bottom_near_cube_preflight_recovery_shift_m=.060,
                  dh_timeout_s=3., dh_grasp_force=50, jog_tolerance_m=.002)
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = MagicMock()
    node._cycle_cancel_requested = lambda: False
    node._publish_status = MagicMock()
    node.get_logger = MagicMock()
    node._set_d405_pretracking = MagicMock()
    node.secondary_auto_resume_requested = threading.Event()
    node.secondary_safety_lock = threading.Lock()
    node.running = node.cycle_enabled = True
    node.near_cube_held_motion_fault = ''
    node.d405_transform_suspended = False
    node.controller = MagicMock()
    node.controller.robot_mode = 5
    pose = [TcpPose(.52, -.17, .19, 180., 0., -108.)]
    arm = PlanarArm()
    node.controller.current_joint.side_effect = lambda: arm.inverse(pose[0], [0.] * 6)
    node.controller.inverse_kinematics.side_effect = lambda target, **kw: arm.inverse(target, kw['joint_near'])
    node.controller.forward_kinematics.side_effect = lambda joints, **kw: arm.forward(joints)
    node._current_command_pose = lambda: pose[0]
    events = []
    node.gripper = MagicMock()
    node.gripper.read_position.return_value = .75
    node.gripper.read_grip_state.return_value = GRIP_GRIPPED
    node.gripper.set_position.side_effect = lambda value, **kw: events.append(('open', value))
    node.gripper.close.side_effect = lambda **kw: events.append(('close',))
    node._confirm_grasp_before_lift = MagicMock()
    node._validate_grasp_feedback = MagicMock()

    @contextmanager
    def stage(name):
        events.append(('stage', name))
        yield

    def tool_rx(delta, label):
        pose[0] = node._tool_rx_target_pose(pose[0], delta)
        events.append(('rx', delta))

    def linear_z(z, motion, label, **kw):
        xy = kw.get('target_xy', (pose[0].x, pose[0].y))
        pose[0] = replace(pose[0], x=xy[0], y=xy[1], z=z)
        events.append(('z', z))

    def move(target, **kw):
        pose[0] = target
        events.append(('shift', target.x))

    node._timed_stage = stage
    node._bottom_center_tool_rx = tool_rx
    node._bottom_center_linear_z = linear_z
    node.controller.move_linear_tcp.side_effect = move
    motion = dict(linear_speed=20, linear_acc=20)
    return SimpleNamespace(node=node, params=params, pose=pose, events=events,
                           motion=motion, mod=mod)


def run_flip(rig):
    rig.node._execute_turntable_near_cube_bottom_flip(rig.pose[0], .095, rig.motion)


def test_normal_flip_is_checked_before_closing_and_keeps_sequence(rig):
    run_flip(rig)
    assert [event for event in rig.events if event[0] == 'rx'] == [
        ('rx', -45.), ('rx', 90.), ('rx', -45.)]
    assert [event for event in rig.events if event[0] == 'z'] == [('z', .22), ('z', .19)]
    checks = [call.args[0] for call in rig.node._publish_status.call_args_list
              if 'sampled preflight passed' in call.args[0]]
    assert len(checks) == 2
    assert rig.events.index(('stage', 'bottom_near_cube_motion_preflight')) < rig.events.index(('close',))
    assert len([e for e in rig.events if e[0] == 'close']) == 2


def test_model_session_is_shared_only_inside_one_flip(rig):
    original = rig.node._new_near_cube_preflight
    checks = []

    def capture():
        result = original()
        checks.append(result)
        return result

    rig.node._new_near_cube_preflight = capture
    run_flip(rig)
    run_flip(rig)
    assert len(checks) == 4
    assert checks[0]._model_session is checks[1]._model_session
    assert checks[2]._model_session is checks[3]._model_session
    assert checks[0]._model_session is not checks[2]._model_session
    assert original()._model_session is None


def test_exception_always_clears_flip_model_session(rig):
    original = rig.node._bottom_center_tool_rx

    def failed(*args):
        raise RuntimeError('operator stop')

    rig.node._bottom_center_tool_rx = failed
    with pytest.raises(RuntimeError, match='operator stop'):
        run_flip(rig)
    assert rig.node._new_near_cube_preflight()._model_session is None
    rig.node._bottom_center_tool_rx = original
    run_flip(rig)


@pytest.mark.parametrize('parameter', ['user_index', 'command_tool_index',
                                     'bottom_near_cube_preflight_min_sigma'])
def test_frame_or_settings_change_cannot_reuse_previous_model_session(rig, parameter):
    @with_near_cube_preflight_session
    def scoped():
        first = rig.node._new_near_cube_preflight()
        rig.params[parameter] += .001 if 'sigma' in parameter else 1
        second = rig.node._new_near_cube_preflight()
        return first, second

    first, second = scoped()
    assert first._model_session is not second._model_session


def test_simultaneous_worker_threads_do_not_share_flip_models(rig):
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(2)

    @with_near_cube_preflight_session
    def scoped():
        first = rig.node._new_near_cube_preflight()
        barrier.wait(timeout=5.)
        second = rig.node._new_near_cube_preflight()
        assert first._model_session is second._model_session
        return first._model_session

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: scoped(), range(2)))
    assert results[0] is not results[1]


def test_adaptive_flip_avoids_relocation_for_large_continuous_steps(rig):
    arm = GearedWristArm(2.5, origin_deg=180.)
    rig.node.controller.current_joint.side_effect = lambda: arm.inverse(rig.pose[0], [0.] * 6)
    rig.node.controller.inverse_kinematics.side_effect = lambda target, **kw: arm.inverse(target, kw['joint_near'])
    rig.node.controller.forward_kinematics.side_effect = lambda joints, **kw: arm.forward(joints)
    run_flip(rig)
    assert not any(event[0] == 'shift' for event in rig.events)
    assert len([event for event in rig.events if event[0] == 'close']) == 2
    passed = [call.args[0] for call in rig.node._publish_status.call_args_list
              if 'sampled preflight passed' in call.args[0]]
    assert len(passed) == 2
    assert all('refinements=0,' not in message for message in passed)


def test_reduced_lift_is_selected_before_gripping(rig):
    checks = []

    def check(z, lift, *args):
        checks.append(lift)
        assert ('close',) not in rig.events
        if lift > .02000001:
            raise NearCubePreflightError('elbow margin')

    rig.node._preflight_near_cube_bottom_flip = check
    run_flip(rig)
    assert checks == pytest.approx([.030, .025, .020])
    assert ('z', .21) in rig.events
    assert not any(e[0] == 'shift' for e in rig.events)


def test_all_lifts_fail_then_one_checked_relocation_is_used(rig):
    attempts = []

    def check(z, lift, *args):
        attempts.append((rig.pose[0].x, lift))
        if rig.pose[0].x > .5:
            raise NearCubePreflightError('elbow margin')

    rig.node._preflight_near_cube_bottom_flip = check
    run_flip(rig)
    assert [x for x, lift in attempts] == pytest.approx([.52, .52, .52, .46])
    assert len([event for event in rig.events if event[0] == 'shift']) == 1
    assert any('near-cube relocation sampled preflight passed' in call.args[0]
               for call in rig.node._publish_status.call_args_list)


def test_initial_attitude_risk_relocates_before_rotation(rig):
    original = rig.node._check_near_cube_tool_rx

    def check(delta, label):
        if label == 'initial pickup attitude':
            raise NearCubePreflightError('risky initial rotation')
        return original(delta, label)

    rig.node._check_near_cube_tool_rx = check
    run_flip(rig)
    shift_index = next(i for i, event in enumerate(rig.events) if event[0] == 'shift')
    rx_index = next(i for i, event in enumerate(rig.events) if event[0] == 'rx')
    assert shift_index < rx_index


def test_relocation_is_not_repeated_when_retry_is_still_unsafe(rig):
    def rejected(*args):
        raise NearCubePreflightError('still singular')

    rig.node._preflight_near_cube_bottom_flip = rejected
    with pytest.raises(NearCubePreflightError, match='still singular'):
        run_flip(rig)
    assert len([event for event in rig.events if event[0] == 'shift']) == 1
    # Only the upright relocation closes; no tilted pickup was attempted.
    assert len([event for event in rig.events if event[0] == 'close']) == 1


@pytest.mark.parametrize('failure', ['fk_query', 'controller_fault'])
def test_unavailable_kinematic_checks_do_not_trigger_relocation(rig, failure):
    if failure == 'fk_query':
        rig.node.controller.forward_kinematics.side_effect = RuntimeError('PositiveKin failed')
    else:
        rig.node.controller.robot_mode = 9
    with pytest.raises(RuntimeError):
        run_flip(rig)
    assert not any(e[0] in ('close', 'rx', 'shift', 'z') for e in rig.events)


@pytest.mark.parametrize('mode', [5, 7, 9, 11])
def test_held_lift_failure_holds_box_and_blocks_auto_recovery(rig, mode):
    def rejected(*args, **kw):
        rig.node.controller.robot_mode = mode
        raise RuntimeError('planning close to elbow singularity')

    rig.node._bottom_center_linear_z = rejected
    rig.node.secondary_auto_resume_requested.set()
    rig.node.controller.current_feedback_tcp_pose_with_timestamp.return_value = (
        rig.pose[0], 0, 1, time.time())
    with pytest.raises(NearCubeHeldMotionFailure, match='short_lift') as failure:
        run_flip(rig)
    assert 'elbow singularity' in str(failure.value)
    assert rig.node.near_cube_held_motion_fault == 'short_lift'
    assert rig.node.cycle_enabled is False
    assert rig.node.d405_transform_suspended is True
    assert not rig.node.secondary_auto_resume_requested.is_set()
    assert rig.node.controller.stop_motion.call_count == int(mode == 7)
    assert [event for event in rig.events if event[0] == 'open'] == [('open', 1.)]
    assert [event for event in rig.events if event[0] == 'close'] == [('close',)]
    assert not any(event[0] == 'shift' for event in rig.events)
    rig.node.controller.clear_error.assert_not_called()
    rig.node.controller.move_joint.assert_not_called()


@pytest.mark.parametrize('failure_stage', ['flip', 'return_table', 'relocation_shift'])
def test_later_held_motion_failures_cannot_release_or_go_home(rig, failure_stage):
    original_rx = rig.node._bottom_center_tool_rx
    original_z = rig.node._bottom_center_linear_z

    def fail():
        rig.node.controller.robot_mode = 9
        raise RuntimeError('controller planning fault')

    if failure_stage == 'flip':
        def rx(delta, label):
            return fail() if delta == 90. else original_rx(delta, label)
        rig.node._bottom_center_tool_rx = rx
    elif failure_stage == 'return_table':
        def z(target, motion, label, **kw):
            return original_z(target, motion, label, **kw) if kw['lifting'] else fail()
        rig.node._bottom_center_linear_z = z
    else:
        def reject_original(*args):
            raise NearCubePreflightError('elbow margin')
        rig.node._preflight_near_cube_bottom_flip = reject_original
        rig.node.controller.move_linear_tcp.side_effect = lambda *args, **kw: fail()

    with pytest.raises(NearCubeHeldMotionFailure, match='controller planning fault'):
        run_flip(rig)
    assert [event for event in rig.events if event[0] == 'open'] == [('open', 1.)]
    assert [event for event in rig.events if event[0] == 'close'] == [('close',)]
    rig.node.controller.move_joint.assert_not_called()
    assert rig.node.d405_transform_suspended


def test_late_safety_resume_request_cannot_restart_held_fault_even_after_opening(rig):
    node = rig.node
    node.near_cube_held_motion_fault = 'short_lift'
    node.cycle_enabled = False
    node.gripper.read_grip_state.return_value = GRIP_REACHED
    node.gripper.read_position.return_value = 1.
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_retreat_active = threading.Event()
    node.secondary_auto_resume_requested.set()
    node.worker = node.secondary_resume_thread = None
    node._ensure_worker = MagicMock()
    node._resume_continuous_after_secondary_clearance()
    assert not node.cycle_enabled
    assert not node.secondary_auto_resume_requested.is_set()
    node._ensure_worker.assert_not_called()
    assert node.near_cube_held_motion_fault == 'short_lift'


@pytest.mark.parametrize('opening,state,allowed', [
    (.75, GRIP_GRIPPED, False), (1., GRIP_GRIPPED, False),
    (float('nan'), GRIP_REACHED, False), (.94, GRIP_REACHED, False),
    (1., GRIP_REACHED, True),
])
def test_restart_gate_requires_verified_open_gripper_before_motion(rig, opening, state, allowed):
    rig.node.near_cube_held_motion_fault = 'short_lift'
    rig.node.gripper.read_position.return_value = opening
    rig.node.gripper.read_grip_state.return_value = state
    if allowed:
        rig.node._require_near_cube_recovery_complete()
        assert rig.node.near_cube_held_motion_fault == ''
    else:
        # Exercise the actual startup entry, so ordering before motion matters.
        with pytest.raises(RuntimeError, match='startup motion was not issued'):
            rig.node._move_startup_and_open()
        assert rig.node.near_cube_held_motion_fault == 'short_lift'
    rig.node.controller.move_joint.assert_not_called()
    rig.node.gripper.set_position.assert_not_called()


def test_faulted_background_transforms_stop_dashboard_queries_until_idle_restart(rig):
    rig.node.controller.robot_mode = 9
    for _ in range(5):
        assert rig.node._transform_vision_pose(None, 0.) is None
    rig.node.controller.current_tcp_pose.assert_not_called()
    rig.node._set_d405_pretracking.assert_called_once_with(False)
    rig.node.controller.robot_mode = 7
    assert rig.node._resume_d405_pose_processing() is False
    rig.node.controller.robot_mode = 5
    # Leaving error mode does not resume queries by itself.
    assert rig.node._d405_transform_allowed() is False
    assert rig.node._resume_d405_pose_processing() is True
    assert rig.node._d405_transform_allowed() is True


def test_real_preflight_checker_rejects_bad_relocation_before_closing(rig):
    rig.params['bottom_near_cube_preflight_recovery_shift_m'] = .060
    rig.node.controller.inverse_kinematics.side_effect = RuntimeError('InverseKin failed: -1')
    with pytest.raises(NearCubePreflightError):
        rig.node._relocate_near_cube_after_preflight_failure(
            .095, rig.motion, lift_m=.020, shift_m=.060)
    assert not any(event[0] in ('close', 'shift', 'z') for event in rig.events)


@pytest.mark.parametrize('fail_parent', [False, True])
def test_nested_recovery_timing_counts_wall_time_once(rig, monkeypatch, fail_parent):
    now = [0.]
    monkeypatch.setattr(rig.mod, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    rig.params['timing_enabled'] = True
    node = rig.node
    node._emit_timing = MagicMock()
    node.active_timing = rig.mod.CycleTiming('nested-recovery', started_at=0.)
    stage = rig.mod.CosmeticBoxSingleArmNode._timed_stage
    try:
        with stage(node, 'preflight'):
            now[0] = 1.
            with stage(node, 'relocation'):
                now[0] = 2.
                with stage(node, 'lift'):
                    now[0] = 3.
                now[0] = 4.
            now[0] = 5.
            if fail_parent:
                raise RuntimeError('preflight rejected')
    except RuntimeError:
        assert fail_parent
    now[0] = 6.
    with stage(node, 'return'):
        now[0] = 7.
    now[0] = 8.
    summary = node.active_timing.summary('fault' if fail_parent else 'success')
    assert summary['total_s'] == 8.
    assert summary['accounted_s'] == 6.
    assert summary['overhead_s'] == 2.
    assert sum(s['duration_s'] for s in summary['stages']) == 10.
    parent = next(s for s in summary['stages'] if s['stage'] == 'preflight')
    assert parent['duration_s'] == 5.
    assert parent['outcome'] == ('error' if fail_parent else 'ok')

