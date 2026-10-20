"""Independent serial arm, read-only query savings and local-IK containment."""
import math
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.controller_kinematics_model import LocalInverseUnavailable
from dobot_nova5_driver.near_cube_motion_safety import (
    NEAR_CUBE_SAFETY_DEFAULTS, NearCubePreflightError, PreflightModelSession,
    PreflightSettings, SampledCartesianPreflight,
)


class SerialArm:
    """Analytic planar elbow, prismatic Z, then a Z/Y/X wrist.

    The last joint is geared, so coarse Tool intervals need refinement.
    No production POE or local inverse helpers are used for reference IK/FK.
    """
    def forward(self, joints):
        shoulder, elbow, z = np.radians(joints[:3])
        return TcpPose(
            .45 * math.cos(shoulder) + .35 * math.cos(shoulder + elbow),
            .45 * math.sin(shoulder) + .35 * math.sin(shoulder + elbow),
            .5 * z, 180. + joints[5] / 2.5, joints[4],
            joints[0] + joints[1] + joints[3],
        )

    def inverse(self, pose, nearby):
        cosine = (pose.x**2 + pose.y**2 - .45**2 - .35**2) / (.9 * .35)
        if abs(cosine) > 1.00000001:
            raise NearCubePreflightError('reference inverse is unreachable')
        candidates = []
        for elbow in (math.acos(np.clip(cosine, -1., 1.)), -math.acos(np.clip(cosine, -1., 1.))):
            shoulder = math.atan2(pose.y, pose.x) - math.atan2(
                .35 * math.sin(elbow), .45 + .35 * math.cos(elbow))
            a, b = math.degrees(shoulder), math.degrees(elbow)
            yaw = pose.rz - a - b
            yaw += 360. * round((nearby[3] - yaw) / 360.)
            roll = pose.rx + 360. * round((180. + nearby[5] / 2.5 - pose.rx) / 360.)
            candidates.append([a, b, math.degrees(pose.z / .5), yaw, pose.ry,
                               (roll - 180.) * 2.5])
        return min(candidates, key=lambda q: np.linalg.norm(np.array(q) - nearby))


def settings(**changes):
    values = {k.removeprefix('bottom_near_cube_preflight_'): v
              for k, v in NEAR_CUBE_SAFETY_DEFAULTS.items()}
    values['joint_limits_deg'] = tuple(values['joint_limits_deg'])
    return replace(PreflightSettings(**values), **changes)


START = TcpPose(.52, -.17, .19, 135., 0., -108.)


def make_checker(arm=None, fk_joint_decimals=None, **changes):
    arm = arm or SerialArm()
    inverse = MagicMock(side_effect=arm.inverse)
    forward = MagicMock(side_effect=arm.forward)
    return SampledCartesianPreflight(inverse, forward, lambda: None, settings(**changes),
                                     fk_joint_decimals=fk_joint_decimals)


def flip_segments():
    high = replace(START, z=.22)
    flipped = replace(high, rx=225.)
    returned = replace(flipped, z=START.z)
    restored = replace(returned, rx=180.)
    return [('lift', high), ('flip', flipped), ('return', returned), ('restore', restored)]


@pytest.mark.parametrize('fk_joint_decimals', [None, 6])
def test_local_inverse_matches_reference_arm_and_keeps_all_path_checks(fk_joint_decimals):
    arm = SerialArm()
    initial = arm.inverse(START, [0.] * 6)
    fast = make_checker(arm, fk_joint_decimals=fk_joint_decimals)
    baseline = make_checker(arm, fk_joint_decimals=fk_joint_decimals, fast_ik=False)
    endpoints = flip_segments()
    final_fast = fast.check(START, initial, endpoints)
    final_baseline = baseline.check(START, initial, endpoints)
    np.testing.assert_allclose(final_fast, final_baseline, atol=1e-6)
    assert fast.model_status == 'verified'
    assert fast.local_ik_count > 0
    assert fast.local_ik_fallbacks == 0
    assert fast.ik_query_count < baseline.ik_query_count * .25
    assert fast.sample_count == baseline.sample_count
    assert fast.refinement_count == baseline.refinement_count
    assert fast.maximum_joint_step_deg <= 20.
    assert fast.minimum_sigma == pytest.approx(baseline.minimum_sigma, abs=1e-6)
    # Each segment endpoint still has a real reference/controller IK query.
    for _, end in endpoints:
        assert any(np.linalg.norm(np.array([c.args[0].x-end.x, c.args[0].y-end.y, c.args[0].z-end.z])) < 1e-9
                   and abs((c.args[0].rx-end.rx + 180.) % 360. - 180.) < 1e-6
                   for c in fast.inverse.call_args_list)
    # Controller FK and derivative audits remain; fewer IK calls do not
    # replace validation with the local model's prediction.
    assert fast.fk_query_count >= 2 * fast.sample_count


def test_quantized_controller_fk_still_validates_local_inverse_samples():
    class QuantizedArm(SerialArm):
        def forward(self, joints):
            q = [float(f'{value:.6f}') for value in joints]
            p = super().forward(q)
            return TcpPose(*(float(np.float32(v*1000.))/1000. for v in (p.x, p.y, p.z)),
                           *(round(v, 4) for v in (p.rx, p.ry, p.rz)))

    arm = QuantizedArm()
    checker = make_checker(arm, fk_joint_decimals=6)
    q = arm.inverse(START, [0.] * 6)
    final = checker.check(START, q, flip_segments())
    assert checker.model_status == 'verified'
    assert checker.local_ik_count > 0
    assert checker.ik_query_count < checker.inverse_attempt_count * .5
    assert checker.minimum_sigma >= .025
    actual = arm.forward(final)
    end = flip_segments()[-1][1]
    assert np.linalg.norm([actual.x-end.x, actual.y-end.y, actual.z-end.z]) < 1e-5


@pytest.mark.parametrize('version', [4, 5])
def test_launch_exposes_the_fast_ik_rollback_switch(version):
    import ast
    from pathlib import Path
    path = Path(__file__).parents[1] / 'launch' / f'cosmetic_box_single_arm_cycle_v{version}_d435.launch.py'
    tree = ast.parse(path.read_text())
    key = 'bottom_near_cube_preflight_fast_ik'
    declarations = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and call.func.id == 'DeclareLaunchArgument'
                    and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == key]
    assert len(declarations) == 1
    assert any(k.arg == 'default_value' and isinstance(k.value, ast.Constant) and k.value.value == 'true'
               for k in declarations[0].keywords)
    assert any(isinstance(node, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == key
                                               for k in node.keys) for node in ast.walk(tree))


def test_local_inverse_handles_independent_dh_chain_with_tool_offset():
    # Load only the pure independent DH fixture; it creates no robot client.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('_dh_arm_fixture', Path(__file__).with_name('test_controller_kinematics_model.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    arm = fixture.DHArm()
    checker = make_checker()
    checker.forward = arm.forward
    q = fixture.REFERENCE
    checker._check_sigma(q, 'model calibration', arm.forward(q))
    for offset in (np.array([1., -2., 1.5, 3., -2., 1.]),
                   np.array([-3., 1., 2., -4., 3., -2.])):
        target = arm.forward(q + offset)
        result = checker._model.inverse_near(
            target, q, limits=checker.settings.joint_limits_deg,
            joint_margin_deg=2., min_sigma=.025, length_scale_m=.5,
            check_active=lambda: None)
        checker._model.require_match(result, target)
        np.testing.assert_allclose(result, q + offset, atol=1e-4)


@pytest.mark.parametrize('fast_jacobian, fast_ik', [(False, True), (True, False)])
def test_disabled_model_or_switch_uses_controller_ik(fast_jacobian, fast_ik):
    arm = SerialArm()
    checker = make_checker(arm, fast_jacobian=fast_jacobian, fast_ik=fast_ik)
    checker.check(START, arm.inverse(START, [0.] * 6), flip_segments())
    assert checker.local_ik_count == checker.local_ik_attempts == 0
    assert checker.ik_query_count == checker.inverse_attempt_count


def test_nonconvergence_falls_back_to_controller_ik(monkeypatch):
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    monkeypatch.setattr(checker._model, 'inverse_near', MagicMock(side_effect=LocalInverseUnavailable('iteration limit')))
    checker.check(START, q, flip_segments())
    assert checker.local_ik_fallbacks > 0
    assert checker.local_ik_count == 0
    assert checker.ik_query_count == checker.inverse_attempt_count


def test_bad_local_solution_is_requeried_and_validated_by_controller(monkeypatch):
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    original = checker._model.inverse_near

    def inaccurate(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0] += 2.  # Large Cartesian error, still inside joint limits.
        return result

    monkeypatch.setattr(checker._model, 'inverse_near', inaccurate)
    final = checker.check(START, q, flip_segments())
    expected = arm.inverse(flip_segments()[-1][1], final)
    np.testing.assert_allclose(final, expected, atol=1e-6)
    assert checker.local_ik_count == 0
    assert checker.local_ik_fallbacks == 1
    assert checker.model_status.startswith('fallback(')
    assert checker.controller_jacobian_count > 0
    assert checker.minimum_sigma >= checker.settings.min_sigma


def test_controller_fk_failure_after_local_solve_propagates():
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    checker.forward = MagicMock(side_effect=RuntimeError('PositiveKin socket failure'))
    with pytest.raises(RuntimeError, match='PositiveKin socket failure'):
        checker._advance(START, np.array(q), replace(START, z=.195), 'interior')
    checker.inverse.assert_not_called()


def test_stop_during_local_iterations_propagates_without_controller_retry():
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    checker.check_active = MagicMock(side_effect=[None, None, RuntimeError('operator Stop')])
    with pytest.raises(RuntimeError, match='operator Stop'):
        checker._advance(START, np.array(q), replace(START, z=.195), 'interior')
    checker.inverse.assert_not_called()


def test_borderline_local_inverse_uses_original_controller_checks():
    arm = SerialArm()
    start, end = replace(START, ry=80.), replace(START, ry=84.)
    q = arm.inverse(start, [0.] * 6)
    end_q = arm.inverse(end, q)
    direct = make_checker(arm, fast_jacobian=False, fast_ik=False)
    sigma = direct._check_sigma(end_q, 'reference')
    checker = make_checker(arm, min_sigma=sigma - .01)
    checker._check_sigma(q, 'start', start)
    checker._advance(start, np.array(q), end, 'borderline interior')
    assert checker.local_ik_fallbacks == 1
    assert checker.local_ik_count == 0
    assert checker.ik_query_count == 1
    assert checker.controller_jacobian_count >= 1


def test_singular_path_is_rejected_after_successful_local_interior_samples():
    arm = SerialArm()
    start, end = replace(START, ry=60.), replace(START, ry=90.)
    checker = make_checker(arm)
    q = arm.inverse(start, [0.] * 6)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        with pytest.raises(NearCubePreflightError, match='singularity margin'):
            checker.check(start, q, [('singular wrist', end)])
    assert checker.local_ik_count > 0
    assert checker.ik_query_count > 0


def test_wrapped_local_branch_is_never_accepted(monkeypatch):
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    original = checker._model.inverse_near

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        result[3] += -360. if result[3] > 0. else 360.
        return result

    monkeypatch.setattr(checker._model, 'inverse_near', wrapped)
    with pytest.raises(NearCubePreflightError, match='unresolved joint step'):
        checker._advance(START, np.array(q), replace(START, z=.195), 'wrapped interior')


def test_local_inverse_iterations_and_total_query_budget_are_bounded(monkeypatch):
    arm = SerialArm()
    checker = make_checker(arm)
    q = arm.inverse(START, [0.] * 6)
    checker._check_sigma(q, 'start', START)
    model = checker._model
    original = model.forward_and_jacobian

    def stalled(joints, length):
        pose, _ = original(q, length)
        return pose, np.eye(6)

    monkeypatch.setattr(model, 'forward_and_jacobian', stalled)
    active = MagicMock()
    with pytest.raises(LocalInverseUnavailable, match='16 iterations'):
        model.inverse_near(replace(START, z=.22), q,
                           limits=checker.settings.joint_limits_deg, joint_margin_deg=2.,
                           min_sigma=.025, length_scale_m=.5, check_active=active)
    assert active.call_count == 16
    checker.inverse_attempt_count = 256
    with pytest.raises(NearCubePreflightError, match='256 IK queries'):
        checker._advance(START, np.array(q), replace(START, z=.195), 'over budget')
    checker.inverse.assert_not_called()


def test_fk_cache_matches_the_six_decimal_positivekin_wire_format():
    arm = SerialArm()
    forward = MagicMock(side_effect=arm.forward)
    checker = SampledCartesianPreflight(arm.inverse, forward, lambda: None,
                                        settings(), fk_joint_decimals=6)
    q = np.array(arm.inverse(START, [0.] * 6))
    q = np.round(q, 6)
    a, b = q.copy(), q.copy()
    a[0] += 2e-8
    b[0] -= 2e-8
    assert checker._fk(a) is checker._fk(b)
    assert checker.fk_query_count == 1
    forward.assert_called_once_with(q.tolist())
