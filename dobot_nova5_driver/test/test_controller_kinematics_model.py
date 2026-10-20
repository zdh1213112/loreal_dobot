"""Independent DH-chain checks for the controller-derived POE optimization."""
import math
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.controller_kinematics_model import (
    ControllerKinematicsModel, ModelMismatch,
)
from dobot_nova5_driver.near_cube_motion_safety import (
    NearCubePreflightError, PreflightModelSession, PreflightSettings, SampledCartesianPreflight,
)


LIMITS = (-360., 360., -180., 180., -160., 160., -360., 360., -360., 360., -360., 360.)
REFERENCE = np.array([20., -40., 70., 15., 45., -20.])


class DHArm:
    """Six independent revolute joints; no implementation POE helpers used."""
    def __init__(self):
        self.user = np.eye(4)
        self.user[:3, :3] = Rotation.from_euler('xyz', [7., -5., 13.], degrees=True).as_matrix()
        self.user[:3, 3] = [.02, -.03, .01]
        self.tool = np.eye(4)
        self.tool[:3, :3] = Rotation.from_euler('xyz', [5., -4., 8.], degrees=True).as_matrix()
        self.tool[:3, 3] = [.00046, -.00255, .22259]

    def forward(self, joints):
        matrix = self.user.copy()
        a = [0., -.3, -.25, 0., 0., 0.]
        d = [.12, 0., 0., .1, .08, .08]
        alpha = [math.pi / 2, 0., 0., math.pi / 2, -math.pi / 2, 0.]
        for i, theta in enumerate(np.radians(joints)):
            c, s = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha[i]), math.sin(alpha[i])
            matrix = matrix @ np.array([
                [c, -s*ca, s*sa, a[i]*c],
                [s, c*ca, -c*sa, a[i]*s],
                [0., sa, ca, d[i]], [0., 0., 0., 1.],
            ])
        matrix = matrix @ self.tool
        rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
        return TcpPose(*matrix[:3, 3], *rpy)


def settings(fast=True, **values):
    return PreflightSettings(.005, 10., 20., .025, .5, 2., LIMITS, fast, **values)


def checker(arm, fast=True, config=None, session=None):
    return SampledCartesianPreflight(MagicMock(), arm.forward, lambda: None,
                                     config or settings(fast), session)


@pytest.mark.parametrize('offset', [np.zeros(6), np.array([-30., 15., -10., 20., -15., 35.])])
def test_derived_model_matches_independent_dh_chain_and_jacobian(offset):
    arm = DHArm()
    origin = REFERENCE + offset
    model = ControllerKinematicsModel.from_controller(origin, arm.forward(origin), arm.forward, LIMITS)
    for shift in (np.zeros(6), np.array([20., -15., 10., 35., -20., 30.]),
                  np.array([-45., 10., 15., -20., 40., -30.])):
        q = origin + shift
        model.require_match(q, arm.forward(q))
        _, local = model.forward_and_jacobian(q, .5)
        direct = checker(arm, fast=False)._controller_jacobian(q, arm.forward(q))
        # Original controller differences are one-sided at +0.25deg. Their
        # expected O(step) error is absent in the exact local Jacobian.
        np.testing.assert_allclose(local, direct, atol=.004, rtol=.01)


def test_quantized_controller_output_is_accepted():
    arm = DHArm()

    def rounded(joints):
        p = arm.forward(joints)
        # Fine-resolution reference fixture; not a firmware precision guarantee.
        return TcpPose(*(round(v, 9) for v in (p.x, p.y, p.z)),
                       *(round(v, 6) for v in (p.rx, p.ry, p.rz)))

    model = ControllerKinematicsModel.from_controller(REFERENCE, rounded(REFERENCE), rounded, LIMITS)
    model.require_match(REFERENCE + [20., -10., 15., 25., -20., 30.],
                        rounded(REFERENCE + [20., -10., 15., 25., -20., 30.]))


def test_controller_precision_stress_keeps_fast_path_during_large_rotation():
    arm = DHArm()

    def quantized(joints):
        p = arm.forward(joints)
        # Stress fixture, not a claim about a particular firmware's format:
        # float32 mm positions and 0.0001deg attitude quantization.
        return TcpPose(*(float(np.float32(v*1000.))/1000. for v in (p.x, p.y, p.z)),
                       *(round(v, 4) for v in (p.rx, p.ry, p.rz)))

    instance = checker(arm)
    instance.forward = quantized
    legacy = checker(arm, fast=False)
    legacy.forward = quantized
    sigmas = [[], []]
    for i in range(34):
        q = REFERENCE + np.array([.1, .08, -.05, 3., .04, -.1]) * i
        for which, target in enumerate((instance, legacy)):
            sigmas[which].append(target._check_sigma(q, 'large rotation', quantized(q)))
    assert instance.model_status == 'verified'
    assert instance.local_jacobian_count == 34
    assert instance.controller_jacobian_count == 0
    assert instance.fk_query_count < legacy.fk_query_count * .40
    np.testing.assert_allclose(sigmas[0], sigmas[1], atol=.004)


def test_calibration_queries_remain_inside_tight_joint_limits():
    arm = DHArm()
    limits = np.column_stack((REFERENCE - .6, REFERENCE + .6)).reshape(-1)
    probes = []

    def forward(joints):
        probes.append(np.array(joints))
        return arm.forward(joints)

    model = ControllerKinematicsModel.from_controller(REFERENCE, arm.forward(REFERENCE), forward, limits)
    for probe in probes:
        assert np.all(probe >= limits[::2])
        assert np.all(probe <= limits[1::2])
    model.require_match(REFERENCE + .1, arm.forward(REFERENCE + .1))


def test_same_34_checks_use_fewer_controller_queries_with_unchanged_threshold():
    arm = DHArm()
    fast = checker(arm)
    legacy = checker(arm, fast=False)
    sigmas = [[], []]
    for i in range(34):
        q = REFERENCE + np.array([.1, .08, -.05, .12, .04, -.1]) * i
        p = arm.forward(q)
        for instance, values in zip((fast, legacy), sigmas):
            values.append(instance._check_sigma(q, 'known DH path', desired=p))
    assert fast.model_status == 'verified'
    assert fast.local_jacobian_count == 34
    assert fast.controller_jacobian_count == 0
    assert fast.fk_query_count < legacy.fk_query_count * .40
    assert legacy.fk_query_count == 34 * 7
    np.testing.assert_allclose(sigmas[0], sigmas[1], atol=.004)


def test_model_drift_falls_back_and_remains_disabled():
    arm = DHArm()
    instance = checker(arm)
    instance._check_sigma(REFERENCE, 'initial', arm.forward(REFERENCE))
    assert instance.local_jacobian_count == 1
    arm.user[0, 3] += .003
    for offset in (7., 8.):
        q = REFERENCE + offset
        instance._check_sigma(q, 'changed User', arm.forward(q))
    assert instance._model is None
    assert instance.model_status.startswith('fallback(')
    assert instance.controller_jacobian_count == 2
    assert instance.local_jacobian_count == 1


def test_perturbed_point_audit_detects_hidden_model_drift():
    arm = DHArm()
    instance = checker(arm)
    instance._check_sigma(REFERENCE, 'initial', arm.forward(REFERENCE))
    original = instance.forward
    next_q = REFERENCE + 7.

    def changed_derivative(joints):
        p = original(joints)
        # Same base pose at next_q, changed derivative of every joint.
        drift = .1 * np.sum(np.asarray(joints) - next_q)
        return replace(p, x=p.x + drift)

    instance.forward = changed_derivative
    instance._check_sigma(next_q, 'derivative drift', changed_derivative(next_q))
    assert instance._model is None
    assert instance.controller_jacobian_count == 1


def test_borderline_sigma_uses_controller_differences():
    arm = DHArm()
    sigma = checker(arm, fast=False)._check_sigma(REFERENCE, 'reference')
    config = replace(settings(), min_sigma=sigma - .01)
    instance = checker(arm, config=config)
    result = instance._check_sigma(REFERENCE, 'borderline')
    assert instance.controller_jacobian_count == 1
    assert instance.local_jacobian_count == 0
    assert result == pytest.approx(sigma)


@pytest.mark.parametrize('fast', [True, False])
def test_true_wrist_singularity_is_still_rejected(fast):
    arm = DHArm()
    q = REFERENCE.copy()
    q[4] = 0.
    with pytest.raises(NearCubePreflightError, match='singularity margin'):
        checker(arm, fast=fast)._check_sigma(q, 'straight wrist')


def test_invalid_model_uses_original_checks():
    arm = DHArm()

    def coupled(joints):
        q = np.array(joints, dtype=float)
        q[4] += .1 * (q[0] - REFERENCE[0])**2
        return arm.forward(q)

    instance = SampledCartesianPreflight(MagicMock(), coupled, lambda: None, settings())
    instance._check_sigma(REFERENCE, 'non-serial compensation', coupled(REFERENCE))
    assert instance.model_status.startswith('fallback(')
    assert instance.local_jacobian_count == 0
    assert instance.controller_jacobian_count == 1


def test_axis_normalization_cannot_hide_nonunit_joint_gearing():
    arm = DHArm()

    def geared(joints):
        q = np.array(joints, dtype=float)
        q[3] *= 1.0002
        return arm.forward(q)

    instance = SampledCartesianPreflight(MagicMock(), geared, lambda: None, settings())
    instance._check_sigma(REFERENCE, 'nonunit gearing', geared(REFERENCE))
    assert instance.model_status.startswith('fallback(')
    assert instance.controller_jacobian_count == 1
    assert instance.local_jacobian_count == 0


def test_model_verification_errors_do_not_hide_controller_failure():
    arm = DHArm()
    instance = checker(arm)
    instance.forward = MagicMock(side_effect=RuntimeError('PositiveKin socket failure'))
    with pytest.raises(RuntimeError, match='PositiveKin socket failure'):
        instance._check_sigma(REFERENCE, 'query failure')


def test_pose_only_evaluation_agrees_with_independent_fk_and_full_evaluation():
    arm = DHArm()
    model = ControllerKinematicsModel.from_controller(REFERENCE, arm.forward(REFERENCE), arm.forward, LIMITS)
    for offset in (0., 15., -25.):
        q = REFERENCE + np.array([1., -.5, .3, .8, -.6, .9]) * offset
        predicted = model.forward(q)
        full, _ = model.forward_and_jacobian(q, .5)
        np.testing.assert_allclose(predicted, full, atol=1e-12)
        model.require_match(q, arm.forward(q), predicted=predicted)


def test_reused_model_repeats_validation_and_saves_twelve_fk_queries():
    arm = DHArm()
    session = PreflightModelSession()
    first = checker(arm, session=session)
    first._check_sigma(REFERENCE, 'pickup')
    second = checker(arm, session=session)
    fresh = checker(arm)
    for index in range(30):
        q = REFERENCE + np.array([.1, .08, -.05, .12, .04, -.1]) * (index + 5)
        actual = arm.forward(q)
        a = second._check_sigma(q, 'flip', actual)
        b = fresh._check_sigma(q, 'flip', actual)
        assert a == pytest.approx(b, abs=1e-10)
    assert second.model_reused
    assert second._model is first._model
    assert second.model_status == 'verified'
    assert fresh.fk_query_count - second.fk_query_count == 12
    assert second.local_jacobian_count == fresh.local_jacobian_count == 30
    # Every check starts a fresh FK cache: the eight validation probes, base
    # poses and rotating derivative audits were queried again, not cached.
    assert second.fk_query_count >= 8 + 30


def test_reuse_with_changed_tool_rebuilds_from_fresh_controller_queries():
    arm = DHArm()
    session = PreflightModelSession()
    first = checker(arm, session=session)
    first._check_sigma(REFERENCE, 'pickup')
    previous = session.model
    arm.tool[0, 3] += .003
    second = checker(arm, session=session)
    q = REFERENCE + 7.
    result = second._check_sigma(q, 'new tool', arm.forward(q))
    assert not second.model_reused
    assert second._model is not previous
    assert second.model_status == 'verified'
    assert result == pytest.approx(checker(arm, fast=False)._check_sigma(q, 'reference'), abs=.004)


def test_reuse_validation_cannot_hide_new_nonserial_compensation():
    arm = DHArm()
    session = PreflightModelSession()
    checker(arm, session=session)._check_sigma(REFERENCE, 'pickup')
    second = checker(arm, session=session)
    q = REFERENCE + 7.

    def coupled(joints):
        changed = np.array(joints, dtype=float)
        changed[4] += .1 * (changed[0] - q[0])**2
        return arm.forward(changed)

    second.forward = coupled
    second._check_sigma(q, 'changed compensation', coupled(q))
    assert second._model is session.model is None
    assert second.model_status.startswith('fallback(')
    assert second.controller_jacobian_count == 1
    third = checker(arm, session=session)
    third.forward = coupled
    third._check_sigma(q + .1, 'fallback stays active', coupled(q + .1))
    assert third.fk_query_count == 7
    assert third.controller_jacobian_count == 1
    assert third.local_jacobian_count == 0


def test_reused_model_still_uses_controller_for_borderline_sigma():
    arm = DHArm()
    session = PreflightModelSession()
    checker(arm, session=session)._check_sigma(REFERENCE, 'pickup')
    q = REFERENCE + 7.
    direct = checker(arm, fast=False)._check_sigma(q, 'reference')
    second = checker(arm, config=replace(settings(), min_sigma=direct - .01), session=session)
    assert second._check_sigma(q, 'borderline') == pytest.approx(direct)
    assert second.model_reused
    assert second.controller_jacobian_count == 1
    assert second.local_jacobian_count == 0


def test_reused_model_still_rejects_a_singular_wrist():
    arm = DHArm()
    session = PreflightModelSession()
    checker(arm, session=session)._check_sigma(REFERENCE, 'pickup')
    q = REFERENCE.copy()
    q[4] = 0.
    second = checker(arm, session=session)
    with pytest.raises(NearCubePreflightError, match='singularity margin'):
        second._check_sigma(q, 'singular wrist')
    assert second.controller_jacobian_count == 1


def test_reuse_probe_failure_is_not_replaced_with_a_successful_local_result():
    arm = DHArm()
    session = PreflightModelSession()
    checker(arm, session=session)._check_sigma(REFERENCE, 'pickup')
    second = checker(arm, session=session)
    q = REFERENCE + 7.
    second.forward = MagicMock(side_effect=[arm.forward(q), RuntimeError('PositiveKin socket failure')])
    with pytest.raises(RuntimeError, match='PositiveKin socket failure'):
        second._check_sigma(q, 'failed validation')
    assert second.local_jacobian_count == second.controller_jacobian_count == 0
    assert second.forward.call_count == 2
