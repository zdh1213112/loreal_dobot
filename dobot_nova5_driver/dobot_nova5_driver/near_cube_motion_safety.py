"""Read-only kinematic checks and fault containment for V4/V5 cube flips.

Numerical sampling is a conservative screen, not a replacement for the
controller's own path planner or its singularity protection.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import math
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .controller_v3 import TcpPose
from .dobot_dh_api_v3 import GRIP_REACHED
from .controller_kinematics_model import (
    ControllerKinematicsModel, LocalInverseUnavailable, ModelMismatch,
)


class NearCubePreflightError(RuntimeError):
    """A geometric/kinematic risk found before closing the jaws."""


class NearCubeHeldMotionFailure(RuntimeError):
    """A failure with a held box; never route through open-jaw grasp retries."""


@dataclass
class PreflightModelSession:
    model: ControllerKinematicsModel | None = None
    fallback_reason: str = ""


_model_sessions = ContextVar("near_cube_preflight_model_sessions", default=None)


def with_near_cube_preflight_session(method):
    """Scope verified-model reuse to one flip, never another cycle/thread."""
    @wraps(method)
    def wrapped(*args, **kwargs):
        token = _model_sessions.set({})
        try:
            return method(*args, **kwargs)
        finally:
            _model_sessions.reset(token)
    return wrapped


NEAR_CUBE_SAFETY_DEFAULTS = {
    "bottom_near_cube_preflight_linear_step_m": 0.005,
    "bottom_near_cube_preflight_rotation_step_deg": 10.0,
    "bottom_near_cube_preflight_joint_step_deg": 20.0,
    "bottom_near_cube_preflight_fast_jacobian": True,
    "bottom_near_cube_preflight_fast_ik": True,
    "bottom_near_cube_preflight_min_sigma": 0.025,
    "bottom_near_cube_preflight_length_scale_m": 0.5,
    "bottom_near_cube_preflight_joint_margin_deg": 2.0,
    # Nova5 joint ranges; applied only to this Nova5 near-cube procedure.
    "bottom_near_cube_preflight_joint_limits_deg": [
        -360.0, 360.0, -180.0, 180.0, -160.0, 160.0,
        -360.0, 360.0, -360.0, 360.0, -360.0, 360.0,
    ],
}


def rotation(pose: TcpPose) -> Rotation:
    return Rotation.from_euler("xyz", [pose.rx, pose.ry, pose.rz], degrees=True)


def position(pose: TcpPose) -> np.ndarray:
    return np.asarray([pose.x, pose.y, pose.z], dtype=float)


@dataclass(frozen=True)
class PreflightSettings:
    linear_step_m: float
    rotation_step_deg: float
    joint_step_deg: float
    min_sigma: float
    length_scale_m: float
    joint_margin_deg: float
    joint_limits_deg: tuple[float, ...]
    fast_jacobian: bool = True
    fast_ik: bool = True

    def __post_init__(self):
        ranges = (
            (self.linear_step_m, .001, .020),
            (self.rotation_step_deg, 1., 15.),
            (self.joint_step_deg, 1., 45.),
            (self.min_sigma, .001, .25),
            (self.length_scale_m, .1, 2.),
            (self.joint_margin_deg, .5, 10.),
        )
        if any(not math.isfinite(v) or not low <= v <= high for v, low, high in ranges):
            raise ValueError("invalid near-cube sampled-preflight settings")
        limits = np.asarray(self.joint_limits_deg, dtype=float)
        if limits.shape != (12,) or not np.all(np.isfinite(limits)):
            raise ValueError("near-cube joint limits must contain six finite lower/upper pairs")
        if np.any(limits[1::2] - limits[::2] <= 2. * self.joint_margin_deg):
            raise ValueError("near-cube joint limits leave no interior margin")


class SampledCartesianPreflight:
    def __init__(self, inverse: Callable, forward: Callable,
                 check_active: Callable, settings: PreflightSettings,
                 model_session: PreflightModelSession | None = None,
                 fk_joint_decimals: int | None = None):
        self.inverse = inverse
        self.forward = forward
        self.check_active = check_active
        self.settings = settings
        self._fk_joint_decimals = fk_joint_decimals
        self._fk_cache = {}
        self.sample_count = 0
        self.minimum_sigma = float("inf")
        self.maximum_joint_step_deg = 0.
        self.maximum_trial_joint_step_deg = 0.
        self.refinement_count = 0
        self.ik_query_count = 0
        self.inverse_attempt_count = 0
        self.local_ik_count = 0
        self.local_ik_attempts = 0
        self.local_ik_fallbacks = 0
        self.local_ik_s = 0.
        self.fk_query_count = 0
        self.local_jacobian_count = 0
        self.controller_jacobian_count = 0
        self.ik_query_s = 0.
        self.fk_query_s = 0.
        self._model_session = model_session
        self._model_attempted = bool(model_session and model_session.fallback_reason)
        self._model = None
        self.model_reused = False
        self.model_status = "pending" if settings.fast_jacobian else "disabled"
        if settings.fast_jacobian and self._model_attempted:
            self.model_status = f"fallback({model_session.fallback_reason})"

    def _fk(self, joints):
        self.check_active()
        key = tuple(float(v) for v in joints)
        if self._fk_joint_decimals is not None:
            # Match the PositiveKin wire format exactly. Different Python
            # floats that serialize to the same query must share one result.
            key = tuple(float(f"{v:.{self._fk_joint_decimals}f}") for v in key)
        if key not in self._fk_cache:
            self.fk_query_count += 1
            started = time.perf_counter()
            try:
                pose = self.forward(list(key))
            finally:
                self.fk_query_s += time.perf_counter() - started
            values = [pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz]
            if not all(math.isfinite(float(v)) for v in values):
                raise RuntimeError("PositiveKin returned a non-finite pose; preflight unavailable")
            self._fk_cache[key] = pose
        return self._fk_cache[key]

    def _reject_model(self, reason):
        self._model = None
        self.model_status = f"fallback({reason})"
        if self._model_session is not None:
            self._model_session.model = None
            self._model_session.fallback_reason = str(reason)

    def _initialize_model(self, joints, base):
        # Reuse only the mathematical model, not previous FK/IK results.
        # The new check has a fresh FK cache and repeats all eight independent
        # controller probes at its actual start before trusting that model.
        candidate = self._model_session.model if self._model_session is not None else None
        if candidate is not None:
            try:
                candidate.require_match(joints, base)
                candidate.validate_controller(joints, self._fk, self.settings.joint_limits_deg)
                self._model = candidate
                self.model_reused = True
            except ModelMismatch:
                # Reference precision can degrade over a large wrist change.
                # Rebuild at the new reference, with the original tolerances.
                pass
        if self._model is None:
            self._model = ControllerKinematicsModel.from_controller(
                joints, base, self._fk, self.settings.joint_limits_deg)
        self.model_status = "verified"
        if self._model_session is not None:
            self._model_session.model = self._model

    def _check_joints(self, joints, label):
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise NearCubePreflightError(f"{label}: inverse solution is not six finite joints")
        limits = np.asarray(self.settings.joint_limits_deg).reshape(6, 2)
        margin = self.settings.joint_margin_deg
        bad = np.where((q < limits[:, 0] + margin) | (q > limits[:, 1] - margin))[0]
        if len(bad):
            axis = int(bad[0])
            raise NearCubePreflightError(
                f"{label}: J{axis+1}={q[axis]:.2f}deg is inside the joint-limit margin"
            )
        return q

    def _controller_jacobian(self, joints, base):
        self.controller_jacobian_count += 1
        # Derivatives are per radian. Normalize translation by a characteristic
        # length before comparing the dimensionless singular values.
        delta_deg = .25
        delta_rad = math.radians(delta_deg)
        jacobian = np.empty((6, 6), dtype=float)
        for axis in range(6):
            shifted = np.array(joints, dtype=float)
            shifted[axis] += delta_deg
            moved = self._fk(shifted)
            jacobian[:3, axis] = (
                (position(moved) - position(base))
                / (delta_rad * self.settings.length_scale_m)
            )
            jacobian[3:, axis] = (
                (rotation(moved) * rotation(base).inv()).as_rotvec() / delta_rad
            )
        return jacobian

    def _sample_jacobian(self, joints, base):
        if self.settings.fast_jacobian and not self._model_attempted:
            self._model_attempted = True
            try:
                self._initialize_model(joints, base)
            except ModelMismatch as exc:
                self._reject_model(exc)
        if self._model is not None:
            try:
                predicted, jacobian = self._model.forward_and_jacobian(
                    joints, self.settings.length_scale_m)
                self._model.require_match(joints, base, predicted=predicted)
                # Audit one perturbed controller FK at every sample, rotating
                # the audited joint so all six axes are exercised repeatedly.
                probe = np.array(joints, dtype=float)
                probe[(self.local_jacobian_count + self.controller_jacobian_count) % 6] += .25
                self._model.require_match(probe, self._fk(probe))
                sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
                # Borderline values retain the original controller-difference
                # decision, including its 0.25deg numerical convention.
                if math.isfinite(sigma) and sigma > self.settings.min_sigma + .020:
                    self.local_jacobian_count += 1
                    return jacobian
            except ModelMismatch as exc:
                self._reject_model(exc)  # Do not retry in this flip session.
        return self._controller_jacobian(joints, base)

    def _check_sigma(self, joints, label, desired=None):
        base = self._fk(joints)
        if desired is not None:
            distance = float(np.linalg.norm(position(base) - position(desired)))
            angle = math.degrees((rotation(base).inv() * rotation(desired)).magnitude())
            if distance > .001 or angle > .5:
                raise NearCubePreflightError(
                    f"{label}: IK/FK mismatch={distance*1000:.2f}mm/{angle:.2f}deg"
                )
        jacobian = self._sample_jacobian(joints, base)
        sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        self.minimum_sigma = min(self.minimum_sigma, sigma)
        if not math.isfinite(sigma) or sigma < self.settings.min_sigma:
            raise NearCubePreflightError(
                f"{label}: singularity margin too small: sigma={sigma:.5f} "
                f"< {self.settings.min_sigma:.5f}; joints={np.round(joints, 3).tolist()}"
            )
        return sigma

    @staticmethod
    def _interpolate(start, end, ratio):
        xyz = position(start) + ratio * (position(end) - position(start))
        origin = rotation(start)
        vector = (origin.inv() * rotation(end)).as_rotvec()
        rpy = (origin * Rotation.from_rotvec(ratio * vector)).as_euler("xyz", degrees=True)
        return TcpPose(*map(float, xyz), *map(float, rpy))

    def _controller_inverse(self, target, q):
        self.check_active()
        if self.ik_query_count >= 256:
            raise NearCubePreflightError("near-cube preflight exceeds 256 IK queries")
        self.ik_query_count += 1
        started = time.perf_counter()
        try:
            return self.inverse(target, q.tolist())
        finally:
            self.ik_query_s += time.perf_counter() - started

    def _inverse_candidate(self, target, q, controller_anchor):
        if self.settings.fast_ik and self._model is not None and not controller_anchor:
            self.local_ik_attempts += 1
            started = time.perf_counter()
            try:
                result = self._model.inverse_near(
                    target, q, limits=self.settings.joint_limits_deg,
                    joint_margin_deg=self.settings.joint_margin_deg,
                    min_sigma=self.settings.min_sigma,
                    length_scale_m=self.settings.length_scale_m,
                    check_active=self.check_active,
                )
                return result, True
            except LocalInverseUnavailable:
                self.local_ik_fallbacks += 1
            finally:
                self.local_ik_s += time.perf_counter() - started
        return self._controller_inverse(target, q), False

    def _advance(self, start, q, target, tag, depth=0, controller_anchor=False):
        """Refine an oversized joint step before diagnosing discontinuity.

        A 10-degree Tool rotation can need more than 20 degrees of joint
        travel without changing IK branches. Re-seeding IK through a midpoint
        distinguishes that case while keeping every accepted step <=20deg.
        These are read-only queries, not commands sent to the robot.
        """
        self.check_active()
        if self.ik_query_count >= 256 or self.inverse_attempt_count >= 256:
            raise NearCubePreflightError("near-cube preflight exceeds 256 IK queries")
        self.inverse_attempt_count += 1
        solution, local = self._inverse_candidate(target, q, controller_anchor)
        next_q = self._check_joints(solution, tag)
        # Every candidate still has a real controller FK and derivative audit.
        # A local solution cannot survive a rejected model or a borderline
        # singularity value: re-query controller IK with the original seed.
        previous_minimum_sigma = self.minimum_sigma
        try:
            sigma = self._check_sigma(next_q, tag, desired=target)
        except NearCubePreflightError as exc:
            if not local:
                raise
            self.local_ik_fallbacks += 1
            self._reject_model(f"local inverse controller validation failed: {exc}")
            self.minimum_sigma = previous_minimum_sigma
            next_q = self._check_joints(self._controller_inverse(target, q), tag)
            sigma = self._check_sigma(next_q, tag, desired=target)
            local = False
        if local:
            if self._model is None or sigma <= self.settings.min_sigma + .020:
                self.local_ik_fallbacks += 1
                self.minimum_sigma = previous_minimum_sigma
                next_q = self._check_joints(self._controller_inverse(target, q), tag)
                sigma = self._check_sigma(next_q, tag, desired=target)
            else:
                self.local_ik_count += 1
        delta = np.abs(next_q - q)
        axis = int(np.argmax(delta))
        step = float(delta[axis])
        self.maximum_trial_joint_step_deg = max(self.maximum_trial_joint_step_deg, step)
        # Geometry and singularity were checked even for an oversized step.
        if step > self.settings.joint_step_deg:
            if depth >= 4:
                raise NearCubePreflightError(
                    f"{tag}: unresolved joint step after {depth} refinements: "
                    f"J{axis+1}={q[axis]:.3f}->{next_q[axis]:.3f}deg, "
                    f"change={step:.2f}deg > {self.settings.joint_step_deg:.2f}deg, "
                    f"sigma={sigma:.5f}; previous_joints={np.round(q, 3).tolist()}, "
                    f"candidate_joints={np.round(next_q, 3).tolist()}"
                )
            self.refinement_count += 1
            middle = self._interpolate(start, target, .5)
            middle_q = self._advance(start, q, middle, tag + "/first-half", depth + 1)
            # Query the endpoint again with the midpoint seed. Do not reuse
            # an endpoint solution that may have chosen another branch.
            return self._advance(middle, middle_q, target, tag + "/second-half", depth + 1,
                                 controller_anchor=controller_anchor)
        self.maximum_joint_step_deg = max(self.maximum_joint_step_deg, step)
        self.sample_count += 1
        return next_q

    def check(self, start: TcpPose, joints, segments):
        self.check_active()
        q = self._check_joints(joints, "path start")
        self._check_sigma(q, "path start", desired=start)
        previous = start
        for label, target in segments:
            values = [target.x, target.y, target.z, target.rx, target.ry, target.rz]
            if not all(math.isfinite(float(v)) for v in values):
                raise NearCubePreflightError(f"{label}: target contains non-finite values")
            offset = position(target) - position(previous)
            rot_start = rotation(previous)
            rotation_vector = (rot_start.inv() * rotation(target)).as_rotvec()
            count = max(
                1, math.ceil(float(np.linalg.norm(offset)) / self.settings.linear_step_m),
                math.ceil(math.degrees(float(np.linalg.norm(rotation_vector)))
                          / self.settings.rotation_step_deg),
            )
            if self.sample_count + count > 256:
                raise NearCubePreflightError("near-cube preflight exceeds 256 path samples")
            for index in range(1, count + 1):
                ratio = index / count
                pose = self._interpolate(previous, target, ratio)
                interval_start = self._interpolate(previous, target, (index - 1) / count)
                tag = f"{label} sample {index}/{count}"
                q = self._advance(interval_start, q, pose, tag, controller_anchor=index == count)
            previous = target
        return q.tolist()


class NearCubeMotionSafetyMixin:
    def _new_near_cube_preflight(self):
        def value(name):
            return self.get_parameter("bottom_near_cube_preflight_" + name).value
        settings = PreflightSettings(
            linear_step_m=float(value("linear_step_m")),
            rotation_step_deg=float(value("rotation_step_deg")),
            joint_step_deg=float(value("joint_step_deg")),
            min_sigma=float(value("min_sigma")),
            length_scale_m=float(value("length_scale_m")),
            joint_margin_deg=float(value("joint_margin_deg")),
            joint_limits_deg=tuple(value("joint_limits_deg")),
            fast_jacobian=bool(value("fast_jacobian")),
            fast_ik=bool(value("fast_ik")),
        )
        user = int(self.get_parameter("user_index").value)
        tool = int(self.get_parameter("command_tool_index").value)

        def active():
            self._require_cycle_active("during near-cube path preflight")
            if self.controller.robot_mode in (9, 11):
                raise RuntimeError("controller is in error/collision mode; motion preflight cancelled")

        def inverse(pose, nearby):
            active()
            try:
                return self.controller.inverse_kinematics(
                    pose, user_index=user, tool_index=tool, joint_near=nearby,
                )
            except RuntimeError as exc:
                active()  # A controller fault must not trigger relocation.
                if "InverseKin failed" not in str(exc):
                    raise
                raise NearCubePreflightError(str(exc)) from exc

        def forward(joints):
            return self.controller.forward_kinematics(joints, user_index=user, tool_index=tool)

        sessions = _model_sessions.get()
        session = None
        if sessions is not None and settings.fast_jacobian:
            # A GUI change of frame, settings, or controller starts a separate
            # model. The ContextVar also isolates simultaneous worker threads.
            key = (id(self.controller), user, tool, settings)
            session = sessions.setdefault(key, PreflightModelSession())
        return SampledCartesianPreflight(inverse, forward, active, settings, session,
                                         fk_joint_decimals=6)

    def _check_near_cube_path(self, start, segments, label):
        started = time.perf_counter()
        checker = self._new_near_cube_preflight()
        checker.check(start, self.controller.current_joint(), segments)
        total_s = time.perf_counter() - started
        other_s = max(0., total_s - checker.ik_query_s - checker.fk_query_s - checker.local_ik_s)
        self._publish_status(
            f"{label} sampled preflight passed: samples={checker.sample_count}, "
            f"minimum_sigma={checker.minimum_sigma:.5f}, "
            f"maximum_joint_step={checker.maximum_joint_step_deg:.2f}deg, "
            f"refinements={checker.refinement_count}, IK_queries={checker.ik_query_count}, "
            f"maximum_trial_joint_step={checker.maximum_trial_joint_step_deg:.2f}deg, "
            f"FK_queries={checker.fk_query_count}, local_J={checker.local_jacobian_count}, "
            f"controller_J={checker.controller_jacobian_count}, model={checker.model_status}, "
            f"model_reused={int(checker.model_reused)}, "
            f"local_IK={checker.local_ik_count}, local_IK_attempts={checker.local_ik_attempts}, "
            f"local_IK_fallbacks={checker.local_ik_fallbacks}, "
            f"IK_ms={checker.ik_query_s*1000:.1f}, FK_ms={checker.fk_query_s*1000:.1f}, "
            f"local_IK_ms={checker.local_ik_s*1000:.1f}, "
            f"other_ms={other_s*1000:.1f}"
        )

    def _check_near_cube_tool_rx(self, delta_deg, label):
        start = self._current_command_pose()
        target = self._tool_rx_target_pose(start, delta_deg)
        self._check_near_cube_path(start, [(label, target)], label)

    def _suspend_d405_after_fault(self, reason):
        if getattr(self, "d405_transform_suspended", False):
            return
        self.d405_transform_suspended = True
        try:
            self._set_d405_pretracking(False)
        except Exception as exc:
            self.get_logger().warning(f"D405 disarm publish failed after fault: {exc}")
        self._publish_status(f"D405 pose processing suspended after fault: {reason}")

    def _d405_transform_allowed(self):
        if getattr(self, "d405_transform_suspended", False):
            return False
        try:
            mode = self.controller.robot_mode
        except Exception:
            return True  # Existing transform error handling reports unready feedback.
        if mode in (9, 11):
            self._suspend_d405_after_fault("101 controller error/collision state")
            return False
        return True

    def _resume_d405_pose_processing(self):
        if not getattr(self, "d405_transform_suspended", False):
            return True
        if self.controller.robot_mode != 5:
            return False
        self.d405_transform_suspended = False
        return True

    def _require_near_cube_recovery_complete(self):
        if not getattr(self, "near_cube_held_motion_fault", ""):
            return
        # An explicit restart cannot carry a box to startup or open it in air.
        # Releasing/placing the material remains a supervised recovery action.
        state = self.gripper.read_grip_state()
        opening = float(self.gripper.read_position())
        if state != GRIP_REACHED or not math.isfinite(opening) or not .95 <= opening <= 1.0:
            raise RuntimeError(
                "near-cube held-motion fault: place the material safely and fully "
                "open the gripper before restarting; startup motion was not issued"
            )
        self.near_cube_held_motion_fault = ""
        self._publish_status("near-cube manual material recovery verified; startup may proceed")

    @contextmanager
    def _near_cube_held_motion_guard(self, stage):
        try:
            yield
        except Exception as exc:
            with getattr(self, "secondary_safety_lock", nullcontext()):
                self.near_cube_held_motion_fault = stage
                self.cycle_enabled = False
                resume = getattr(self, "secondary_auto_resume_requested", None)
                if resume is not None:
                    resume.clear()
            self._suspend_d405_after_fault(stage)
            stop_note = ""
            try:
                mode = self.controller.robot_mode
            except Exception:
                mode = None
            if mode in (None, 7, 8, 10):
                try:
                    self.controller.stop_motion()
                except Exception as stop_exc:
                    stop_note = f"; Stop unconfirmed: {stop_exc}"
            feedback_note = "feedback pose unavailable"
            try:
                pose, user, tool, received_at = self.controller.current_feedback_tcp_pose_with_timestamp()
                age = time.time() - received_at
                if 0. <= age <= .10:
                    feedback_note = (
                        f"feedback User={user}/Tool={tool}, age={age*1000:.0f}ms, "
                        f"XYZ=({pose.x*1000:.1f},{pose.y*1000:.1f},{pose.z*1000:.1f})mm"
                    )
            except Exception:
                pass
            raise NearCubeHeldMotionFailure(
                f"near-cube held-motion failure at {stage}: {exc}; "
                f"{feedback_note}; last gripper command held; "
                f"automatic relocation/startup is blocked{stop_note}"
            ) from exc
