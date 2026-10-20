"""Flip-scoped serial-chain model derived from read-only controller FK.

Uses space-frame product of exponentials, with the measured configuration
as the reference. No robot dimensions, unverified URDF, or cross-cycle cache.
Reuse in a second preflight requires fresh independent controller validation.
See Modern Robotics 4.1.1 and 5.1.1 for the POE and space-Jacobian formulas.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation


class ModelMismatch(RuntimeError):
    """Local model is not an adequate match; use controller differences."""


class LocalInverseUnavailable(RuntimeError):
    """The bounded local continuation needs a controller inverse query."""


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(
        "xyz", [pose.rx, pose.ry, pose.rz], degrees=True).as_matrix()
    result[:3, 3] = [pose.x, pose.y, pose.z]
    return result


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(vector):
    theta = float(np.linalg.norm(vector))
    cross = skew(vector)
    if theta < 1e-5:
        a = .5 - theta**2 / 24. + theta**4 / 720.
        b = 1. / 6. - theta**2 / 120. + theta**4 / 5040.
    else:
        a = (1. - math.cos(theta)) / theta**2
        b = (theta - math.sin(theta)) / theta**3
    return np.eye(3) + a * cross + b * (cross @ cross)


def twist_exp(twist, angle):
    angular = twist[:3] * angle
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(angular).as_matrix()
    result[:3, 3] = left_jacobian(angular) @ (twist[3:] * angle)
    return result


class ControllerKinematicsModel:
    # Tight compared with the existing 1mm/0.5deg IK/FK check. Controllers
    # with unsuitable quantization or non-serial compensation fall back.
    POSITION_TOLERANCE_M = .000005
    ANGLE_TOLERANCE_RAD = math.radians(.0005)

    def __init__(self, reference_joints, home, axes):
        self.reference_joints = np.asarray(reference_joints, dtype=float)
        self.home = home
        self.axes = np.asarray(axes)

    @staticmethod
    def _probe_configuration(joints, limits):
        q = np.asarray(joints, dtype=float)
        limits = np.asarray(limits).reshape(6, 2)
        directions = np.where(q <= limits.mean(axis=1), 1., -1.)
        available = np.where(directions > 0., limits[:, 1] - q, q - limits[:, 0])
        return q, directions, np.minimum(30., .4 * available)

    @classmethod
    def from_controller(cls, joints, base_pose, forward, limits):
        q, directions, probe_sizes = cls._probe_configuration(joints, limits)
        home = pose_matrix(base_pose)
        axes = []
        # Estimate each screw from TWO queried poses separated by up to 30deg.
        # A 2deg difference from home amplifies controller orientation rounding
        # over a 90deg flip. These are read-only FK inputs, never robot moves.
        # Both probes point inward and remain within half the available range.
        for axis in range(6):
            step_deg = probe_sizes[axis] * directions[axis]
            first, second = q.copy(), q.copy()
            first[axis] += .25 * step_deg
            second[axis] += 1.25 * step_deg
            first_pose = pose_matrix(forward(first))
            inverse_first = np.eye(4)
            inverse_first[:3, :3] = first_pose[:3, :3].T
            inverse_first[:3, 3] = -first_pose[:3, :3].T @ first_pose[:3, 3]
            relative = pose_matrix(forward(second)) @ inverse_first
            angular = Rotation.from_matrix(relative[:3, :3]).as_rotvec()
            linear = np.linalg.solve(left_jacobian(angular), relative[:3, 3])
            screw = np.concatenate((angular, linear)) / math.radians(step_deg)
            angular_norm = float(np.linalg.norm(screw[:3]))
            # A revolute joint's angular axis has unit norm. Correct only
            # near-unit numerical error; general/prismatic screws stay intact.
            # Independent FK probes below still reject an unsuitable model.
            if abs(angular_norm - 1.) < .001:
                screw /= angular_norm
            axes.append(screw)
        model = cls(q, home, axes)
        model.validate_controller(q, forward, limits)
        return model

    def validate_controller(self, joints, forward, limits):
        """Fresh eight-pose validation, including when reused in one flip."""
        q, directions, probe_sizes = self._probe_configuration(joints, limits)
        # Independent single-axis and multi-axis poses must agree. The latter
        # detect an incorrect joint order or coupling not representable by POE.
        for axis in range(6):
            probe = q.copy()
            probe[axis] += .25 * directions[axis]
            self.require_match(probe, forward(probe))
        for offsets in ([.8, 1.1, .6, 1.3, .9, 1.2], [1.4, .5, 1.2, .7, 1.1, .4]):
            probe = q + directions * np.asarray(offsets) * probe_sizes / 2.
            self.require_match(probe, forward(probe))

    def forward(self, joints):
        """Evaluate the pose without constructing an unused Jacobian."""
        prefix = np.eye(4)
        delta = np.radians(np.asarray(joints) - self.reference_joints)
        for axis in range(6):
            prefix = prefix @ twist_exp(self.axes[axis], delta[axis])
        return prefix @ self.home

    def forward_and_jacobian(self, joints, length_scale_m):
        prefix = np.eye(4)
        spatial = np.empty((6, 6))
        delta = np.radians(np.asarray(joints) - self.reference_joints)
        for axis in range(6):
            screw = self.axes[axis]
            angular = prefix[:3, :3] @ screw[:3]
            linear = prefix[:3, :3] @ screw[3:] + np.cross(prefix[:3, 3], angular)
            spatial[:3, axis] = angular
            spatial[3:, axis] = linear
            prefix = prefix @ twist_exp(screw, delta[axis])
        pose = prefix @ self.home
        jacobian = np.empty((6, 6))
        for axis in range(6):
            jacobian[:3, axis] = (
                spatial[3:, axis] + np.cross(spatial[:3, axis], pose[:3, 3])
            ) / length_scale_m
        jacobian[3:, :] = spatial[:3, :]
        return pose, jacobian

    def require_match(self, joints, actual_pose, *, predicted=None):
        if predicted is None:
            predicted = self.forward(joints)
        actual = pose_matrix(actual_pose)
        distance = float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3]))
        angle = Rotation.from_matrix(predicted[:3, :3] @ actual[:3, :3].T).magnitude()
        if (not math.isfinite(distance + angle)
                or distance > self.POSITION_TOLERANCE_M
                or angle > self.ANGLE_TOLERANCE_RAD):
            raise ModelMismatch(f"local/controller FK difference={distance*1000:.5f}mm/{math.degrees(angle):.5f}deg")

    def inverse_near(self, target, nearby, *, limits, joint_margin_deg,
                     min_sigma, length_scale_m, check_active):
        """Bounded damped Newton continuation; controller FK must audit it.

        This returns a query candidate, never a motion command. Spatial
        angular error and TCP position error match the geometric Jacobian.
        Segment endpoints are still solved by the controller by the caller.
        """
        goal = pose_matrix(target)
        q = np.array(nearby, dtype=float)
        limits = np.asarray(limits).reshape(6, 2)
        low, high = limits[:, 0] + joint_margin_deg, limits[:, 1] - joint_margin_deg
        for _ in range(16):
            check_active()
            if not np.all(np.isfinite(q)) or np.any(q < low) or np.any(q > high):
                raise LocalInverseUnavailable("local inverse reached a joint-limit margin")
            predicted, jacobian = self.forward_and_jacobian(q, length_scale_m)
            xyz = goal[:3, 3] - predicted[:3, 3]
            angular = Rotation.from_matrix(goal[:3, :3] @ predicted[:3, :3].T).as_rotvec()
            sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            if not math.isfinite(sigma) or sigma <= min_sigma + .020:
                raise LocalInverseUnavailable("local inverse needs controller singularity checks")
            if np.linalg.norm(xyz) <= 1e-8 and np.linalg.norm(angular) <= 1e-7:
                return q.tolist()
            error = np.concatenate((xyz / length_scale_m, angular))
            try:
                delta = np.linalg.solve(jacobian.T @ jacobian + 1e-8 * np.eye(6),
                                        jacobian.T @ error)
            except np.linalg.LinAlgError as exc:
                raise LocalInverseUnavailable("local inverse solve is ill-conditioned") from exc
            step = np.degrees(delta)
            maximum = float(np.max(np.abs(step)))
            if not math.isfinite(maximum):
                raise LocalInverseUnavailable("local inverse step is not finite")
            q += step * min(1., 5. / max(maximum, 1e-12))
        raise LocalInverseUnavailable("local inverse did not converge within 16 iterations")
