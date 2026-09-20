"""Pure geometry and state helpers for the V3 turntable workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass


GRASP_DEPTH_RATIO = 0.75
MIN_TURNTABLE_ROI_SIZE_PX = 20


def classify_barcode_face(side_barcode: str, top_barcode: str) -> str:
    """Apply the production priority: D435 side, D405 top, then bottom."""

    if str(side_barcode).strip():
        return "side"
    if str(top_barcode).strip():
        return "top"
    return "bottom"


def turntable_departure_target_z(
    current_tcp_z_m: float,
    minimum_lift_m: float,
    safe_transfer_z_m: float,
) -> float:
    """Return the absolute Z for a straight post-grasp turntable departure.

    The departure must satisfy both the configured grasp-lift distance and the
    absolute safe transfer height.  Using an absolute target prevents a short
    object-dependent lift from handing control to a joint-space transfer while
    the box is still close to the turntable.
    """

    values = (current_tcp_z_m, minimum_lift_m, safe_transfer_z_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("turntable departure parameters must be finite")
    if float(minimum_lift_m) <= 0.0:
        raise ValueError("turntable minimum lift must be positive")
    return max(
        float(current_tcp_z_m) + float(minimum_lift_m),
        float(safe_transfer_z_m),
    )


def nearest_face_anchor_deg(
    current_joint_deg: float,
    reference_anchor_deg: float,
    face_step_deg: float,
    safe_joint_limit_deg: float,
) -> float:
    """Return the closest safe J6 anchor on the V2 90-degree face grid.

    V2 establishes its first standard barcode face at the transfer joint and
    spaces the remaining faces by ``barcode_flip_step_deg``.  V3 can arrive at
    the safe turntable departure height with any equivalent J6 winding, so it
    needs the complete periodic grid instead of only the four anchors visited
    by one scanner sweep.
    """

    values = (
        current_joint_deg,
        reference_anchor_deg,
        face_step_deg,
        safe_joint_limit_deg,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("face-anchor parameters must be finite")
    step_deg = abs(float(face_step_deg))
    safe_limit_deg = abs(float(safe_joint_limit_deg))
    if step_deg <= 0.0:
        raise ValueError("face-anchor step must be non-zero")
    if safe_limit_deg <= 0.0:
        raise ValueError("face-anchor joint limit must be positive")

    reference_deg = float(reference_anchor_deg)
    first_index = math.ceil((-safe_limit_deg - reference_deg) / step_deg)
    last_index = math.floor((safe_limit_deg - reference_deg) / step_deg)
    anchors = [
        reference_deg + float(index) * step_deg
        for index in range(first_index, last_index + 1)
    ]
    if not anchors:
        raise ValueError(
            "no standard face anchor lies inside the configured joint limit"
        )
    current_deg = float(current_joint_deg)
    return min(
        anchors,
        key=lambda anchor: (abs(current_deg - anchor), abs(anchor)),
    )


def normalize_image_roi(
    start: tuple[int, int],
    end: tuple[int, int],
    image_width: int,
    image_height: int,
    minimum_size_px: int = MIN_TURNTABLE_ROI_SIZE_PX,
) -> tuple[int, int, int, int] | None:
    """Clamp a mouse drag and return ``left, top, right, bottom`` bounds."""

    image_width = int(image_width)
    image_height = int(image_height)
    minimum_size_px = max(1, int(minimum_size_px))
    if image_width <= 0 or image_height <= 0:
        raise ValueError("ROI image dimensions must be positive")

    x0 = max(0, min(image_width - 1, int(start[0])))
    y0 = max(0, min(image_height - 1, int(start[1])))
    x1 = max(0, min(image_width - 1, int(end[0])))
    y1 = max(0, min(image_height - 1, int(end[1])))
    left, right = min(x0, x1), max(x0, x1) + 1
    top, bottom = min(y0, y1), max(y0, y1) + 1
    if right - left < minimum_size_px or bottom - top < minimum_size_px:
        return None
    return left, top, right, bottom


@dataclass(frozen=True)
class TurntableHeightCheck:
    estimated_surface_z_m: float
    configured_surface_z_m: float
    surface_error_m: float
    minimum_command_tcp_z_m: float
    command_tcp_z_m: float


@dataclass
class PlacementRetreatTrigger:
    """Latch one VLA place-and-retreat sequence from passive TCP feedback.

    A cycle must first observe the monitored TCP on the turntable side of the
    User-Y boundary.  It fires only after both User-Y has retreated across the
    boundary and User-Z is at or above the safe height continuously for the
    configured dwell.  Starting in the safe region therefore never fires.
    """

    place_y_m: float
    safe_z_m: float
    stable_s: float
    place_seen: bool = False
    safe_since_s: float | None = None

    def __post_init__(self) -> None:
        values = (self.place_y_m, self.safe_z_m, self.stable_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("placement-retreat trigger parameters must be finite")
        if self.stable_s < 0.0:
            raise ValueError("placement-retreat stable time must be non-negative")

    def reset(self) -> None:
        self.place_seen = False
        self.safe_since_s = None

    def update(self, tcp_y_m: float, tcp_z_m: float, now_s: float) -> bool:
        values = (tcp_y_m, tcp_z_m, now_s)
        if not all(math.isfinite(float(value)) for value in values):
            self.safe_since_s = None
            return False

        tcp_y_m = float(tcp_y_m)
        tcp_z_m = float(tcp_z_m)
        now_s = float(now_s)
        if not self.place_seen:
            if tcp_y_m >= float(self.place_y_m):
                self.place_seen = True
            return False

        safe_now = (
            tcp_y_m < float(self.place_y_m)
            and tcp_z_m >= float(self.safe_z_m)
        )
        if not safe_now:
            self.safe_since_s = None
            return False

        if self.safe_since_s is None:
            self.safe_since_s = now_s
        if now_s - self.safe_since_s < float(self.stable_s):
            return False

        self.reset()
        return True


def estimate_support_surface_z(
    vision_target_z_m: float,
    box_height_m: float,
    grasp_depth_ratio: float = GRASP_DEPTH_RATIO,
) -> float:
    """Estimate the horizontal support plane from the 75%-depth target.

    The D405 target is measured from the top of the box toward its support
    plane.  A ratio of 0.75 therefore leaves one quarter of the box height
    between the target and the support surface.
    """

    values = (vision_target_z_m, box_height_m, grasp_depth_ratio)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("turntable height inputs must be finite")
    if box_height_m <= 0.0:
        raise ValueError(f"box height must be positive, got {box_height_m}")
    if not 0.0 < grasp_depth_ratio < 1.0:
        raise ValueError(
            f"grasp depth ratio must be in (0, 1), got {grasp_depth_ratio}"
        )
    return float(vision_target_z_m) - (1.0 - float(grasp_depth_ratio)) * float(
        box_height_m
    )


def validate_turntable_grasp_height(
    *,
    vision_target_z_m: float,
    command_target_z_m: float,
    box_height_m: float,
    configured_surface_z_m: float,
    surface_tolerance_m: float,
    tcp_below_target_m: float,
    surface_clearance_m: float,
) -> TurntableHeightCheck:
    """Reject a target inconsistent with the measured turntable surface.

    ``tcp_below_target_m`` models any gripper/tool material below the command
    TCP.  The function intentionally rejects unsafe values rather than
    clamping the grasp point, because clamping silently changes grasp depth.
    """

    finite_values = (
        vision_target_z_m,
        command_target_z_m,
        box_height_m,
        configured_surface_z_m,
        surface_tolerance_m,
        tcp_below_target_m,
        surface_clearance_m,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("turntable safety parameters must be finite")
    if configured_surface_z_m < 0.0:
        raise ValueError(
            "turntable_surface_z_m is not configured; measure the User-frame "
            "turntable top surface before enabling automatic descent"
        )
    if surface_tolerance_m <= 0.0:
        raise ValueError("turntable_surface_tolerance_m must be positive")
    if tcp_below_target_m < 0.0 or surface_clearance_m < 0.0:
        raise ValueError("turntable tool offset and clearance must be non-negative")

    estimated_surface_z_m = estimate_support_surface_z(
        vision_target_z_m,
        box_height_m,
    )
    surface_error_m = estimated_surface_z_m - float(configured_surface_z_m)
    if abs(surface_error_m) > float(surface_tolerance_m):
        raise ValueError(
            "D405 target is inconsistent with the configured turntable surface: "
            f"estimated={estimated_surface_z_m:.4f}m, "
            f"configured={configured_surface_z_m:.4f}m, "
            f"error={surface_error_m * 1000.0:+.1f}mm, "
            f"limit={surface_tolerance_m * 1000.0:.1f}mm"
        )

    minimum_command_tcp_z_m = (
        float(configured_surface_z_m)
        + float(tcp_below_target_m)
        + float(surface_clearance_m)
    )
    if float(command_target_z_m) < minimum_command_tcp_z_m:
        raise ValueError(
            "turntable grasp TCP is below the configured tool/surface safety floor: "
            f"command={command_target_z_m:.4f}m, "
            f"minimum={minimum_command_tcp_z_m:.4f}m"
        )

    return TurntableHeightCheck(
        estimated_surface_z_m=estimated_surface_z_m,
        configured_surface_z_m=float(configured_surface_z_m),
        surface_error_m=surface_error_m,
        minimum_command_tcp_z_m=minimum_command_tcp_z_m,
        command_tcp_z_m=float(command_target_z_m),
    )
