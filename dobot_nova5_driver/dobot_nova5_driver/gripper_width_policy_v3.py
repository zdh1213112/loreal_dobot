"""Gripper pre-shape policy for cosmetic-box picks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperWidthDecision:
    """Selected opening and the geometry that caused the decision."""

    command_m: float
    aspect_ratio: float
    full_opening: bool
    reason: str


def select_gripper_width(
    box_length_m: float,
    box_width_m: float,
    *,
    clearance_m: float,
    max_opening_m: float,
    full_open_width_threshold_m: float,
    near_square_aspect_ratio: float,
) -> GripperWidthDecision:
    """Choose measured-width-plus-clearance or the fully open position.

    Widths strictly above ``full_open_width_threshold_m`` use the maximum
    opening.  Near-square top surfaces also use the maximum because their two
    planar axes can exchange labels between otherwise valid measurements.
    """

    values = (
        box_length_m,
        box_width_m,
        clearance_m,
        max_opening_m,
        full_open_width_threshold_m,
        near_square_aspect_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gripper width policy values must be finite")
    if box_length_m <= 0.0 or box_width_m <= 0.0:
        raise ValueError("box length and width must be positive")
    if clearance_m < 0.0 or max_opening_m <= 0.0:
        raise ValueError("clearance must be non-negative and maximum opening positive")
    if full_open_width_threshold_m <= 0.0 or near_square_aspect_ratio <= 1.0:
        raise ValueError("width threshold must be positive and aspect threshold greater than one")

    aspect_ratio = max(box_length_m, box_width_m) / min(box_length_m, box_width_m)
    reasons = []
    if box_width_m > full_open_width_threshold_m:
        reasons.append("wide")
    if aspect_ratio < near_square_aspect_ratio:
        reasons.append("near_square")

    if reasons:
        return GripperWidthDecision(
            command_m=float(max_opening_m),
            aspect_ratio=float(aspect_ratio),
            full_opening=True,
            reason="+".join(reasons),
        )

    return GripperWidthDecision(
        command_m=float(min(max_opening_m, box_width_m + clearance_m)),
        aspect_ratio=float(aspect_ratio),
        full_opening=False,
        reason="measured_plus_clearance",
    )
