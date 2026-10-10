"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: import numpy as np


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    transform = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    sensor_position = np.asarray(meas.z, dtype=float).reshape(-1)[:3]
    vehicle_position = rotation @ sensor_position + translation

    state = np.zeros((params.dim_state, 1), dtype=float)
    state[:3, 0] = vehicle_position
    covariance = np.zeros((params.dim_state, params.dim_state), dtype=float)
    covariance[:3, :3] = rotation @ np.asarray(meas.R, dtype=float) @ rotation.T
    covariance[3:, 3:] = np.diag(
        [params.sigma_p44, params.sigma_p55, params.sigma_p66]
    )
    return {
        "x": np.asmatrix(state),
        "P": np.asmatrix(covariance),
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    increment = 1.0 / params.window
    if associated:
        track["score"] = min(1.0, float(track["score"]) + increment)
        if track["score"] > params.confirmed_threshold:
            track["state"] = "confirmed"
        elif track["state"] != "confirmed":
            track["state"] = "tentative"
    else:
        track["score"] = float(track["score"]) - increment
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    covariance = np.asarray(track["P"], dtype=float)
    excessive_uncertainty = (
        covariance[0, 0] > params.max_P or covariance[1, 1] > params.max_P
    )
    if track["state"] == "confirmed":
        exhausted_score = float(track["score"]) < params.delete_threshold
    else:
        exhausted_score = float(track["score"]) <= 0.0
    return bool(excessive_uncertainty or exhausted_score)
