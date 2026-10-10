"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

kalman = load_workspace_module("kalman")

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: from fusion_lab.workspace_loader import load_workspace_module
# vi: kalman = load_workspace_module("kalman")  # không dùng `import kalman`


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    jacobian = np.asmatrix(meas.sensor.get_H(track.x))
    residual = np.asmatrix(kalman.innovation(track.x, meas))
    covariance = np.asmatrix(
        kalman.innovation_covariance(track.P, meas, jacobian)
    )
    solved = np.linalg.solve(np.asarray(covariance), np.asarray(residual))
    return float((np.asarray(residual.T) @ solved).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(np.isfinite(mhd_sq) and mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    costs = np.full((len(track_list), len(meas_list)), np.inf, dtype=float)
    for track_index, track in enumerate(track_list):
        for measurement_index, measurement in enumerate(meas_list):
            if not measurement.sensor.in_fov(track.x):
                continue
            distance = mahalanobis_distance(track, measurement)
            if chi2_gate(distance, measurement.sensor):
                costs[track_index, measurement_index] = distance
    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    matrix = np.asmatrix(association_matrix)
    remaining_tracks = list(unassigned_tracks)
    remaining_meas = list(unassigned_meas)
    if matrix.size == 0 or not np.isfinite(np.asarray(matrix)).any():
        return np.nan, np.nan, matrix, remaining_tracks, remaining_meas

    flat_index = int(np.nanargmin(np.asarray(matrix)))
    track_index, measurement_index = np.unravel_index(flat_index, matrix.shape)
    track = remaining_tracks.pop(track_index)
    measurement = remaining_meas.pop(measurement_index)
    reduced = np.delete(np.delete(np.asarray(matrix), track_index, axis=0),
                        measurement_index, axis=1)
    return track, measurement, np.asmatrix(reduced), remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    association_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

    while True:
        track, measurement, association_matrix, unassigned_tracks, unassigned_meas = (
            pick_next_pair(association_matrix, unassigned_tracks, unassigned_meas)
        )
        if isinstance(track, (float, np.floating)) and np.isnan(track):
            break
        filter_obj.update(track, measurement)
        manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
