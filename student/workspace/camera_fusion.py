"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    sensor_position = transform[:3, :3] @ position + transform[:3, 3]
    if not np.isfinite(sensor_position).all():
        return False
    if sensor.name == "camera":
        minimum_depth = float(getattr(sensor, "depth_epsilon", 1e-6))
        if sensor_position[0] <= minimum_depth:
            return False
    horizontal_angle = float(np.arctan2(sensor_position[1], sensor_position[0]))
    lower, upper = map(float, sensor.fov)
    return lower <= horizontal_angle <= upper


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    sensor_position = transform[:3, :3] @ position + transform[:3, 3]
    minimum_depth = float(getattr(sensor, "depth_epsilon", 1e-6))
    if not np.isfinite(sensor_position).all() or sensor_position[0] <= minimum_depth:
        raise ValueError(
            "Camera projection needs finite coordinates and positive depth "
            f"> {minimum_depth:g}; sensor position={sensor_position.tolist()}"
        )
    depth, left, up = sensor_position
    pixel_i = sensor.c_i - sensor.f_i * left / depth
    pixel_j = sensor.c_j - sensor.f_j * up / depth
    return np.asmatrix([[pixel_i], [pixel_j]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    values = np.asarray(z, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        raise ValueError(f"Camera measurement needs two finite pixels; received {values.tolist()}")
    params = get_tracking_params()
    covariance = np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2])
    return {
        "z": np.asmatrix(values[:2]).T,
        "R": np.asmatrix(covariance),
        "sensor": sensor,
    }
