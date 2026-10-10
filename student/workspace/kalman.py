"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: Gợi ý module — params = get_tracking_params() sau khi import ở trên.


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    params = get_tracking_params()
    time_step = params.dt if dt is None else float(dt)
    transition = np.asmatrix(np.eye(params.dim_state, dtype=float))
    transition[0, 3] = time_step
    transition[1, 4] = time_step
    transition[2, 5] = time_step
    return transition


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    params = get_tracking_params()
    time_step = params.dt if dt is None else float(dt)
    noise_scale = params.q if q is None else float(q)
    return np.asmatrix(np.eye(params.dim_state, dtype=float) * time_step * noise_scale)


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    transition = build_F() if F is None else np.asmatrix(F)
    process_noise = build_Q() if Q is None else np.asmatrix(Q)
    state = np.asmatrix(x)
    covariance = np.asmatrix(P)
    predicted_state = transition @ state
    predicted_covariance = transition @ covariance @ transition.T + process_noise
    return np.asmatrix(predicted_state), np.asmatrix(predicted_covariance)


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    return np.asmatrix(meas.z) - np.asmatrix(meas.sensor.get_hx(x))


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    jacobian = np.asmatrix(H)
    covariance = np.asmatrix(P)
    return jacobian @ covariance @ jacobian.T + np.asmatrix(meas.R)


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    params = get_tracking_params()
    state = np.asmatrix(x)
    covariance = np.asmatrix(P)
    jacobian = np.asmatrix(meas.sensor.get_H(state))
    residual = innovation(state, meas)
    residual_covariance = innovation_covariance(covariance, meas, jacobian)

    # Solve K S = P H.T instead of explicitly forming S^-1.
    gain = np.asmatrix(
        np.linalg.solve(
            np.asarray(residual_covariance).T,
            np.asarray(covariance @ jacobian.T).T,
        ).T
    )
    updated_state = state + gain @ residual
    identity = np.asmatrix(np.eye(params.dim_state, dtype=float))
    updated_covariance = (identity - gain @ jacobian) @ covariance
    return np.asmatrix(updated_state), np.asmatrix(updated_covariance)
