import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

Vec3: TypeAlias = NDArray[np.float64]
Quat: TypeAlias = NDArray[np.float64]

EPS: float = 1e-12

def quat_norm(q: Quat) -> Quat:
        n: float = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        if n > EPS:
                return q / n
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

def quat_mul(q1: Quat, Q2: Quat) -> Quat:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = Q2
        return np.array(
                [
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ],
                dtype=np.float64)

def quat_conj(q: Quat) -> Quat:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def delta_quat_from_omega(omega: Vec3, dt: float) -> Quat:
        """
        Build incremental quaternion from angular velocity 𝜔[rad/s] and dt[s].
        Δq = [cos(θ/2), u⋅sin(θ/2)],
        θ = ||𝜔||⋅dt
        """
        mag: float = float(np.linalg.norm(omega))

        if mag < EPS or dt <= 0.0:
                return (np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

        theta: float = mag * dt
        u: Vec3 = omega / mag
        half: float = theta * 0.5
        return np.array(
                [np.cos(half), u[0]*np.sin(half), u[1]*np.sin(half), u[2]*np.sin(half)],
                dtype=np.float64)

def rotate_world_to_body(q: Quat, v_world: Vec3) -> Vec3:
        """
        v_world = q ⊗ v ⊗ q⁻¹
        v: embed as pure quaternion (0, v)
        """
        vq: Quat = np.array([0, v_world[0], v_world[1], v_world[2]], dtype=np.float64)
        res: Quat = quat_mul(quat_mul(quat_conj(q), vq), q)
        return np.array([res[1], res[2], res[3]], dtype=np.float64)

def rotate_body_to_world(q: Quat, v_body: Vec3) -> Vec3:
        vq: Quat = np.array([0, v_body[0], v_body[1], v_body[2]], dtype=np.float64)
        res: Quat = quat_mul(quat_mul(q, vq), quat_conj(q))
        return np.array([res[1], res[2], res[3]], dtype=np.float64)

"""
Calculating relative rotation (error quaternion)
q_err = q_custom⁻¹ ⊗ q_ref
(q_custom ⊗ q_err = q_ref, q_err = q_custom⁻¹ ⊗ q_ref)

q⁻¹ = q* / ||q||²,
q⁻¹ = q* when q is unit quaternion
"""
def calc_angle_err(q_est: NDArray[np.float64], q_ref: NDArray[np.float64]) -> NDArray[np.float64]:
        w_err: NDArray[np.float64] = np.empty(len(q_est), dtype=np.float64)
        for i in range(len(q_est)):
                q_err: Quat = quat_mul(quat_conj(q_est[i]), q_ref[i])
                w_err[i] = np.clip(np.abs(q_err[0]), 0.0, 1.0)
        return 2 * np.arccos(w_err)
