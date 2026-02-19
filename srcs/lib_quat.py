import numpy as np

from my_types import Vec3, Quat
from my_types import as_vec3, as_quat

EPS: float = 1e-12

def quat_norm(q: Quat) -> Quat:
        n: float = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        if n > EPS:
                return q / n
        return as_quat(np.array([1, 0, 0, 0]))

def quat_mul(q1: Quat, q2: Quat) -> Quat:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return as_quat(np.array(
                [
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ]))

def quat_conj(q: Quat) -> Quat:
        return as_quat(np.array([q[0], -q[1], -q[2], -q[3]]))

def delta_quat_from_omega(omega: Vec3, dt: float) -> Quat:
        """
        Build incremental quaternion from angular velocity 𝜔[rad/s] and dt[s].
        Δq = [cos(θ/2), u⋅sin(θ/2)],
        θ = ||𝜔||⋅dt
        """
        mag: float = float(np.linalg.norm(omega))

        if mag < EPS or dt <= 0:
                return as_quat(np.array([1, 0, 0, 0]))

        theta: float = mag * dt
        u: Vec3 = omega / mag
        half: float = theta / 2
        return as_quat(np.array(
                [np.cos(half), u[0]*np.sin(half), u[1]*np.sin(half), u[2]*np.sin(half)]))

def rotate_world_to_body(q: Quat, v_world: Vec3) -> Vec3:
        """
        v_world = q ⊗ v ⊗ q⁻¹
        v: embed as pure quaternion (0, v)
        """
        vq: Quat = as_quat(np.array([0, v_world[0], v_world[1], v_world[2]]))
        res: Quat = quat_mul(quat_mul(quat_conj(q), vq), q)
        return as_vec3(np.array([res[1], res[2], res[3]]))

def rotate_body_to_world(q: Quat, v_body: Vec3) -> Vec3:
        vq: Quat = as_quat(np.array([0, v_body[0], v_body[1], v_body[2]]))
        res: Quat = quat_mul(quat_mul(q, vq), quat_conj(q))
        return as_vec3(np.array([res[1], res[2], res[3]]))

