import numpy as np

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_quat, as_quat_batch
import lib_quat as libq

EPS: float = 1e-9

def gyro_predict(q: Quat, w_avg: Vec3, dt: float) -> Quat:
        dq: Quat = libq.delta_quat_from_omega(w_avg, dt)
        q_pred: Quat = libq.quat_norm(libq.quat_mul(q, dq))
        return q_pred

def integrate_gyro(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch) -> QuatBatch:
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        
        for i in range(len(dt)):
                q: Quat = gyro_predict(q, w_avg[i], dt[i])
                res[i] = q
        return res

def safe_unit(v: Vec3) -> Vec3:
        norm: float = np.linalg.norm(v)
        return v / max(norm, EPS)

def predict_gravity_body_frame(q_pred: Quat, g_world: Vec3) -> Vec3:
       g_pred: Vec3 = libq.rotate_world_to_body(q_pred, g_world)
       return safe_unit(g_pred)

def generate_angle_correction_quat(e_axis: Vec3, K: float) -> Quat:
        dq_corr: Quat = as_quat(np.array([
                1,
                0.5 * K * e_axis[0],
                0.5 * K * e_axis[1],
                0.5 * K * e_axis[2]]))
        return libq.quat_norm(dq_corr)

def integrate_gyro_grav(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                        K: float, g_world_unit: Vec3, g_meas_body: Vec3Batch,) -> QuatBatch:
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                g_meas: Vec3 = safe_unit(g_meas_body[i].copy())

                e_axis: Vec3 = np.cross(g_pred, g_meas)

                dq_corr: Quat = generate_angle_correction_quat(e_axis, K)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q
        return res