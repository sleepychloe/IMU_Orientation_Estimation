import numpy as np

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_vec3, as_quat, as_vec3_batch, as_quat_batch
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
        norm: float = float(np.linalg.norm(v))
        return v / max(norm, EPS)

def predict_gravity_body_frame(q_pred: Quat, g_world_unit: Vec3) -> Vec3:
       g_pred: Vec3 = libq.rotate_world_to_body(q_pred, g_world_unit)
       return safe_unit(g_pred)

def small_angle_correction_quat(K_eff: float, e_axis: Vec3) -> Quat:
        dq_corr: Quat = as_quat(np.array([
                1,
                0.5 * K_eff * e_axis[0],
                0.5 * K_eff * e_axis[1],
                0.5 * K_eff * e_axis[2]]))
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
                dq_corr: Quat = small_angle_correction_quat(K, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q
        return res

def select_acc_measurement(a_src: Vec3, g_pred: Vec3) -> Vec3:
        a_meas_norm: float = float(np.linalg.norm(a_src))
        if a_meas_norm < EPS:
                return g_pred
        return a_src

def calc_acc_gating(g0: float, gate_sigma: float, a_meas: Vec3) -> float:
        # accel trust gating: if |a| deviates from g0, trust less
        dev: float = abs(float(np.linalg.norm(a_meas)) - g0)
        # w in [0,1]. 1 near static, 0 high linear acceleration
        weight_acc : float = np.exp(-0.5 * (dev / gate_sigma) ** 2)
        return weight_acc

def calc_gyro_gating(w_sigma: float, w: Vec3) -> float:
        w_norm: float = float(np.linalg.norm(w))
        weight_gyro: float = np.exp(-0.5 * (w_norm / w_sigma) ** 2)
        return weight_gyro

def integrate_gyro_acc(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                       K: float, g0: float, g_world_unit: Vec3,
                       acc_gate_sigma: float, gyro_gate_sigma: float, a_src: Vec3Batch
                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch]:
        """
        Returns:
                res (q_gyro_acc): (N,4) QuatBatch
                g_body_est: (N,3) Vec3Batch
                a_lin_est: (N,3) Vec3Batch
        """
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
                a_unit: Vec3 = safe_unit(a_meas)

                weight_acc: float = calc_acc_gating(g0, acc_gate_sigma, a_meas)
                weight_gyro: float = calc_gyro_gating(gyro_gate_sigma, w_avg[i])
                weight_acc *= weight_gyro

                e_axis: Vec3 = np.cross(g_pred, a_unit)
                dq_corr: Quat = small_angle_correction_quat(K * weight_acc, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q

                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
                a_lin_est[i] = a_src[i] + g_body_est[i]
        return res, g_body_est, a_lin_est

def calc_mag_gating(m0: float, gate_sigma: float, m_meas: Vec3,
                    q_pred: Vec3, g_pred: Vec3, m_world_h_unit: Vec3) -> tuple[float, Vec3]:
        m_norm: float = float(np.linalg.norm(m_meas))

        if m_norm < EPS:
                return 0, as_vec3(np.array([0, 0, 0]))

        m_unit: Vec3 = safe_unit(m_meas)

        # gate by magnitude deviation
        dev: float = abs(m_norm - m0)
        # w in [0,1]. 1 near static, 0 high linear acceleration
        weight_mag : float = np.exp(-0.5 * (dev / gate_sigma) ** 2)

        # tilt compensation: keep only horizontal component (remove along gravity)
        m_body_h: Vec3 = safe_unit(m_unit - np.dot(m_unit, g_pred) * g_pred)
        if np.linalg.norm(m_body_h) < 0.2:
                return 0, as_vec3(np.array([0, 0, 0]))

        # predicted horizontal mag in body
        m_pred: Vec3 = libq.rotate_world_to_body(q_pred, m_world_h_unit)
        m_pred_h: Vec3 = safe_unit(m_pred - np.dot(m_pred, g_pred) * g_pred)
        e_axis_mag: Vec3 = np.cross(m_body_h, m_pred_h)
        #e_axis_mag: Vec3 = np.cross(m_pred_h, m_body_h)
        return weight_mag, e_axis_mag

#def integrate_gyro_acc_mag(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
#                       K_acc: float, g0: float, g_world_unit: Vec3,
#                       acc_gate_sigma: float, gyro_gate_sigma: float,
#                       a_floor: float, m_floor: float, a_src: Vec3Batch,
#                       K_mag: float, m0: float, m_world_h_unit:Vec3,
#                       mag_gate_sigma: float, m_src: Vec3Batch
#                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch]:
#        """
#        Returns:
#                res (q_gyro_acc_mag): (N,4) QuatBatch
#                g_body_est: (N,3) Vec3Batch
#                a_lin_est: (N,3) Vec3Batch
#        """
#        q: Quat = q0.copy()
#        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
#        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
#        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))

#        wg_list = []
#        wa_list = []
#        wm_list = []
#        dev_list = []

#        for i in range(len(dt)):
#                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

#                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
#                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
#                a_unit: Vec3 = safe_unit(a_meas)

#                dev: float = abs(float(np.linalg.norm(a_meas)) - g0)
#                weight_acc = float(np.exp(-(dev / acc_gate_sigma)**4))
#                e_axis_acc: Vec3 = np.cross(g_pred, a_unit)

#                m_meas: Vec3 = m_src[i].copy()
#                weight_gyro: float = calc_gyro_gating(gyro_gate_sigma, w_avg[i])
#                weight_mag, e_axis_mag = calc_mag_gating(m0, mag_gate_sigma, m_meas,
#                                                         q_pred, g_pred, m_world_h_unit)
#                mag_err_sigma: float = 0.3
#                weight_mag *= float(np.exp(-0.5 * ((np.linalg.norm(e_axis_mag)) / mag_err_sigma) ** 2))
        
#                e_axis: Vec3 = K_acc * weight_acc * e_axis_acc + K_mag * weight_mag * e_axis_mag
#                dq_corr: Quat = small_angle_correction_quat(1, e_axis)

#                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
#                res[i] = q

#                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
#                a_lin_est[i] = a_src[i] + g_body_est[i]
                
                
                
#                wg_list.append(weight_gyro)
#                wa_list.append(weight_acc)
#                wm_list.append(weight_mag)
#                dev_list.append(dev)
        
        
#        wg = np.array(wg_list, dtype=np.float64)
#        wa = np.array(wa_list, dtype=np.float64)
#        wm = np.array(wm_list, dtype=np.float64)
#        d = np.array(dev_list, dtype=np.float64)
#        print("weight_gyro pcts:", np.percentile(wg, [0,25,50,75,90,95,99]))
#        print("weight_acc  pcts:", np.percentile(wa, [0,25,50,75,90,95,99]))
#        print("weight_mag  pcts:", np.percentile(wm, [0,25,50,75,90,95,99]))
#        print("dev  pcts:", np.percentile(d, [0,25,50,75,90,95,99]))
#        print("")
#        return res, g_body_est, a_lin_est

def integrate_gyro_acc_mag(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                       K_acc: float, g0: float, g_world_unit: Vec3, acc_gate_sigma: float, a_src: Vec3Batch,
                       K_mag: float, m0: float, m_world_h_unit:Vec3, mag_gate_sigma: float, m_src: Vec3Batch
                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch]:
        """
        Returns:
                res (q_gyro_acc_mag): (N,4) QuatBatch
                g_body_est: (N,3) Vec3Batch
                a_lin_est: (N,3) Vec3Batch
        """
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))

        wa_list = []
        wm_list = []
        dev_list = []

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
                a_unit: Vec3 = safe_unit(a_meas)

                dev: float = abs(float(np.linalg.norm(a_meas)) - g0)

                weight_acc: float = calc_acc_gating(g0, acc_gate_sigma, a_meas)
                e_axis_acc: Vec3 = np.cross(g_pred, a_unit)

                m_meas: Vec3 = m_src[i].copy()
                weight_mag, e_axis_mag = calc_mag_gating(m0, mag_gate_sigma, m_meas,
                                                         q_pred, g_pred, m_world_h_unit)

                e_axis: Vec3 = K_acc * weight_acc * e_axis_acc + K_mag * weight_mag * e_axis_mag
                dq_corr: Quat = small_angle_correction_quat(1, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q

                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
                a_lin_est[i] = a_src[i] + g_body_est[i]

                wa_list.append(weight_acc)
                wm_list.append(weight_mag)
                dev_list.append(dev)
        
        wa = np.array(wa_list, dtype=np.float64)
        wm = np.array(wm_list, dtype=np.float64)
        d = np.array(dev_list, dtype=np.float64)
        print("weight_acc  pcts:", np.percentile(wa, [0,25,50,75,90,95,99]))
        print("weight_mag  pcts:", np.percentile(wm, [0,25,50,75,90,95,99]))
        print("dev  pcts:", np.percentile(d, [0,25,50,75,90,95,99]))
        print("")
        return res, g_body_est, a_lin_est
