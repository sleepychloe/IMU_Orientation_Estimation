import numpy as np
from typing import Any

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, QuatBatch, BoolBatch, Int8Batch, Int32Batch, Int64Batch
from my_types import as_vec3, as_scalar_batch, as_bool_batch, as_int8_batch
from pipelines import integrate_gyro_acc
from evaluation import calc_angle_err

EPS: float = 1e-6
DELTA: float = 1e-12

def auto_setup_imu_frame(q_ref: QuatBatch, w: Vec3Batch, dt: ScalarBatch,
                         g0: float, a_src: Vec3Batch) -> tuple[Vec3Batch, Vec3]:
        """
        Returns:
                a_src_interp: (N,3): converted unit [m/s²]
                g_world_unit: Vec3
        """
        if np.median(np.linalg.norm(a_src, axis=1)) < 2:
                print("Detected accel unit in [g] → converting to [m/s²]")
                a_src = a_src * g0
        else:
                print("Detected accel unit in [m/s²]")

        g_world_dir_candidate: tuple[Vec3, Vec3] = [
                as_vec3(np.array([0, 0, -1])),
                as_vec3(np.array([0, 0, 1]))
        ]
        score_tmp: float = np.inf
        g_world_unit: Vec3 = None
        for g_dir in g_world_dir_candidate:
                q_tmp, _, _, _, _ = integrate_gyro_acc(
                        q_ref[0].copy(), w[:2000], dt[:2000],
                        (np.median(dt[:2000]) / 0.5), g0, g_dir,
                        np.inf, np.inf, a_src[:2000])
                err_tmp = np.mean(calc_angle_err(q_tmp[:2000], q_ref[:2000]))
                if score_tmp > err_tmp:
                        score_tmp = err_tmp
                        g_world_unit = g_dir
        print("Selected g_world_unit:", g_world_unit)
        return a_src, g_world_unit

def stats_basic(x: float) -> tuple[float, ...]:
        """
        Returns:
                min, max, mean, p50, p90, p99
        """
        return (np.min(x), np.max(x), np.mean(x),
                np.percentile(x, 50), np.percentile(x, 90), np.percentile(x, 99))

def largest_true_run(mask: BoolBatch) -> tuple[int, int, int] | None:
        """
        Returns:
                start, end, length
        """
        if mask.size == 0 or not mask.any():
                return None

        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))

        # diff == 1: 0(False) → 1(True): start point
        # diff == -1: 1(True) → 0(False): end point
        diff: Int8Batch = as_int8_batch(np.diff(m).astype(np.int8, copy=False))
        
        start: Int64Batch = np.where(diff == 1)[0] + 1
        end: Int64Batch = np.where(diff == -1)[0] + 1

        if bool(mask[0]):
                start = np.r_[np.array([0], dtype=start.dtype), start]
        if bool(mask[-1]):
                end = np.r_[end, np.array([mask.size], dtype=end.dtype)]

        length: Int64Batch = end - start

        i: int = int(np.argmax(length))
        return int(start[i]), int(end[i]), int(length[i])

def smooth_bool(mask: BoolBatch, win: int = 5) -> BoolBatch:
        """
        Simple debouncing: True if majority True in a window.
        """
        if win <= 1:
                return mask

        # BoolBatch → Int8Batch for 0/1 majority vote
        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))
        kernel: Int32Batch = np.ones(win, dtype=np.int32)

        s: Int64Batch = np.convolve(m, kernel, mode="same")
        res: BoolBatch = (s >= (win//2 + 1))
        return as_bool_batch(res)

def quasi_static_detector(w: Vec3Batch, a: Vec3Batch, dt: ScalarBatch, g0: float,
                          w_thr: float, a_thr: float,
                          min_duration_s: float, smooth_win: int,
                          ) -> tuple[BoolBatch, tuple[int, int, int]]:
        """
        Returns:
                quasi_static_mask: (N,) BoolBatch
                quasi_static_info: dict[str, Any], threshold and stats
                best_run: tuple[int, int, int], longest quasi_static segment
        """
        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w, axis=1))
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a, axis=1))
        acc_resid: ScalarBatch = np.abs(a_norm - g0)

        if w_thr is None:
                w_thr = float(np.percentile(w_norm, 5))
        if a_thr is None:
                a_thr = float(np.percentile(acc_resid, 5))

        raw_mask: BoolBatch = (w_norm < w_thr) & (acc_resid < a_thr)
        quasi_static_mask: BoolBatch = smooth_bool(raw_mask, win=smooth_win)

        # enforce min duration
        # convert duration to samples roughly using median dt
        dt_midean: float = np.median(dt)
        min_len: int = int(np.ceil(min_duration_s / max(dt_midean, EPS)))

        best_quasi_static: tuple[int, int, int] = largest_true_run(quasi_static_mask)
        if best_quasi_static is not None and best_quasi_static[2] < min_len:
                best_quasi_static = None
                print("Best quasi static not found")
        print("Best quasi static(start, end, length): ", best_quasi_static)
        return quasi_static_mask, best_quasi_static

def suggest_gate_sigma(w: Vec3Batch, a: Vec3Batch, g0: float,
                       p_gyro: int, p_acc: int, sigma_floor: float,
                       best_quasi_static: tuple[float, float, float] = None
                       ) -> tuple[float, float]:
        """
        Returns:
                gyro_sigma: float
                acc_sigma: float
        """
        if best_quasi_static is not None:
                s, e, _ = best_quasi_static
                w_use = w[s:e]
                a_use = a[s:e]
        else:
                w_use = w
                a_use = a

        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w_use, axis=1))
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a_use, axis=1))
        acc_resid: ScalarBatch = np.abs(a_norm - g0)

        if p_gyro is None:
                gyro_sigma = np.inf
        else:
                gyro_sigma: float = max(sigma_floor, float(np.percentile(w_norm, p_gyro)))
        
        if p_acc is None:
                acc_sigma = np.inf
        else:
                acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))
        print("Suggested gyro_sigma: ", gyro_sigma)
        print("Suggested acc_sigma: ", acc_sigma)
        return gyro_sigma, acc_sigma

def choose_tau_from_quasi_static(q0: Quat, w: Vec3Batch, dt: ScalarBatch, a: Vec3Batch,
                                 g0: float, g_world_unit: Vec3,
                                 acc_gate_sigma: float, gyro_gate_sigma: float,
                                 best_quasi_static: tuple[int, int, int] | None = None,
                                 tau_candidates: tuple[float, ...] = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1)
                                 ) -> tuple[list[dict[str, Any]], float, float]:
        """
        Returns:
                tau_table: dict[str, Any]
                best_tau: float
                K: float
        """
        dt_midean: float = float(np.median(dt))

        if best_quasi_static is None:
                return 0.25,[{"tau": 0.25, "K": float(dt_midean / 0.25)}]

        s, e, _ = best_quasi_static
        tau_table: list[dict[str, Any]] = []
        for tau in tau_candidates:
                K = float(dt_midean / tau)

                _, g_body_est, _, _, _ = integrate_gyro_acc(
                                                        q0.copy(), w, dt,
                                                        K, g0, g_world_unit,
                                                        acc_gate_sigma, gyro_gate_sigma, a)

                # In quasi_static segment, gravity in body frame should be stable.
                # Use direction variance as a stability proxy.
                gb = g_body_est[s:e]
                gb_unit = gb / (np.linalg.norm(gb, axis=1, keepdims=True) + DELTA)

                # score: average angular deviation from mean direction
                mean_dir: Vec3 = as_vec3(np.mean(gb_unit, axis=0))
                mean_dir = mean_dir / (np.linalg.norm(mean_dir) + DELTA)
                dot: ScalarBatch = np.clip(gb_unit @ mean_dir, -1, 1)
                ang: ScalarBatch = np.arccos(dot)
                score: float = float(np.mean(ang)) # lower is better

                tau_table.append({
                        "tau": float(tau), "K": K,
                        "quasi_static_score_mean_angle(rad)": score})
        tau_table.sort(key=lambda d: d["quasi_static_score_mean_angle(rad)"])
        best_tau: float = tau_table[0]["tau"]
        return tau_table, best_tau, K
