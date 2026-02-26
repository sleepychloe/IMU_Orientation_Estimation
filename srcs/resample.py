import numpy as np
from typing import overload, Any, Sequence

from my_types import ScalarBatch, Vec3Batch, QuatBatch
from pipelines import integrate_gyro
from evaluation import calc_angle_err

@overload
def resample_batch(t_new: ScalarBatch, t_src: ScalarBatch, src: Vec3Batch) -> Vec3Batch: ...

@overload
def resample_batch(t_new: ScalarBatch, t_src: ScalarBatch, src: QuatBatch) -> QuatBatch: ...

def resample_batch(t_new, t_src, src):
        dim: int = src.shape[1]
        res = np.empty((len(t_new), dim), dtype=np.float64)
        for i in range(dim):
                res[:, i] = np.interp(t_new, t_src, src[:, i])
        return res

def find_stable_start_idx(dt: ScalarBatch, w: Vec3Batch, q_ref: QuatBatch,
                          sample_window: int, threshold: float, sample_hz: int,
                          consecutive: int, min_cut_second: int, max_cut_second: int
                          ) -> int:
        n: int = len(dt)
        if n <= sample_window:
                return 0

        max_idx: int = min(sample_hz * max_cut_second, n - sample_window)
        best_idx: int = None
        cons: int = 0
        i: int = 0

        while i <= max_idx:
                dt_tmp: ScalarBatch = dt[i : i + sample_window]
                w_tmp: Vec3Batch = w[i : i + sample_window]
                q_ref_tmp: QuatBatch = q_ref[i : i + sample_window]

                q0_tmp = q_ref[i].copy()
                q_gyro_tmp: QuatBatch = integrate_gyro(q0_tmp, w_tmp, dt_tmp)
                angle_err_tmp: ScalarBatch = calc_angle_err(q_gyro_tmp, q_ref_tmp)
                p90: float = float(np.percentile(np.asarray(angle_err_tmp).reshape(-1), 90))

                if p90 < threshold:
                        cons += 1
                        if cons >= consecutive:
                                best_idx = i - (consecutive - 1) * sample_hz
                                best_idx = max(0, best_idx)
                                break
                else:  
                        cons = 0
                i += sample_hz

        max_cut_idx: int = min(sample_hz * max_cut_second, n - sample_window)
        min_cut_idx: int = min(sample_hz * min_cut_second, n - sample_window)
        if best_idx is None:
                print(f"[WARN] Stabilization not found within {max_cut_second}s, applying fallback cut={max_cut_second}s")
                return max_cut_idx
        elif (best_idx / sample_hz) < min_cut_second:
                print(f"[INFO] Stabilization detected too early (< min_cut), applying min_cut={min_cut_second}s policy")
                return min_cut_idx
        print(f"[OK] Stabilization detected, cut idx {best_idx} (≈ {best_idx / sample_hz:.1f}s)")
        return best_idx

def cut_sample(win: int, sample: Sequence[Any] = None) -> list[Any]:
        return [x[win:] for x in sample]
