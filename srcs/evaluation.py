import numpy as np
import pandas as pd
from pathlib import Path

from my_types import Quat, ScalarBatch, QuatBatch
from my_types import as_scalar_batch
import lib_quat as libq

"""
Calculating relative rotation (error quaternion)
q_err = q_custom⁻¹ ⊗ q_ref
(q_custom ⊗ q_err = q_ref, q_err = q_custom⁻¹ ⊗ q_ref)

q⁻¹ = q* / ||q||²,
q⁻¹ = q* when q is unit quaternion
"""
def calc_angle_err(q_est: QuatBatch, q_ref: QuatBatch) -> ScalarBatch:
        w_err: ScalarBatch = as_scalar_batch(np.empty(len(q_est)))
        for i in range(len(q_est)):
                q_err: Quat = libq.quat_mul(libq.quat_conj(q_est[i]), q_ref[i])
                w_err[i] = np.clip(np.abs(q_err[0]), 0.0, 1.0)
        return as_scalar_batch(2 * np.arccos(w_err))

def print_err_status(label: str, err: ScalarBatch) -> None:
        print(f"{label} angle error in rad — min/max/mean")
        print(err.min(), err.max(), err.mean())
        print(f"\n{label} angle error in deg — min/max/mean")
        print(np.rad2deg(err.min()), np.rad2deg(err.max()), np.rad2deg(err.mean()))

def save_err_csv(path: Path, t: ScalarBatch, err: ScalarBatch) -> None:
        df: pd.DataFrame = pd.DataFrame({
                "seconds_elapsed": t.astype(np.float64),
                "angle_err" : err.astype(np.float64)
        })
        df.to_csv(path, index=False)