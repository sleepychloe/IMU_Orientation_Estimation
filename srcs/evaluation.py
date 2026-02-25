import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from my_types import Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_scalar_batch, as_vec3_batch
import lib_quat as libq

"""
Calculating relative rotation (error quaternion)
q_err = q_est⁻¹ ⊗ q_ref
(q_est ⊗ q_err = q_ref, q_err = q_est⁻¹ ⊗ q_ref)

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
        print(f"{label} angle error in rad — min/max/mean/p90")
        print(err.min(), err.max(), err.mean(), np.percentile(err, 90))
        print(f"\n{label} angle error in deg — min/max/mean/p90")
        print(np.rad2deg(err.min()), np.rad2deg(err.max()), np.rad2deg(err.mean()), np.rad2deg(np.percentile(err, 90)))

def save_err_csv(path: Path, t: ScalarBatch, err: ScalarBatch) -> None:
        df: pd.DataFrame = pd.DataFrame({
                "seconds_elapsed": t.astype(np.float64),
                "angle_err" : err.astype(np.float64)
        })
        df.to_csv(path, index=False)

def load_err_csv(path: Path) -> tuple[ScalarBatch, ScalarBatch]:
        df: pd.DataFrame = pd.read_csv(path)
        t: ScalarBatch = as_scalar_batch(df["seconds_elapsed"].to_numpy())
        err: ScalarBatch = as_scalar_batch(df["angle_err"].to_numpy())
        return t, err

def plot_err_from_csv(series: list[tuple[str, Path]], save_path: Path | None = None) -> None:
        plt.figure(figsize=(12,4))
        plt.title("Orientation error (rad)")
        plt.xlabel("seconds_elapsed (s)")
        plt.ylabel("angle error (rad)")

        for label, path in series:
                t, err = load_err_csv(path)
                plt.plot(t, err, label=label)
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

def plot_err_colored_by_weights(t: ScalarBatch, err: ScalarBatch,
                                weight: ScalarBatch, title: str) -> None:
        plt.figure(figsize=(12,4))
        sc = plt.scatter(t, err, c=weight, s=6)
        plt.title("Error colored by " + title)
        plt.xlabel("seconds_elapsed (s)")
        plt.ylabel("angle error (rad)")

        plt.plot(t, err, alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.colorbar(sc, label="weight")
        plt.tight_layout()
        plt.show()

def evaluate_estimated_vec3_autosign(est: Vec3Batch, ref: Vec3Batch) -> tuple[Vec3Batch, float]:
        d: Vec3Batch = None
        est_sign_fixed: Vec3Batch = est.copy()

        d_minus: Vec3Batch = as_vec3_batch(est - ref)
        d_plus: Vec3Batch = as_vec3_batch(est + ref)

        rmse_minus: float = np.sqrt(np.mean(np.sum(d_minus*d_minus, axis=1)))
        rmse_plus: float = np.sqrt(np.mean(np.sum(d_plus*d_plus, axis=1)))

        if (rmse_plus < rmse_minus):
                d = d_plus
                est_sign_fixed *= -1
        else:
                d = d_minus

        rmse_norm: float = np.sqrt(np.mean(np.sum(d*d, axis=1)))
        print("RMSE norm:", rmse_norm)
        return est_sign_fixed, rmse_norm

def save_estimated_vec3_csv(path: Path, t: ScalarBatch, vec3_batch: Vec3Batch) -> None:
        df: pd.DataFrame = pd.DataFrame({
                "seconds_elapsed": t.astype(np.float64),
                "x": vec3_batch[:,0].astype(np.float64),
                "y": vec3_batch[:,1].astype(np.float64),
                "z": vec3_batch[:,2].astype(np.float64)
	})
        df.to_csv(path, index=False)
