import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from my_types import Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_scalar_batch, as_vec3_batch
import lib_quat as libq
from pipelines import safe_unit

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
        print(np.rad2deg(err.min()), np.rad2deg(err.max()),
              np.rad2deg(err.mean()), np.rad2deg(np.percentile(err, 90)))

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

def plot_err_from_csv(series: list[tuple[str, Path]], save_path: Path = None) -> None:
        plt.figure(figsize=(12,4))
        plt.title("Orientation error (rad)")
        plt.xlabel("seconds_elapsed (s)")
        plt.ylabel("angle error (rad)")

        for label, path in series:
                t, err = load_err_csv(path)
                plt.plot(t, err, label=label)
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

def plot_quasi_static_diagnostic(t: ScalarBatch, w: Vec3Batch, a: Vec3Batch, g0: float,
                                 best_quasi_static: tuple[float, float ,float] = None,
                                 save_path: Path = None) -> None:
        w_norm: ScalarBatch = np.linalg.norm(w, axis=1)
        a_norm: ScalarBatch = np.linalg.norm(a, axis=1)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 4), height_ratios=[1, 1])

        axs[0].set_ylabel("||w|| (rad/s)")
        axs[0].plot(t, w_norm)
        axs[0].grid(True, alpha=0.3)

        axs[1].set_xlabel("seconds_elapsed (s)")
        axs[1].plot(t, a_norm, label="||a||")
        axs[1].axhline(g0, linestyle="--", label="g0")
        axs[1].set_ylabel("||a|| (m/s²)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        if best_quasi_static is not None:
                s, e, l = best_quasi_static
                t0, t1 = t[s], t[e - 1]
                for ax in axs:
                        ax.axvspan(t0, t1, alpha=0.3)
                axs[0].set_title(f"Quasi staticness diagnostics — quasi static run(longest): [{t0:.2f}s, {t1:.2f}s], {l} samples")
        else:
                axs[0].set_title("Quasi staticness diagnostics — no quasi static run found")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

def plot_err_colored_by_weight(t: ScalarBatch, err: ScalarBatch,
                               w_acc: ScalarBatch, w_gyro: ScalarBatch, w_total: ScalarBatch,
                               save_path: Path = None) -> None:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 6), height_ratios=[1, 1, 1])

        axs[0].set_title("[exp 2-3] Error colored by weight")
        sc_t = axs[0].scatter(t, err, c=w_total, s=6)
        fig.colorbar(sc_t, ax=axs[0]).set_label("weight_total")
        axs[0].plot(t, err, alpha=0.5)
        axs[0].set_ylabel("angle error (rad)")
        axs[0].grid(True, alpha=0.3)

        sc_a = axs[1].scatter(t, err, c=w_acc, s=6)
        fig.colorbar(sc_a, ax=axs[1]).set_label("weight_acc")
        axs[1].plot(t, err, alpha=0.5)
        axs[1].set_ylabel("angle error (rad)")
        axs[1].grid(True, alpha=0.3)

        axs[2].set_xlabel("seconds_elapsed (s)")
        sc_g = axs[2].scatter(t, err, c=w_gyro, s=6)
        fig.colorbar(sc_g, ax=axs[2]).set_label("weight_gyro")
        axs[2].plot(t, err, alpha=0.5)
        axs[2].set_ylabel("angle error (rad)")
        axs[2].grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

def evaluate_estimated_vec3_autosign(est: Vec3Batch, ref: Vec3Batch
                                     ) -> tuple[Vec3Batch, float]:
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

def calc_vec3_direction_angle_err(est: Vec3Batch, ref: Vec3Batch) -> ScalarBatch:
        norm_est: Vec3Batch = safe_unit(est)
        norm_ref: Vec3Batch = safe_unit(ref)

        dot: ScalarBatch = np.sum(norm_est * norm_ref, axis=1)
        dot = np.clip(dot, -1, 1)
        err: ScalarBatch = np.arccos(dot)
        return err
