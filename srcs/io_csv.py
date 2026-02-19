from pathlib import Path
import pandas as pd
import numpy as np

from my_types import ScalarBatch, Vec3, Vec3Batch, QuatBatch
from my_types import as_scalar_batch, as_vec3_batch, as_quat_batch
from resample import resample_batch

def load_sorted_frame(csv_path: Path, usecols: list[str], sort_by: list[str]) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(csv_path, usecols=usecols)
        df[sort_by] = df[sort_by].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=sort_by).sort_values("seconds_elapsed").reset_index(drop=True)
        df = df[sort_by]
        return df

def load_gyro_base(csv_path: Path
                   ) -> tuple[ScalarBatch, Vec3Batch, ScalarBatch, ScalarBatch, Vec3Batch]:
        """
        Returns:
                t_src: (N,) original gyro time (ScalarBatch)
                w_src: (N,3) gyro angular velocity (Vec3Batch)
                dt: (N-1,) dt (ScalarBatch)
                t_new: (N-1,) custom time — standard time (ScalarBatch)
                w_avg: (N-1,3) section average angular velocity (Vec3Batch)
        """
        df: pd.DataFrame = load_sorted_frame(csv_path,
                                             ["seconds_elapsed", "z", "y", "x"],
                                             ["seconds_elapsed", "x", "y", "z"])
        t_src: ScalarBatch = as_scalar_batch(df["seconds_elapsed"].to_numpy())
        w_src: Vec3Batch = as_vec3_batch(df[["x", "y", "z"]].to_numpy())

        dt: ScalarBatch = np.diff(t_src)
        t_new: ScalarBatch = t_src[1:]
        w_avg: Vec3Batch = 0.5 * (w_src[1:] + w_src[:-1]) # section average angular velocity
        return t_src, w_src, dt, t_new, w_avg

def load_ref(csv_path: Path, t_new: ScalarBatch) -> QuatBatch:
        """
        Returns:
                q_src_interp: (N-1,) interpolated ref quaternion (QuatBatch)
        """
        df: pd.DataFrame = load_sorted_frame(csv_path,
                                             ["seconds_elapsed", "qz", "qy", "qx", "qw"],
                                             ["seconds_elapsed", "qw", "qx", "qy", "qz"])

        t_src: ScalarBatch = as_scalar_batch(df["seconds_elapsed"].to_numpy())
        q_src: QuatBatch = as_quat_batch(df[["qw", "qx", "qy", "qz"]].to_numpy())

        # fix quaternion sign continuity
        for i in range(1, len(q_src)):
                if np.dot(q_src[i - 1], q_src[i]) < 0:
                        q_src[i] *= -1

        q_src_interp: QuatBatch = as_quat_batch(resample_batch(t_new, t_src, q_src))
        return q_src_interp

def load_grav_ref(csv_path: Path, t_new: ScalarBatch) -> Vec3Batch:
        """
        Returns:
                g_src_interp: (N-1,) interpolated ref gravity (Vec3Batch)
	"""
        df: pd.DataFrame = load_sorted_frame(csv_path,
                                             ["seconds_elapsed", "z", "y", "x"],
                                             ["seconds_elapsed", "x", "y", "z"])
        t_src: ScalarBatch = as_scalar_batch(df["seconds_elapsed"].to_numpy())
        g_src: Vec3Batch = as_vec3_batch(df[["x", "y", "z"]].to_numpy())

        g_src_interp: Vec3 = as_vec3_batch(resample_batch(t_new, t_src, g_src))
        return g_src_interp

def load_acc(csv_path: Path, t_new: ScalarBatch) -> Vec3Batch:
        """
        Returns:
                a_src_interp: (N-1,) interpolated acc (Vec3Batch)
        """
        df: pd.DataFrame = load_sorted_frame(csv_path,
                                             ["seconds_elapsed", "z", "y", "x"],
                                             ["seconds_elapsed", "x", "y", "z"])
        t_src: ScalarBatch = as_scalar_batch(df["seconds_elapsed"].to_numpy())
        a_src: Vec3Batch = as_vec3_batch(df[["x", "y", "z"]].to_numpy())

        a_src_interp: Vec3Batch = as_vec3_batch(resample_batch(t_new, t_src, a_src))
        return a_src_interp