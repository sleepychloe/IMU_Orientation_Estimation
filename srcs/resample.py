import numpy as np
from typing import overload

from my_types import ScalarBatch, Vec3Batch, QuatBatch

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
