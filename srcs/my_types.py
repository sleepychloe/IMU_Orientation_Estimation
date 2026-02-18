from __future__ import annotations
from typing import Annotated, TypeAlias, Literal, TypeGuard, cast
import numpy as np
from numpy.typing import NDArray

ScalarBatch: TypeAlias = Annotated[NDArray[np.float64], Literal["shape=(N,) dtype=float64"]]

Vec3: TypeAlias = Annotated[NDArray[np.float64], Literal["shape=(3,) dtype=float64"]]
Quat: TypeAlias = Annotated[NDArray[np.float64], Literal["shape=(4,) dtype=float64"]]

Vec3Batch: TypeAlias = Annotated[NDArray[np.float64], Literal["shape=(N,3) dtype=float64"]]
QuatBatch: TypeAlias = Annotated[NDArray[np.float64], Literal["shape=(N,4) dtype=float64"]]

def is_scalar_batch(x: NDArray[np.float64]) -> TypeGuard[ScalarBatch]:
        return x.ndim == 1

def as_scalar_batch(x: NDArray[np.float64]) -> ScalarBatch:
        if not is_scalar_batch(x):
                raise ValueError(f"Expected ScalarBatch shape (N,), got {x.shape} (ndim={x.ndim})")
        return cast(ScalarBatch, x)

def is_vec3(x: NDArray[np.float64]) -> TypeGuard[Vec3]:
        return x.ndim == 1 and x.shape == (3,)

def as_vec3(x: NDArray[np.float64]) -> Vec3:
        if not is_vec3(x):
                raise ValueError(f"Expected Vec3 shape (3,), got {x.shape} (ndim={x.ndim})")
        return cast(Vec3, x)

def is_quat(x: NDArray[np.float64]) -> TypeGuard[Quat]:
        return x.ndim == 1 and x.shape == (4,)

def as_quat(x: NDArray[np.float64]) -> Quat:
        if not is_quat(x):
                raise ValueError(f"Expected Quat shape (4,), got {x.shape} (ndim={x.ndim})")
        return cast(Quat, x)

def is_vec3_batch(x: NDArray[np.float64]) -> TypeGuard[Vec3Batch]:
        return x.ndim == 2 and x.shape[1] == 3

def as_vec3_batch(x: NDArray[np.float64]) -> Vec3Batch:
        if not is_vec3_batch(x):
                raise ValueError(f"Expected Vec3Batch shape (N,3), got {x.shape} (ndim={x.ndim})")
        return cast(Vec3Batch, x)

def is_quat_batch(x: NDArray[np.float64]) -> TypeGuard[QuatBatch]:
        return x.ndim == 2 and x.shape[1] == 4

def as_quat_batch(x: NDArray[np.float64]) -> QuatBatch:
        if not is_quat_batch(x):
                raise ValueError(f"Expected QuatBatch shape (N,4), got {x.shape} (ndim={x.ndim})")
        return cast(QuatBatch, x)
