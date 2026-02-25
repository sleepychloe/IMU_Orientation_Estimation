
 * [Experiment 1](#exp-1) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Setup](#setup) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Quaternion Convention](#setup-quat) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Gyro Propagation](#setup-gyro) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Error Metric](#setup-err) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why trimming is needed (Fair comparison)](#why-trim) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Stabilization Trimming Detector](#why-trim-detector) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Datasets](#data) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Results](#res) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 01 — 5 min](#res-data-01) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 02 — 9 min](#res-data-02) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 03 — 13 min](#res-data-03) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 04 — 96 min](#res-data-04) <br>


<br>
<br>

## Experiment 1 — Gyro-only propagation <a name="exp-1"></a>

### Goal <a name="goal"></a>

This experiment isolates the gyro propagation step to answer two questions:<br>

1. How unstable is gyro-only orientation over time?

2. Why must we trim the initial stabilization period for fair evaluation?

<br>

I intentionally do not apply accelerometer or magnetometer corrections here.<br>
Only the gyroscope is integrated, and the result is compared against REF.<br>

<br>
<br>
<br>
<br>

## Setup <a name="setup"></a>

### Quaternion Convention <a name="setup-quat"></a>

Orientation is a unit quaternion `q` mapping:<br>

```
	q : body → world
```
<br>
<br>
<br>

### Gyro Propagation <a name="setup-gyro"></a>

At each timestep, angular velocity ω is converted into a small delta quaternion Δq, then update:<br>

```
	q_pred = normalize(q ⊗ Δq)
```
<br>
<br>

#### Implementation

(in pipelines.py)

```
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
```

<br>
<br>
<br>

### Error Metric <a name="setup-err"></a>

The angular distance between the gyro estimate and REF:<br>

```
	q_err = q_est⁻¹ ⊗ q_ref
```

<br>

min / max/ mean / p90 are reported in radian and degree.<br>

<br>
<br>

#### Implementation

(in evaluation.py)

```
def calc_angle_err(q_est: QuatBatch, q_ref: QuatBatch) -> ScalarBatch:
        w_err: ScalarBatch = as_scalar_batch(np.empty(len(q_est)))
        for i in range(len(q_est)):
                q_err: Quat = libq.quat_mul(libq.quat_conj(q_est[i]), q_ref[i])
                w_err[i] = np.clip(np.abs(q_err[0]), 0.0, 1.0)
        return as_scalar_batch(2 * np.arccos(w_err))
```

<br>
<br>
<br>
<br>

## Why trimming is needed (Fair comparison) <a name="why-trim"></a>

In real logs, the first seconds often contain transient effects:<br>

- Sensor warm-up and bias settling
- Reference filter convergence (REF is not instantly stable)
- Initial handling motion (phone not yet steady)

<br>

If we start evaluating from `t = 0 s`,<br>
a short transient can permanently dominate error statistics,<br>
and even push the gyro-only estimate into an unrecoverable bad trajectory.<br>

<br>
<br>

Therefore, two conditions are evaluated:<br>

1. [exp 1-1] No initial sample cut
2. [exp 1-2] Initial stabilization trimmed

<br>
<br>

### Stabilization Trimming Detector <a name="why-trim-detector"></a>

Instead of cutting a fixed number of seconds, a detector that searches for a stable window is applied.<br>

<br>

Detector logic:<br>

- Slide a window of length `sample_window` (10s = 1000 samples at 100 Hz)
- Start the gyro integration inside the window using `q_ref[i]` as initial quaternion
- Compute p90 angular error within the window
- Declare stabilization when `p90 < threshold` for consecutive windows

<br>
<br>

Policy notes:<br>

Even if stabilization is detected extremely early,<br>
`min_cut_second` is enforced because realistic logs still contain initial handling/sync artifacts.<br>

<br>
<br>

If stabilization is not found by `max_cut_second`,<br>
`max_cut_second` is applied as fallback cut.<br>

<br>
<br>

Parameters used in this experiment:<br>

- `sample_hz` = 100
- `sample_window` = 1000 (10 seconds)
- `threshold` = 0.5 rad (p90 criterion)
- `consecutive` = 3
- `min_cut_second` = 10
- `max_cut_second` = 30

<br>
<br>

#### Implementation

(in resample.py)

```
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
                        print(f"i: {i} | p90(err): {p90:.10f} | cons: {cons}")
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
                print(f"[WARN] stabilization not found within {max_cut_second}s. applying fallback cut={max_cut_second}s")
                return max_cut_idx
        elif (best_idx / sample_hz) < min_cut_second:
                print(f"[INFO] stabilization detected too early (< min_cut). applying min_cut={min_cut_second}s policy")
                return min_cut_idx
        print(f"[OK] stabilization detected. cut idx {best_idx} (≈ {best_idx / sample_hz:.1f}s)")
        return best_idx
```
<br>
<br>
<br>
<br>

## Datasets <a name="data"></a>

4 data is used for the experiment recorded by Sensor Logger application.<br>

| Dataset | Duration | Measured by | Posture  | Notes                        |
|:--------|---------:|:-----------:|:--------:|:-----------------------------|
| data 01 | 5 min    | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 02 | 9 min    | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 03 | 13 min   | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 04 | 96 min   | B           | unknown  | <ul><li>uncontrolled environment</li><li>indoor ↔ outdoor transition</li><li>pedestrian + public transport(metro/tram)</li></ul> | 

<br>
<br>
<br>
<br>

## Results <a name="res"></a>

Each plot compares:<br>

- blue: exp 1-1
- orange: exp 1-2

<br>
<br>

### Dataset 01 — 5 min <a name="res-data-01"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data01_exp1.png" width="952" height="311">

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data02_exp1.png" width="952" height="311">

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data03_exp1.png" width="952" height="311">

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data04_exp1.png" width="952" height="311">

<br>
<br>
<br>
<br>