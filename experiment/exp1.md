
 * [Experiment 1](#exp-1) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-1-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Setup](#exp-1-setup) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Quaternion Convention](#exp-1-setup-quat) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Gyro Propagation](#exp-1-setup-gyro) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Error Metric](#exp-1-setup-err) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why trimming is needed (Fair comparison)](#exp-1-why-trim) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Stabilization Trimming Detector](#exp-1-why-trim-detector) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Datasets](#exp-1-data) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Results](#exp-1-res) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 01 — 5 min](#exp-1-res-data-01) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 02 — 9 min](#exp-1-res-data-02) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 03 — 13 min](#exp-1-res-data-03) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 04 — 96 min](#exp-1-res-data-04) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Cross-dataset Summary](#exp-1-data-sum) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Conclusion](#exp-1-conclusion) <br>


<br>
<br>

## Experiment 1 — Gyro-only propagation <a name="exp-1"></a>

### Goal <a name="exp-1-goal"></a>

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

### Setup <a name="exp-1-setup"></a>

#### Quaternion Convention <a name="exp-1-setup-quat"></a>

Orientation is a unit quaternion `q` mapping:<br>

```
	q : body → world
```
<br>
<br>

#### Gyro Propagation <a name="exp-1-setup-gyro"></a>

At each timestep, angular velocity ω is converted into a small delta quaternion Δq, then updated:<br>

```
	q_pred = normalize(q ⊗ Δq)
```
<br>

##### [Implementation]

```py
# pipelines.py

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

#### Error Metric <a name="exp-1-setup-err"></a>

The angular distance between the gyro estimate and REF:<br>

```
	q_err = q_est⁻¹ ⊗ q_ref

	Angle error is extracted from q_err = (w, x, y, z):
		angle = 2 * acos( clamp(|w|, 0, 1) )
```

<br>

min / max / mean / p90 are reported in radians and degrees.<br>

<br>

##### [Implementation]

```py
# evaluation.py

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

## Why trimming is needed (Fair comparison) <a name="exp-1-why-trim"></a>

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

### Stabilization Trimming Detector <a name="exp-1-why-trim-detector"></a>

Instead of cutting a fixed number of seconds, a detector that searches for a stable window is applied.<br>

<br>

Detector logic:<br>

- Slide a window of length `sample_window` (10s = 1000 samples at 100 Hz)
- Start the gyro integration inside the window using `q_ref[i]` as initial quaternion
- Compute p90 angular error within the window
- Declare stabilization when `p90 < threshold` for consecutive windows

<br>

Policy notes:<br>

Even if stabilization is detected extremely early,<br>
`min_cut_second` is enforced because realistic logs still contain initial handling/sync artifacts.<br>

<br>
If stabilization is not found by `max_cut_second`,<br>
`max_cut_second` is applied as fallback cut.<br>

<br>

Parameters used in this experiment:<br>

- `sample_hz` = 100
- `sample_window` = 1000 (10 seconds)
- `threshold` = 0.5 rad (p90 criterion)
- `consecutive` = 3
- `min_cut_second` = 10
- `max_cut_second` = 30

<br>

`threshold = 0.5` rad was chosen as a conservative stability criterion<br>
to avoid premature stabilization detection under mild motion/noise.<br>

<br>
<br>

##### [Implementation]

```py
# resample.py

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
```
<br>
<br>
<br>
<br>

## Datasets <a name="exp-1-data"></a>

4 datasets are used for the experiment recorded by Sensor Logger application.<br>

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

## Results <a name="exp-1-res"></a>

Each plot compares:<br>

- blue: exp 1-1
- orange: exp 1-2

<br>
<br>

### Dataset 01 — 5 min <a name="exp-1-res-data-01"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data01_exp1.png" width="952" height="311">

#### [Trim decision]

- Stabilization satisfied early `t = 1 s`, but this is considered too early
- Policy enforced (min cut = 10s)

<br>

```
[START] 2026-02-26 20:27:02.739

[INFO] Stabilization detected too early (< min_cut), applying min_cut=10s policy

[END] 2026-02-26 20:27:02.993
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-1 | <ul><li>0.54751 rad</li><li>31.37006 deg</li></ul> | <ul><li>0.62856 rad</li><li>36.01396 deg</li></ul> |
| exp 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |

<br>

```
[START] 2026-02-26 20:27:00.922

Gyro only: no sample cut angle error in rad — min/max/mean/p90
0.016900799939842653 1.0850920070565426 0.5475108313340219 0.6285622393965337

Gyro only: no sample cut angle error in deg — min/max/mean/p90
0.9683445069479396 62.171192387719636 31.370059873138523 36.01396347871306

[END] 2026-02-26 20:27:02.728



[START] 2026-02-26 20:27:03.010

Gyro only: sample cut 10.0s angle error in rad — min/max/mean/p90
0.002692606276233363 1.41733946975559 0.39107310298265935 0.5663011403381867

Gyro only: sample cut 10.0s angle error in deg — min/max/mean/p90
0.15427497551860841 81.2075697543053 22.406838281991387 32.446665274823836

[END] 2026-02-26 20:27:04.854
```

<br>

#### [Observation]

- The trimmed curve shows a cleaner drift trend
- Early segment inflates metrics despite long-term behavior being similar

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-1-res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data02_exp1.png" width="952" height="311">

#### [Trim decision]

- Stabilization satisfied early `t = 3 s`, but this is considered too early
- Policy enforced (min cut = 10s)

<br>

```
[START] 2026-02-26 20:26:15.470

[INFO] Stabilization detected too early (< min_cut), applying min_cut=10s policy

[END] 2026-02-26 20:26:15.976
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-1 | <ul><li>1.90991 rad</li><li>109.42990 deg</li></ul> | <ul><li>2.05530 rad</li><li>117.76002 deg</li></ul> |
| exp 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |

<br>

```
[START] 2026-02-26 20:26:12.539

Gyro only: no sample cut angle error in rad — min/max/mean/p90
0.009464702257402596 2.6688536472865168 1.909912008440155 2.055300097693392

Gyro only: no sample cut angle error in deg — min/max/mean/p90
0.5422874936971117 152.91405012761385 109.42989732497534 117.76002123065714

[END] 2026-02-26 20:26:15.460



[START] 2026-02-26 20:26:15.991

Gyro only: sample cut 10.0s angle error in rad — min/max/mean/p90
0.004019503743449339 0.7197238920239989 0.3838298145823989 0.5440975838643741

Gyro only: sample cut 10.0s angle error in deg — min/max/mean/p90
0.23030060023668236 41.23714142770451 21.991828426860398 31.174495198693997

[END] 2026-02-26 20:26:18.959
```

<br>

#### [Observation]

- Without trimming, the estimator looks much worse than it actually is
- After trimming, drift becomes interpretable (bias accumulation)

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-1-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data03_exp1.png" width="952" height="311">

#### [Trim decision]

- Stabilization detected `t = 23 s`

<br>

```
[START] 2026-02-26 20:24:54.374

[OK] Stabilization detected, cut idx 2300 (≈ 23.0s)

[END] 2026-02-26 20:24:55.835
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-1 | <ul><li>2.98634 rad</li><li>171.10485 deg</li></ul> | <ul><li>3.12675 rad</li><li>179.14982 deg</li></ul> |
| exp 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |

<br>

```
[START] 2026-02-26 20:24:49.801

Gyro only: no sample cut angle error in rad — min/max/mean/p90
0.006497650316757928 3.1415925744092994 2.9863430261847146 3.126754275860046

Gyro only: no sample cut angle error in deg — min/max/mean/p90
0.37228793990207176 179.9999954632919 171.10485157871042 179.14982358126457

[END] 2026-02-26 20:24:54.362



[START] 2026-02-26 20:24:55.846

Gyro only: sample cut 23.0s angle error in rad — min/max/mean/p90
0.00827387251043809 1.2902038650243064 0.5377823159898106 0.8127713667204784

Gyro only: sample cut 23.0s angle error in deg — min/max/mean/p90
0.47405797507741365 73.92323617735929 30.812657002986956 46.56836902216311

[END] 2026-02-26 20:25:00.565
```

<br>

#### [Observation]

- The no-cut run collapses near π radians (≈180° flip), indicating catastrophic divergence
- After trimming, error behaves as expected: steady drift with occasional spikes

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-1-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data04_exp1.png" width="952" height="311">

#### [Trim decision]

- Stabilization satisfied early `t = 0 s`, but this is considered too early
- Policy enforced (min cut = 10s)

<br>

```
[START] 2026-02-26 20:23:56.918

[INFO] Stabilization detected too early (< min_cut), applying min_cut=10s policy

[END] 2026-02-26 20:23:57.101
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-1 | <ul><li>1.04519 rad</li><li>59.88517 deg</li></ul> | <ul><li>2.30597 rad</li><li>132.12219 deg</li></ul> |
| exp 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |

<br>

```
[START] 2026-02-26 20:23:23.880

Gyro only: no sample cut angle error in rad — min/max/mean/p90
0.0017503209553177612 3.1380681380116897 1.0451934334037227 2.3059672108235416

Gyro only: no sample cut angle error in deg — min/max/mean/p90
0.10028600353301406 179.79806013254657 59.88517250882119 132.12218887574306

[END] 2026-02-26 20:23:56.902



[START] 2026-02-26 20:23:57.115

Gyro only: sample cut 10.0s angle error in rad — min/max/mean/p90
0.003631716393424654 3.138840519423176 0.8880786152784693 2.036924604732547

Gyro only: sample cut 10.0s angle error in deg — min/max/mean/p90
0.2080820217317055 179.8423143275991 50.883156531278644 116.70718303752838

[END] 2026-02-26 20:24:28.423
```

<br>

#### [Observation]

- Long uncontrolled motion produces large error spikes regardless
- Still, trimming reduces mean/p90, supporting the presence of a startup transient

<br>
<br>
<br>
<br>

## Cross-dataset Summary <a name="exp-1-data-sum"></a>

| Dataset | Detector passed | Cut  | Mean (no cut) | Mean (cut)    | p90 (no cut)  | p90 (cut)     |
|:--------|----------------:|-----:|--------------:|--------------:|--------------:|--------------:|
| data 01 | 1 s             | 10 s | 31.37006 deg  | 22.40684 deg  | 36.01396 deg  | 32.44667 deg  |
| data 02 | 3 s             | 10 s | 109.42990 deg | 21.99183 deg  | 117.76002 deg | 31.17450 deg  |
| data 03 | 23 s            | 23 s | 171.10485 deg | 30.81266 deg  | 179.14982 deg | 46.56837 deg  |
| data 04 | 0 s             | 10 s | 59.88517 deg  | 50.88316 deg  | 132.12219 deg | 116.70718 deg |

<br>

Across all datasets:<br>

- Initial trimming consistently reduces mean/p90 error
- Without trimming, gyro-only can produce misleadingly catastrophic statistics (especially data03)
- Even after trimming, gyro-only exhibits unavoidable drift over time (bias accumulation)

<br>

Therefore:<br>

Trimming is necessary for fair evaluation, but not sufficient for long-term stability.<br>

<br>
<br>
<br>
<br>

## Conclusion <a name="exp-1-conclusion"></a>

Experiment 1 confirms:<br>

1. Gyro-only integration is inherently unstable over time (drift is unavoidable)
2. The initial stabilization period must be trimmed to avoid transient artifacts dominating results

<br>

Next steps:<br>

- Experiment 2: add accelerometer correction to stabilize roll/pitch and reduce tilt drift
- Experiment 3: add magnetometer correction to constrain yaw drift, with gating for robustness

<br>
<br>
<br>
<br>
