Currently in progress


## Lists

 * [Project IMU Orientation Estimation](#project) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Introduction](#project-intro) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why a reference is needed — Sensor Logger orientation as REF)](#project-ref) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why trimming the initial seconds matters — Fair Comparison)](#project-fair-comparison) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experimental roadmap (progressive complexity)](#project-exp) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Experiment 1 — Gyro-only propagation](#project-exp-1) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Experiment 2 — Gyro + Accelerometer (roll/pitch correction)](#project-exp-2) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Experiment 3 — Gyro + Accelerometer + Magnetometer (yaw correction)](#project-exp-3) <br>

 * [Understanding Coordinate Systems and Sensors](#orientation) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Coordinate Frame](#orientation-coordinate) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [World Frame (Inertial Frame)](#orientation-coordinate-world) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Body Frame (Sensor Frame)](#orientation-coordinate-body) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Sensor Model](#orientation-sensor) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Gyroscope](#orientation-sensor-gyro) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Accelerometer](#orientation-sensor-acc) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Magnetometer](#orientation-sensor-mag) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Gravity vs Magnetic Field](#orientation-grav-mag) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Initial Magnetic Reference (Yaw Anchor)](#orientation-grav-mag-init-mag-ref) <br>

 * [Implementation – IMU Orientation Estimation](#implementation) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 1] Gyroscope Propagation](#implementation-gyro) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 2] Accelerometer Correction (Roll/Pitch)](#implementation-acc) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 3] Linear Acceleration Estimation](#implementation-acc-linear) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 4] Magnetometer Correction (Yaw)](#implementation-mag) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Gating Observation After Implementation](#implementation-gating-observation) <br>

 * [Quaternion](#quaternion) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Quaternion](#quaternion-quaternion) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Axis-Angle to Quaternion](#quaternion-axis-angle-to-quaternion) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Euler Angle vs Quaternion](#quaternion-euler-angle-vs-quaternion) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Applications of Quaternion](#quaternion-applications-of-quaternion) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [IMU Orientation Update](#quaternion-applications-imu) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Rotating a Vector with Quaternion](#quaternion-applications-vector) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Rotation](#quaternion-rotation) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [2D Rotation](#quaternion-rotation-2d) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [3D Rotation](#quaternion-rotation-3d) <br>


<br>
<br>

## Project IMU Orientation Estimation <a name="project"></a>

### Introduction <a name="project-intro"></a>

This project focuses on building a custom orientation estimation pipeline from IMU data (gyroscope, accelerometer, magnetometer).<br>

<br>

The goal is not to use a black-box sensor fusion library, but to implement orientation estimation from first principles and understand the full chain of reasoning:<br>
how each sensor behaves, how errors accumulate, and how different correction strategies affect stability and drift.<br>

<br>
<br>

The project emphasizes:<br>
- Low-level quaternion-based orientation modeling (frame conventions, quaternion dynamics, vector rotations)
- Drift analysis of gyro-only integration and why correction is necessary
- Real-world evaluation methodology using recorded logs and quantitative error metrics
- Structured debugging through controlled experiments (ablation-style comparison of gating/correction options)

<br>
<br>

The long-term goal is to build a robust and fully controlled orientation estimation framework that remains stable in real-world dynamic environments (indoor/outdoor transitions, transportation, varying motion patterns).<br>

<br>
<br>
<br>

### Why a reference is needed — Sensor Logger orientation as REF <a name="project-ref"></a>

To debug and evaluate the estimator, this project uses the orientation provided by the Sensor Logger application as a reference (REF).<br>

<br>

REF is not treated as perfect ground truth, it is a practical baseline that helps validate implementation correctness and identify failure modes.<br>

<br>

The workflow is intentionally comparative:<br>

- Extract IMU streams (gyro/acc/mag) from recorded logs
- Extract Sensor Logger orientation as REF
- Produce a custom orientation estimate from IMU integration + corrections
- Compute angular error between the estimate and REF
- Analyze drift, transients, and long-horizon behavior
- Iterate on filtering, gating, and calibration logic

<br>
<br>
<br>

### Why trimming the initial seconds matters — Fair Comparison <a name="project-fair-comparison"></a>

A key observation is that the first seconds of a log often contain transient effects — sensor warm-up, bias settling, initial user motion, and convergence of the reference filter.<br>

<br>

These effects can inflate error metrics even when the estimator is correct, and can mislead tuning decisions.<br>

<br>

For this reason, evaluation is performed after trimming an initial stabilization window.<br>
This provides a more fair comparison focused on the behavior that matters for long-term tracking (gyro drift + correction performance), rather than start-up artifacts.<br>


<br>
<br>
<br>

### Experimental roadmap (progressive complexity) <a name="project-exp"></a>

The estimator is developed and validated through a sequence of experiments,<br>
starting from the simplest model and progressively adding corrections and trust control (gating).<br>
Each stage is evaluated using the same error pipeline against REF.<br>

<br>

This progression makes it possible to separate and understand each error source,<br>
such as gyro bias/drift, gravity misinterpretation during translation, magnetic disturbances indoors, and the interaction between tilt estimation and yaw correction.<br>

<br>
<br>

#### Experiment 1 — Gyro-only propagation <a name="project-exp-1"></a>

- Integrate angular velocity to propagate orientation
- Observe drift accumulation over time
- Justify and quantify the impact of initial trimming using before/after plots and statistics

<br>
<br>

#### Experiment 2 — Gyro + Accelerometer (roll/pitch correction) <a name="project-exp-2"></a>

- [exp 2-1] No gating (always trust accelerometer)
- [exp 2-2] Accelerometer gating (trust gravity only when ||a|| ≈ g0)
- [exp 2-3] Gyro + Accelerometer gating (stationary-aware weighting)

<br>
<br>

#### Experiment 3 — Gyro + Accelerometer + Magnetometer (yaw correction) <a name="project-exp-3"></a>

- [exp 3-1] No gating (always trust magnetometer)
- [exp 3-2] Gyro + Accelerometer gating
- [exp 3-3] Gyro + Accelerometer + Magnetometer gating (norm + innovation gating)

<br>
<br>
<br>
<br>

## Understanding Coordinate Systems and Sensors <a name="orientation"></a>

The physical modeling and implementation logic behind orientation estimation using:<br>

- Gyroscope
- Accelerometer
- Magnetometer
<br>

The orientation is represented using a quaternion `q` that maps:<br>

```
	q : body → world
```

<br>
<br>

### Coordinate Frame <a name="orientation-coordinate"></a>

#### World Frame (Inertial Frame) <a name="orientation-coordinate-world"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/00_coord_world.png" width="610" height="255">

The world frame is fixed.<br>
<br>

Gravity is assumed constant:<br>

```
	g_world = (0, 0, -g0),
	|| g_world|| = g0

	g0 ≈ 9.81 m/s²
```
<br>
<br>


#### Body Frame (Sensor Frame) <a name="orientation-coordinate-body"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/01_coord_body.png" width="610" height="255">

The body frame is attached to the device (smartphone).<br>

All sensors measure in the body frame.<br>

When the device rotates:<br>

```
	g_world = (0, 0, -g0),
	g_body = (g𝑥, g𝑦, g𝑧),
	||g_body|| = g0
```
<br>

Quaternion definition:<body>

```
	v_world = R(q) ⋅ v_body
	v_body = R(q)ᵀ ⋅ v_world
```
<br>
<br>
<br>

## Sensor Model <a name="orientation-sensor"></a>

### Gyroscope <a name="orientation-sensor-gyro"></a>

Measurement model:<br>

```
	ω_meas(t) = ω_true(t) + b_g + n_gyro(t)

	b_g: gyro bias
	n_gyro: measurement noise
```

Bias can be estimated during stationary periods.<br>

<br>
<br>

### Accelerometer <a name="orientation-sensor-acc"></a>

The accelerometer measures proper acceleration.<br>
<br>

World frame:<br>

```
	a_proper_world = a_linear_world - g_world
```
<br>

Body frame:<br>

```
	a_apparent_body = R(q)ᵀ⋅(a_linear_world - g_world)
```
<br>

Measurement model:<br>

```
	a_meas(t) = R(q)ᵀ⋅(a_linear_world - g_world) + b_a + n_acc(t)

	b_a: acc bias
	n_acc: measurement noise
```
<br>

At rest (a_linear_world = 0):<br>

```
	a_meas ≈ - R(q)ᵀ⋅g_world,

	a_meas / ||a_meas|| ≈ - g_body / ||g_body|| = - g_body_unit
```

<br>
<br>

### Magnetometer <a name="orientation-sensor-mag"></a>

Measurement model:<br>
```
	m_meas(t) = A * m_true(t) + b_hi n_mag(t)

	A: soft-iron 3x3 matrix
	b_hi: hard-iron bias
	n_mag: measurement noise
```
<br>

Ideal case:<br>

```
	m_body = R(q)ᵀ⋅ m_world
```
<br>
Distortions include:<br>

- Hard-iron offset
- Soft-iron scaling (3x3 matrix)

<br>
Thus magnetometer reliability requires:<br>

- Norm gate
- Innovation gate
- Calibration

<br>
<br>
<br>

### Gravity vs Magnetic Field <a name="orientation-grav-mag"></a>

Gravity:<br>

- Nearly constant magnitude
- Direction fixed in world frame
- ||g_world|| = g0
<br>

Magnetic field:<br>

- Magnitude varies by location
- Affected by indoor environment
- Yaw reference is not physically absolute
<br>

Two possible heading references in magnetic field:<br>

1. Absolute heading (true north, declination corrected)
2. Relative heading (initial direction = yaw 0, often more stable indoors)

<br>
<br>
<br>

#### Initial Magnetic Reference (Yaw Anchor) <a name="orientation-grav-mag-init-mag-ref"></a>

Goal: `m_ref_world_h`, used to correct yaw drift.<br>
<br>

1. Stationary Detection

```
	For reliable bias estimation:

	1. | ||a_meas|| - g0| ≈ 0
	2. |ω| ≈ 0
```

<br>

2. Horizontal Projection

```
	m̂ = m_meas / ||m_meas||
	m_body_h = m̂ - (m̂ ⋅ g_body_unit) * g_body_unit
	m̂_body_h = m_body_h / ||m_body_h||
```

<br>

3. Transform to world frame

```
	m̂_world_h = R(q_pred)⋅ m̂_body_h
	                    Σ_{t ∈ T} weight(t) * m̂_world_h(t)
	m_ref_world_h = ───────────────────────────────────────
	                 || Σ_{t ∈ T} weight(t) * m̂_world_h(t) ||

	T: initial stable window
	weight(t): weighting (stationary + norm gate)
``` 

<br>

This defines a stable yaw reference without requiring absolute north.<br>

<br>
<br>
<br>
<br>

## Implementation – IMU Orientation Estimation <a name="implementation"></a>

Implementation logic of quaternion-based orientation estimation using:<br>

- Gyroscope (Propagation)
- Accelerometer (Roll/Pitch correction)
- Magnetometer (Yaw correction)
<br>

The quaternion `q` is defined as:<br>

```
	q : body → world
```

<br>
<br>

### [Step 1] Gyroscope Propagation <a name="implementation-gyro"></a>

1. Continuous quaternion dynamics (body → world):<br>

```
	q̇(t) = 1/2 ⋅ q(t) ⊗ Ω(ω𝑚𝑒𝑎𝑠(t))

	Ω(ω) : [0, ω𝑥, ω𝑦, ω𝑧]
```
<br>

2. Discrete integration over dt:<br>

```
	θ = ||ω||⋅dt
	u = ω / ||ω|| (if ||ω|| > 0)
```
<br>

3. Small rotation quaternion ∆q𝑔𝑦𝑟𝑜:<br>

```
	        ┏              ┓
	        ┃   cos(θ/2)   ┃
	∆q𝑔𝑦𝑟𝑜 = ┃  u𝑥⋅sin(θ/2) ┃
	        ┃  u𝑦⋅sin(θ/2) ┃
	        ┃  u𝑧⋅sin(θ/2) ┃
	        ┗              ┛
```
<br>

4. Prediction update:<br>

```
	q𝑝𝑟𝑒𝑑 = normalize(q ⊗ ∆q𝑔𝑦𝑟𝑜)
```

<br>
<br>

### [Step 2] Accelerometer Correction (Roll/Pitch) <a name="implementation-acc"></a>

1. World gravity direction:<br>

```
	g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡 = g𝑤𝑜𝑟𝑙𝑑 / ||g𝑤𝑜𝑟𝑙𝑑|| = (0, 0, -1)
```
<br>

2.  Predicted gravity direction in body frame:<br>

```
	g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 = R(q𝑝𝑟𝑒𝑑)ᵀ⋅g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡
```
<br>

3. Error axis:<br>

```
	e_axis𝑎𝑐𝑐 = g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 × (-a𝑚𝑒𝑎𝑠_𝑢𝑛𝑖𝑡)

	||e_axis𝑎𝑐𝑐|| = sinφ
```
<br>

4. Accel gating:<br>

```
	dev𝑎𝑐𝑐 = | ||a𝑚𝑒𝑎𝑠|| - g0 |
	weight𝑎𝑐𝑐 = exp( -1/2 * (dev𝑎𝑐𝑐 / σ𝑎𝑐𝑐)²)

	σ𝑎𝑐𝑐: accel gating sigma
```
<br>

(optional) Gyro gating:<br>

```
	weight𝑔𝑦𝑟𝑜 = exp( -1/2 * ( ω𝑛𝑜𝑟𝑚 / σ𝑔𝑦𝑟𝑜)²)
	weight𝑎𝑐𝑐 = weight𝑎𝑐𝑐 * weight𝑔𝑦𝑟𝑜

	σ𝑔𝑦𝑟𝑜: gyro gating sigma
```
<br>

5. Correction quaternion ∆q𝑐𝑜𝑟𝑟:<br>

```
	        ┏                                     ┓
	        ┃                  1                  ┃
	∆q𝑐𝑜𝑟𝑟 = ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑥 ┃
	        ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑦 ┃
	        ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑧 ┃
	        ┗                                     ┛
```
<br>

6. Update:<br>

```
	q̂ = normalize(q𝑝𝑟𝑒𝑑 ⊗ ∆q𝑐𝑜𝑟𝑟)
```


<br>
<br>

### [Step 3] Linear Acceleration Estimation <a name="implementation-acc-linear"></a>

1. Predicted gravity vector in body frame:<br>

```
	g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 = g0 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡
```
<br>

2. Predicted linear acceleration in body frame:<br>
```
	a𝑒𝑠𝑡_𝑙𝑖𝑛𝑒𝑎𝑟_𝑏𝑜𝑑𝑦 ≈ a𝑚𝑒𝑎𝑠 + g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦
```
<br>
<br>

### [Step 4] Magnetometer Correction (Yaw) <a name="implementation-mag"></a>

Yaw does not change gravity direction:<br>

```
	R𝑧(ψ)ᵀ ⋅ g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡 = g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡
```
<br>
Thus magnetometer is required for heading correction.<br>
<br>

1. Tilt compensation(Projecting mag onto the horizontal plane — remove gravity component):<br>

```
	Normalize magnetic measurement:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 = m𝑚𝑒𝑎𝑠 / ||m𝑚𝑒𝑎𝑠||

	Remove gravity component:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ = m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 - (m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡

	Normalize:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 = m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ / ||m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ||
```
<br>

2. Predicted magnetic direction:<br>

```
	Predicted magnetic field in body frame:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 = R(q𝑝𝑟𝑒𝑑)ᵀ⋅m𝑟𝑒𝑓_𝑤𝑜𝑟𝑙𝑑_ℎ

	Remove gravity component:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 - (m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡

	Normalize:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ / ||m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ||
```
<br>

3. Error axis:<br>

```
	e_axis𝑚𝑎𝑔 = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 × m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡

	||e_axis𝑚𝑎𝑔|| = sinφ
```
<br>

4. Magnetometer gating<br>

4-1. Norm gate:<br>

```
	m₀ = median( ||m𝑚𝑒𝑎𝑠|| )
	dev𝑚𝑎𝑔 = | ||m𝑚𝑒𝑎𝑠|| - m₀ |
	weight𝑚𝑎𝑔 = exp ( -1/2 * (dev𝑚𝑎𝑔 / σ𝑚𝑎𝑔)²)

	σ𝑚𝑎𝑔: mag gating sigma
```
<br>

Use `median` because it is robust to outlier.<br>

<br>

(optional) Gyro gating:<br>

```
	weight𝑔𝑦𝑟𝑜 = exp( -1/2 * ( ω𝑛𝑜𝑟𝑚 / σ𝑔𝑦𝑟𝑜)²)
	weight𝑚𝑎𝑔 = weight𝑚𝑎𝑔 * weight𝑔𝑦𝑟𝑜

	σ𝑔𝑦𝑟𝑜: gyro gating sigma
```
<br>

4-2. Innovation gate:<br>

```
	weight𝑚𝑎𝑔 = weight𝑚𝑎𝑔 * exp( -1/2 * (||e_axis𝑚𝑎𝑔|| / σ𝑒_𝑚𝑎𝑔)² )

	σ𝑒_𝑚𝑎𝑔: mag error sigma (σ𝑒_𝑚𝑎𝑔 and σ𝑚𝑎𝑔 above is independent value)
```
<br>

Reduce impact of mag if ||e_axis𝑚𝑎𝑔|| changes abruptly.<br>

<br>
<br>

5. Correction quaternion ∆q𝑐𝑜𝑟𝑟:<br>

Total correction vector:<br>

```
	e𝑡𝑜𝑡𝑎𝑙 = K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐 + K𝑚𝑎𝑔 ⋅ weight𝑚𝑎𝑔 ⋅ e_axis𝑚𝑎𝑔

	       ┏                    ┓
	       ┃          1         ┃
	∆𝑐𝑜𝑟𝑟 = ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑥 ┃
	       ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑦 ┃
	       ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑧 ┃
	       ┗                    ┛
```
<br>

6. Final update:<br>

```
	q̂ = normalize(q𝑝𝑟𝑒𝑑 ⊗ ∆q𝑐𝑜𝑟𝑟)
```

<br>
<br>

### Gating Observation After Implementation <a name="implementation-gating-observation"></a>

I observed that gyro-based stationary gating improves Gyro+Acc performance, but can degrade Gyro+Acc+Mag performance.<br>

<br>

In Gyro+Acc, a gyro-based stationary gate (small ||ω||) helps because it reduces the probability of applying accelerometer-based gravity correction during motion.<br>

<br>

During translation, ||a𝑚𝑒𝑎𝑠|| often deviates from g0,<br>
meaning the accelerometer direction is no longer a clean gravity measurement and can inject incorrect tilt corrections.<br>

<br>
<br>

However, in Gyro+Acc+Mag, magnetometer yaw correction depends on a stable tilt estimate (roll/pitch),<br>
because tilt compensation projects the measured magnetic vector onto the gravity-defined horizontal plane:<br>

```
	m_ℎ = m_𝑢𝑛𝑖𝑡 - (m_𝑢𝑛𝑖𝑡 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡
```
<br>

If gyro gating suppresses accel correction too aggressively (especially during rotation or dynamic motion), roll/pitch estimation can become noisier (gyro drift + less accel correction).<br>
The destabilizes g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡, which makes tilt compensation less reliable, and can corrupt the magnetometer error axis,<br>
worsening the yaw correction even when mag gating is enabled.<br>

<br>
<br>

Conculution:<br>

1. Use gyro-based gating primarily to control accelerometer trust (gravity correction)

2. For magnetometer updates, prefer magnitude/innovation gates, and avoid fully disabling mag updates during rotation

<br>
<br>
<br>
<br>


## Quaternion <a name="quaternion"></a>

### Quaternion <a name="quaternion-quaternion"></a>

In 3D, a rotation must represent both:<br>
- rotation axis
- rotation angle
<br>

Quaternion is a 4D vector representation for expressing 3D rotations.<br>
<br>

A unit quaternion is written as:<br>

```
	q = (w, x, y, z),

	where w is a scalar component,
	(x, y, z) are the vector components.

	and unit-norm condition:
	||q|| = 1
```

or equivalently:<br>

```
	q = w + xi + yj + zk,

	where i, j, k are imaginary basis elements satisfying:
	i² = j² = k² = ijk = -1,
	ij = k, jk = i, ki = j,
	ji = -k, kj = -i, ik = -j,

	and unit-norm condition:
	w² + x² + y² + z² = 1
```

Because of constraint `w² + x² + y² + z² = 1`, the actual degrees of freedom are 3,<br>
when a quaternion is 4D(4 values).<br>

Also, note that `q` and `-q` represent the same physical 3D rotation<br>
(double-cover property of unit quaternions over SO(3)).<br>
<br>
<br>

### Axis-Angle to Quaternion <a name="quaternion-axis-angle-to-quaternion"></a>

A quaternion expresses "rotation by θ around some axis".<br>

```
	q = (cos(θ/2), sin(θ/2)⋅u𝑥, sin(θ/2)⋅u𝑦, sin(θ/2)⋅u𝑧)

	θ: rotation angle
	u(u𝑥, u𝑦, u𝑧): rotation axis unit vector

	in other words:

	w = cos(θ/2),
	(x, y, z) = u⋅sin(θ/2), where ||u|| = 1
```
<br>
<br>

### Euler Angle vs Quaternion <a name="quaternion-euler-angle-vs-quaternion"></a>

- Euler Angle
	- Represents orientation with 3 sequential rotations (roll/pitch/yaw)
	- Parameterization by sequential axis rotations (stores its process — dependent)
	- Intuitive, but can suffer from gimbal lock(loss of one rotational DOF when axes align)

- Quaternion
	- Unit quaternion representation of 3D orientation, maps to SO(3) up to sign (stores its result — independent)
	- Rotation can be composed via quaternion multiplication
	- Stable in continuous rotation
	- No gimbal lock
	- Numerically robust for continuous IMU integration
	- `q` and `-q` encode the same orientation (same rotation in 3D space)

<br>
<br>

### Applications of Quaternion <a name="quaternion-applications-of-quaternion"></a>

#### IMU Orientation Update <a name="quaternion-applications-imu"></a>

Using gyroscope angular velocity `ω = (ω𝑥, ω𝑦, ω𝑧)`(integrated over time),<br>
orientation is propagated by:<br>

```
	qₜ₊Δₜ = qₜ ⊗ Δq

	⊗: quaternion multiplication,
	Δq: incremental rotation over Δt.
```

This is the core orientation prediction step in inertial navigation.<br>

<br>
<br>

#### Rotating a Vector with Quaternion <a name="quaternion-applications-vector"></a>

To rotate a vector v into world frame:<br>

```
	v𝑤𝑜𝑟𝑙𝑑 = q ⊗ v ⊗ q⁻¹

	v: embed as pure quaternion (0, v)

	(Depending on active/passive rotation convention and frame definition,
	equivalent forms such as `q ⊗ v ⊗ q⁻¹` or `q⁻¹ ⊗ v ⊗ q` may appear)
```

This is used to separate gravity, or convert sensor-frame quantities into global coordinates.<br>

<br>
<br>
<br>

### Rotation <a name="quaternion-rotation"></a>

#### 2D Rotation <a name="quaternion-rotation-2d"></a>

In 2D (complex plane), Euler's formula gives:<br>

```
	eⁱᶿ = cosθ + i⋅sinθ

	(A point on the unit circle in the complex plane)
```

Taylor expansion of exponential function:<br>

```
	1. by Taylor series, any smooth function can be expanded like:

	   f(x) = f(0) + f′(0)⋅x + f″(0)/2!⋅x² + f′″(0)/3!⋅x³ + ⋅⋅⋅.


	                             d
	2. on exponential function, ── eˣ = eˣ, so all derivatives are 1 at x=0.
	                            dx


	3. eⁱᶿ = 1 + iθ + (iθ)²/2! + (iθ)³/3! + (iθ)⁴/4! + ⋅⋅⋅,

	   since i² = -1, when we divide the even/odd degree term,

	   [even degree term] 1 + (iθ)²/2! + (iθ)⁴/4! + ⋅⋅⋅
	                       = 1 - θ²/2! + θ⁴/4! - θ⁶/6! + ⋅⋅⋅
			       = cosθ
	   [odd degree term] iθ + (iθ)³/3! + (iθ)⁵/5! + ⋅⋅⋅
	                      = i⋅(θ - θ³/3! + θ⁵/5! - θ⁷/7! + ⋅⋅⋅)
			      = i⋅sinθ
	
	∴ eⁱᶿ = cosθ + i⋅sinθ
```

2D Rotation is a movement of θ on a unit circle.<br>
Also, complex number multiplication is angle addition.<br>

```
	eⁱᵃ ⋅ eⁱᵇ = eⁱ⁽ᵃ⁺ᵇ⁾ = cos(a+b) + i⋅sin(a+b)
```

<br>
<br>

#### 3D Rotation <a name="quaternion-rotation-3d"></a>

Whereas 2D has only one rotation axis (z-axis), 3D has infinitely many axes of rotation.<br>
That's why 2D rotations can be represented by complex numbers,<br>
but general 3D rotations require quaternions.<br>

In quaternion q = (w, u𝑥, u𝑦, u𝑧),<br>
u is u𝑥⋅i + u𝑦⋅j + u𝑧⋅k,<br>
when u is unit axis, u² = -1.<br>

Taylor expansion of exponential function:<br>

```
	eᵘᶿ = 1 + uθ + (uθ)²/2! + (uθ)³/3! + (uθ)⁴/4! + ⋅⋅⋅,

	   since u² = -1, when we divide the even/odd degree term,

	   [even degree term] 1 + (uθ)²/2! + (uθ)⁴/4! + ⋅⋅⋅
	                       = 1 - θ²/2! + θ⁴/4! - θ⁶/6! + ⋅⋅⋅
			       = cosθ
	   [odd degree term] uθ + (uθ)³/3! + (uθ)⁵/5! + ⋅⋅⋅
	                      = u⋅(θ - θ³/3! + θ⁵/5! - θ⁷/7! + ⋅⋅⋅)
			      = u⋅sinθ

	∴ eᵘᶿ = cosθ + u⋅sinθ
	      = cosθ + sinθ⋅u𝑥⋅i + sinθ⋅u𝑦⋅j + sinθ⋅u𝑧⋅k
```

How we rotate vector in 3D is `v' = q ⊗ v ⊗ q⁻¹`, which applies quaternion multiplication on both sides.<br>
So when we define `q = cos(θ/2) + u⋅sin(θ/2)`, the result of `v' = q ⊗ v ⊗ q⁻¹` rotates exactly θ.<br>

Since the space where 3d vector(R³) and quaternion(R⁴) exist are different,<br>
If it is simply multiplied from the left like when 2D rotation is applied,<br>
a scalar component that should not be there is created.<br>

That's why 3D rotation is applied by the "sandwich" form,<br>
which is interpreted:<br>

1. lift v into quaternion space (pure quaternion)
2. rotate by quaternion multiplication
3. project back to 3D vector

<br>
<br>
<br>
<br>