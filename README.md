## Lists

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
	(x, y, z) = u * sin(θ/2), where ||u|| = 1
```
<br>
<br>

### Euler Angle vs Quaternion <a name="quaternion-euler-angle-vs-quaternion"></a>

- Euler Angle
	- Represents orientation with 3 sequential rotations (roll/pitch/yaw)
	- Parameterization by sequential axis rotations (stores its process — dependent)
	- Intuitive, but can suffer from gimbal lock(loss of one rotational DOF when axes align)

- Quaternion
	- Unit quaternion representation of 3D orientation (maps to SO(3) up to sign) (stores its result — independent)
	- Rotation can be composed via quaternion multiplication
	- Stable in continuous rotation
	- No gimbal lock
	- Numerically robust for continuous IMU integration
	- `q` and `-q` encode the same orientation (same rotation in 3D space)

<br>
<br>

### Applications of Quaternion <a name="quaternion-applications-of-quaternion"></a>

#### IMU Orientation Update <a name="quaternion-applications-imu"></a>

Using gyroscope angular velocity `𝜔 = (𝜔𝑥, 𝜔𝑦, ω𝑧)`(integrated over time),<br>
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
