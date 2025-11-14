"""Python translation of the `ZS_RD_commented.m` MATLAB script.

The goal of this module is to provide a like-for-like implementation of the
field-computation utilities used to study Halbach-like magnet assemblies for a
Zeeman slower.  Functions are vectorised with NumPy and can operate on scalars
or arrays.  A small demo in ``main()`` reproduces the plots and calculations
shown in the MATLAB script (SciPy is optional for the non-linear fit).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # SciPy is optional; the demo skips the fit when it is unavailable.
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - optional dependency guard
    least_squares = None  # type: ignore[assignment]


_BETAS = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
_COS_BETAS = np.cos(_BETAS)
_SIN_BETAS = np.sin(_BETAS)
_COS_DOUBLE_BETAS = np.cos(2.0 * _BETAS)
_SIN_DOUBLE_BETAS = np.sin(2.0 * _BETAS)


def _broadcast_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast inputs to a shared shape and return them as float arrays."""

    return np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )


def brond(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the kernel used to build the analytic field of a block magnet."""

    x, y, z = _broadcast_xyz(x, y, z)
    r = np.sqrt(x * x + y * y + z * z)
    # Analytic expressions for the vector potential gradients of a cuboid
    # magnet; each component corresponds to the integral of dB over one axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        b_x = 0.5 * np.log((r - z) / (r + z))
        b_y = -np.arctan2(y * r, x * z)
        b_z = 0.5 * np.log((r - x) / (r + x))
    stacked = np.stack((b_x, b_y, b_z), axis=-1)
    return np.nan_to_num(stacked)


def bmagnet(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
    br: float,
) -> np.ndarray:
    """Field of a rectangular magnet centred at the origin."""

    # Superpose the contributions of the eight cuboid corners while applying
    # the analytical prefactor Br/(4*pi) that appears in the closed-form field.
    prefactor = br / (4.0 * np.pi)
    terms = (
        brond(x - a, y - b, z - c)
        - brond(x + a, y - b, z - c)
        + brond(x + a, y + b, z - c)
        - brond(x - a, y + b, z - c)
        + brond(x - a, y + b, z + c)
        - brond(x - a, y - b, z + c)
        + brond(x + a, y - b, z + c)
        - brond(x + a, y + b, z + c)
    )
    return prefactor * terms


def bdisrot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
    br: float,
    r0: float,
    z0: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Translate/rotate evaluation points before computing a magnet's field."""

    x, y, z = _broadcast_xyz(x, y, z)

    # Translate coordinates into the magnet's local frame.
    xp = x - r0 * np.sin(beta)
    yp = y - r0 * np.cos(beta)
    zp = z - z0

    # First rotation: 2 * beta about the z-axis.
    xs = np.cos(2 * beta) * xp - np.sin(2 * beta) * yp
    ys = np.sin(2 * beta) * xp + np.cos(2 * beta) * yp
    zs = zp

    # Second rotation: alpha around the axis at azimuth beta (puts the magnet
    # tilt on the plane bisecting the ring), which reorients the magnetization.
    cb, sb = np.cos(beta), np.sin(beta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    xt = (
        (cb * cb * (1 - ca) + ca) * xs
        + cb * sb * (1 - ca) * ys
        + sb * sa * zs
    )
    yt = (
        cb * sb * (1 - ca) * xs
        + (sb * sb * (1 - ca) + ca) * ys
        - cb * sa * zs
    )
    zt = -sb * sa * xs + cb * sa * ys + ca * zs

    b_vec = bmagnet(xt, yt, zt, a, b, c, br)
    orig_shape = b_vec.shape[:-1]
    b_flat = b_vec.reshape(-1, 3)

    # Undo the second rotation for vector components so the field returns to
    # the intermediate frame aligned with the ring.
    bxs = (
        (cb * cb * (1 - ca) + ca) * b_flat[:, 0]
        + cb * sb * (1 - ca) * b_flat[:, 1]
        - sb * sa * b_flat[:, 2]
    )
    bys = (
        cb * sb * (1 - ca) * b_flat[:, 0]
        + (sb * sb * (1 - ca) + ca) * b_flat[:, 1]
        + cb * sa * b_flat[:, 2]
    )
    bzs = sb * sa * b_flat[:, 0] - cb * sa * b_flat[:, 1] + ca * b_flat[:, 2]

    # Undo the first rotation to go back to the laboratory frame.
    bx = np.cos(2 * beta) * bxs + np.sin(2 * beta) * bys
    by = -np.sin(2 * beta) * bxs + np.cos(2 * beta) * bys
    bz = bzs

    return np.stack((bx, by, bz), axis=-1).reshape(*orig_shape, 3)


def bhallbach8(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
    br: float,
    r0: float,
    z0: float,
    alpha: float,
) -> np.ndarray:
    """Assemble eight magnets spaced by 45° to form a Halbach-like ring.

    This vectorised implementation avoids Python-level loops by computing the
    contribution of the eight blocks in parallel using the pre-computed
    trigonometric tables above. Broadcasting lets ``r0`` or ``alpha`` be either
    scalars or arrays that match the shape of ``x``.
    """

    x, y, z = _broadcast_xyz(x, y, z)
    r0 = np.asarray(r0, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    r0, z0 = np.broadcast_arrays(r0, z0, subok=True)
    x, y, z, r0, z0 = np.broadcast_arrays(x, y, z, r0, z0)
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), x.shape)

    # Add a beta-axis (size 8) so the eight magnets are evaluated in parallel.
    xp = x[..., None] - r0[..., None] * _SIN_BETAS
    yp = y[..., None] - r0[..., None] * _COS_BETAS
    zp = z[..., None] - z0[..., None]

    xs = _COS_DOUBLE_BETAS * xp - _SIN_DOUBLE_BETAS * yp
    ys = _SIN_DOUBLE_BETAS * xp + _COS_DOUBLE_BETAS * yp
    zs = zp

    cb = _COS_BETAS
    sb = _SIN_BETAS
    ca = np.cos(alpha_arr)[..., None]
    sa = np.sin(alpha_arr)[..., None]
    one_minus_ca = 1.0 - ca

    xt = (
        ((cb * cb) * one_minus_ca + ca) * xs
        + cb * sb * one_minus_ca * ys
        + sb * sa * zs
    )
    yt = (
        cb * sb * one_minus_ca * xs
        + ((sb * sb) * one_minus_ca + ca) * ys
        - cb * sa * zs
    )
    zt = -sb * sa * xs + cb * sa * ys + ca * zs

    # Evaluate the rotated/translated blocks and convert them back to the
    # laboratory frame before summing over the eight azimuthal positions.
    b_vec = bmagnet(xt, yt, zt, a, b, c, br)
    bx_local = b_vec[..., 0]
    by_local = b_vec[..., 1]
    bz_local = b_vec[..., 2]

    bxs = (
        ((cb * cb) * one_minus_ca + ca) * bx_local
        + cb * sb * one_minus_ca * by_local
        - sb * sa * bz_local
    )
    bys = (
        cb * sb * one_minus_ca * bx_local
        + ((sb * sb) * one_minus_ca + ca) * by_local
        + cb * sa * bz_local
    )
    bzs = sb * sa * bx_local - cb * sa * by_local + ca * bz_local

    bx = _COS_DOUBLE_BETAS * bxs + _SIN_DOUBLE_BETAS * bys
    by = -_SIN_DOUBLE_BETAS * bxs + _COS_DOUBLE_BETAS * bys
    bz = bzs

    summed = np.stack(
        (np.sum(bx, axis=-1), np.sum(by, axis=-1), np.sum(bz, axis=-1)),
        axis=-1,
    )
    return summed


def by_component(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
    br: float,
    r0: float,
    z0: float,
    alpha: float,
) -> np.ndarray:
    """Return only the y-component of the assembled Halbach field."""

    return bhallbach8(x, y, z, a, b, c, br, r0, z0, alpha)[..., 1]


def ideal_profile(
    z: np.ndarray, length: float, b0: float, bL: float
) -> np.ndarray:
    """Ideal decreasing-field Zeeman profile used for fitting."""

    # Standard square-root roll-off used for adiabatic Zeeman slowers.
    profile = bL + (b0 - bL) * np.sqrt(np.maximum(0.0, 1.0 - z / length))
    return profile * ((z >= 0.0) & (z <= length))


def build_segment_configuration(
    z_center: float,
    half_length: float,
    radius: float,
    base_alpha: float,
    n_segments: int = 8,
    delta_z: Optional[np.ndarray] = None,
    delta_r: Optional[np.ndarray] = None,
    delta_alpha: Optional[np.ndarray] = None,
) -> list[dict[str, float]]:
    """Split a magnet into segments with individual translations/rotations.

    The helper guarantees that the resulting segments do not overlap along the
    z-axis by sorting the translated centers and validating the spacing.
    """

    # Nominal length per segment before applying perturbations.
    seg_length = 2.0 * half_length / n_segments
    base_centers = (
        z_center - half_length + (np.arange(n_segments) + 0.5) * seg_length
    )

    dz = (
        np.zeros(n_segments)
        if delta_z is None
        else np.asarray(delta_z, dtype=float)
    )
    dr = (
        np.zeros(n_segments)
        if delta_r is None
        else np.asarray(delta_r, dtype=float)
    )
    da = (
        np.zeros(n_segments)
        if delta_alpha is None
        else np.asarray(delta_alpha, dtype=float)
    )

    if dz.size != n_segments or dr.size != n_segments or da.size != n_segments:
        raise ValueError("Delta arrays must have length equal to n_segments")

    centers = base_centers + dz
    order = np.argsort(centers)
    centers = centers[order]
    dr = dr[order]
    da = da[order]

    if np.any(np.diff(centers) < seg_length - 1e-9):
        raise ValueError("Segment translations cause overlap along z-axis")

    segments = []
    for center, delta_r_val, delta_alpha_val in zip(centers, dr, da):
        segments.append(
            {
                "center": float(center),
                "radius": float(radius + delta_r_val),
                "alpha": float(base_alpha + delta_alpha_val),
                "length": float(seg_length),
            }
        )
    return segments


def scattering_force_scalar(
    B_gauss: float,
    velocity: float,
    params: ZeemanParameters,
    s0: float,
    delta_laser: float,
) -> float:
    """Return scattering force magnitude (N) for given B and velocity."""

    hbar = params.planck / (2 * np.pi)
    # Detuning includes laser detuning, Doppler shift (k*v), and Zeeman shift.
    delta = (
        delta_laser
        + params.k_l * velocity
        - params.g_jp * params.mu_b * B_gauss / hbar
    )
    # Lorentzian response: Gamma/2 * s0 / (1 + s0 + (2*delta/Gamma)^2).
    denom = 1.0 + s0 + (2.0 * delta / params.gamma) ** 2
    force = hbar * params.k_l * params.gamma / 2.0 * (s0 / denom)
    return force


def scattering_force_grid(
    B_profile_gauss: np.ndarray,
    velocities: np.ndarray,
    params: ZeemanParameters,
    s0: float,
    delta_laser: float,
) -> np.ndarray:
    """Compute scattering force over (z, v) grid."""

    hbar = params.planck / (2 * np.pi)
    B = B_profile_gauss[:, None]
    v = velocities[None, :]
    delta = delta_laser + params.k_l * v - params.g_jp * params.mu_b * B / hbar
    denom = 1.0 + s0 + (2.0 * delta / params.gamma) ** 2
    force = hbar * params.k_l * params.gamma / 2.0 * (s0 / denom)
    return force


def integrate_trajectory(
    z_axis: np.ndarray,
    B_profile_gauss: np.ndarray,
    v_init: float,
    params: ZeemanParameters,
    s0: float,
    delta_laser: float,
) -> np.ndarray:
    """Simulate 1D velocity vs position under the scattering force."""

    velocities = np.zeros_like(z_axis)
    velocities[0] = v_init
    mass = params.mass
    for idx in range(len(z_axis) - 1):
        dz = z_axis[idx + 1] - z_axis[idx]
        v_curr = max(velocities[idx], 1e-6)
        B_local = B_profile_gauss[idx]
        force = scattering_force_scalar(
            B_local, v_curr, params, s0, delta_laser
        )
        acc = force / mass  # F = m * a
        # dv/dz = (dv/dt)/(dz/dt) = a / v; use explicit Euler for integration.
        dv = -acc * dz / v_curr
        velocities[idx + 1] = max(velocities[idx] + dv, 0.0)
    return velocities


def capture_efficiency(
    s0: np.ndarray | float,
    params: ZeemanParameters,
    v_capture: float,
    v_final: float,
    slower_length: float,
) -> np.ndarray:
    """Return fractional capture efficiency as a function of ``s0``.

    The metric compares the acceleration attainable at a given saturation
    parameter to the acceleration required to bring an atom from ``v_capture``
    down to ``v_final`` within ``slower_length``. Values above one are clipped
    to unity (100% efficiency).
    """

    s0 = np.asarray(s0, dtype=float)
    hbar = params.planck / (2 * np.pi)
    accel_needed = (v_capture * v_capture - v_final * v_final) / (
        2.0 * slower_length
    )
    accel_available = (
        hbar
        * params.k_l
        * params.gamma
        / (2.0 * params.mass)
        * s0
        / (1.0 + s0)
    )
    return np.minimum(accel_available / accel_needed, 1.0)


@dataclass
class ZeemanParameters:
    """Container for the Zeeman slower constants used in the MATLAB script."""

    gamma: float = 1 / 4.6e-9  # natural linewidth (1/s)
    k_l: float = 2 * np.pi / 423e-9  # wavevector for lambda = 423 nm (1/m)
    mass: float = 40 * 1.66e-27  # kg (Ca-40)
    planck: float = 6.626e-34  # J*s
    # Approximate conversion used in the MATLAB script (J/G) for mu_B.
    mu_b: float = 6.626e-34 * 1.4e6
    g_jp: float = 1.0  # Landé g-factor for the excited state

    def derived_quantities(
        self, v0: float, vL: float, eta: float
    ) -> Tuple[float, float, float, float]:
        """Return (Delta_B, B0, BL, L) for the provided capture parameters."""

        hbar = self.planck / (2 * np.pi)
        # Delta_B = hbar*k*(v0 - vL)/(g*mu_B) per the adiabatic slowing
        # condition derived from dv/dz = mu_B*g*B'(z)/(hbar*k).
        delta_b = hbar * self.k_l * (v0 - vL) / (self.g_jp * self.mu_b)
        b0 = 250.0
        bL = b0 + delta_b
        # Slower length from integrating dv/dz = (ħ*k*Gamma/2m)*s0/eta.
        length = self.mass * v0 * v0 / (hbar * self.k_l * self.gamma * eta)
        return delta_b, b0, bL, length


def run_fit(
    zgrid: np.ndarray,
    target_profile: np.ndarray,
    a: float,
    b: float,
    br: float,
    guess: np.ndarray,
) -> Optional[np.ndarray]:
    """Perform the least-squares fit if SciPy is available."""

    if least_squares is None:
        print("SciPy not installed; skipping non-linear fit.")
        return None

    def residual(params: np.ndarray) -> np.ndarray:
        c_len, r0, z0, alpha = params
        model = by_component(0.0, 0.0, zgrid, a, b, c_len, br, r0, z0, alpha)
        return target_profile - model

    result = least_squares(residual, guess, method="trf")
    if not result.success:
        print("Fit did not converge:", result.message)
    return result.x if result.success else None


def main(show_plots: bool = True) -> None:
    """Reproduce the MATLAB script's calculations and plots."""

    if not show_plots:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Magnet geometry specified in millimetres (matching the MATLAB code)
    # ------------------------------------------------------------------
    a_mm = 6.0 / 2.0
    b_mm = 6.0 / 2.0
    c_mm = 128.0 / 2.0
    br = 1.08e4  # Gauss

    r0_mm = 54.0 / 2.0
    z0_mm = 0.0
    alpha = -0.97 * np.pi / 180.0

    zgrid_mm = np.linspace(-200.0, 200.0, 1001)
    field_mm = bhallbach8(
        0.0, 0.0, zgrid_mm, a_mm, b_mm, c_mm, br, r0_mm, z0_mm, alpha
    )

    plt.figure(figsize=(7, 4))
    plt.plot(zgrid_mm, field_mm[:, 0], label="Bx")
    plt.plot(zgrid_mm, field_mm[:, 1], label="By")
    plt.plot(zgrid_mm, field_mm[:, 2], label="Bz")
    plt.axvline(-c_mm, color="k", linestyle="--", alpha=0.4)
    plt.axvline(c_mm, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("z (mm)")
    plt.ylabel("B (G)")
    plt.title("Halbach ring field components along axis")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Spatial distribution (transverse slices) of the magnetic field
    # ------------------------------------------------------------------
    xgrid = np.linspace(-60.0, 60.0, 101)
    ygrid = np.linspace(-60.0, 60.0, 101)
    X, Y = np.meshgrid(xgrid, ygrid)
    slice_field = bhallbach8(
        X,
        Y,
        0.0,
        a_mm,
        b_mm,
        c_mm,
        br,
        r0_mm,
        z0_mm,
        alpha,
    )
    magnitude = np.linalg.norm(slice_field, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    components = [
        ("Bx (G)", slice_field[..., 0]),
        ("By (G)", slice_field[..., 1]),
        ("Bz (G)", slice_field[..., 2]),
        ("|B| (G)", magnitude),
    ]
    for ax, (title, data) in zip(axes.flat, components):
        im = ax.imshow(
            data,
            extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
            origin="lower",
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # ------------------------------------------------------------------
    # Zeeman slower parameter calculations (SI units)
    # ------------------------------------------------------------------
    params = ZeemanParameters()
    s0_design = 20.0  # baseline saturation used in capture estimates
    v0, vL, eta = 1000.0, 50.0, 0.75
    delta_b, b0, bL, length = params.derived_quantities(v0, vL, eta)
    hbar = params.planck / (2 * np.pi)
    delta_L = (
        -params.k_l * (v0 + vL) / 2
        - params.g_jp * params.mu_b / hbar * (bL + b0) / 2
    )
    print(f"Velocity from {v0:.0f} to {vL:.0f} m/s")
    print(f"B field from {b0:.1f} to {bL:.1f} G (Delta_B = {delta_b:.1f} G)")
    print(f"Detuning {delta_L / (2 * np.pi * 1e9):.3f} GHz")
    print(f"Slower length {length * 100:.1f} cm (eta={eta:.2f})")
    capture_eff_at_design = capture_efficiency(
        s0_design, params, v0, vL, length
    )
    print(
        f"Capture efficiency at s0={s0_design:.1f}: "
        f"{capture_eff_at_design:.3f}"
    )

    bid = ideal_profile

    r0_scan = np.linspace(19 / 2 + 2 + a_mm, 50.0, 101)
    by_center = bhallbach8(
        0.0, 0.0, c_mm / 2.0, a_mm, b_mm, c_mm, br, r0_scan, 0.0, 0.0
    )[:, 1]
    plt.figure(figsize=(6, 3.5))
    plt.plot(r0_scan, by_center)
    plt.axhline(bL, color="r", linestyle="--", label="B_L target")
    plt.xlabel("R0 (mm)")
    plt.ylabel("By (G)")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Capture efficiency as a function of saturation parameter s0
    # ------------------------------------------------------------------
    s0_values = np.linspace(0.1, 40.0, 400)
    efficiency_curve = capture_efficiency(s0_values, params, v0, vL, length)
    plt.figure(figsize=(6, 3.5))
    plt.plot(s0_values, efficiency_curve, label="Capture efficiency")
    plt.axvline(
        s0_design,
        color="r",
        linestyle="--",
        label=f"Design s0={s0_design:.1f}",
    )
    plt.xlabel("s0 (saturation parameter)")
    plt.ylabel("Efficiency (fraction)")
    plt.ylim(0.0, 1.05)
    plt.title("Fraction of required deceleration achievable vs. s0")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Fit example using metre-based geometry (matches MATLAB section)
    # ------------------------------------------------------------------
    zgrid = length * np.linspace(-0.3, 1.3, 1001)
    a_m = 6e-3 / 2.0
    b_m = 6e-3 / 2.0

    target = bid(zgrid, length * 1.05, b0, bL)
    initial_guess = np.array([length / 2, 0.028, length / 2, -5 * np.pi / 180])
    fit_params = run_fit(zgrid, target, a_m, b_m, br, initial_guess)

    plt.figure(figsize=(7, 4))
    plt.plot(zgrid, target, "r--", label="Ideal profile")
    if fit_params is not None:
        fit_field = by_component(
            0.0,
            0.0,
            zgrid,
            a_m,
            b_m,
            fit_params[0],
            br,
            fit_params[1],
            fit_params[2],
            fit_params[3],
        )
        plt.plot(zgrid, fit_field, "b-", label="Fitted magnet")
        alpha_deg = fit_params[3] * 180 / np.pi
        print(
            "Fit parameters (c, R0, z0, alpha):",
            f"({fit_params[0]:.4f} m, {fit_params[1]:.4f} m, "
            f"{fit_params[2]:.4f} m, {alpha_deg:.2f} deg)",
        )
        # Additional magnet (second stage) and combined profile
        R2 = 27e-3
        c2 = 6e-3
        z2 = length
        alpha2 = -5 * np.pi / 180.0
        secondary_field = by_component(
            0.0,
            0.0,
            zgrid,
            a_m,
            b_m,
            c2,
            br,
            R2,
            z2,
            alpha2,
        )
        combined_field = fit_field + secondary_field

        fig_combined, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True
        )
        ax_top.plot(zgrid, target, "r--", label="Ideal profile")
        ax_top.plot(zgrid, fit_field, "b-", label="Fitted magnet")
        ax_top.plot(
            zgrid,
            combined_field,
            "m-",
            label="Fitted + secondary magnet",
        )
        ax_top.set_ylabel("B_y (G)")
        ax_top.legend()

        # Simple geometry sketch (projected positions)
        ax_bottom.plot(
            [zgrid.min(), zgrid.max()],
            np.full(2, 1e-3 * (19 / 2 + 2)),
            "k--",
            label="vacuum chamber",
        )
        ax_bottom.plot(
            fit_params[2] + fit_params[0] * np.array([-1, 1]),
            fit_params[1]
            + np.sin(fit_params[3]) * fit_params[0] * np.array([-1, 1]),
            "r-",
            label="primary magnet extent",
        )
        ax_bottom.plot(
            z2 + c2 * np.array([-1, 1]),
            R2 * np.ones(2),
            "m-",
            label="secondary magnet extent",
        )
        ax_bottom.set_xlabel("z (m)")
        ax_bottom.set_ylabel("R (m)")
        ax_bottom.set_xlim(zgrid.min(), zgrid.max())
        ax_bottom.legend()
    # ------------------------------------------------------------------
    # Phase-space trajectories overlaid on scattering force profile
    # ------------------------------------------------------------------
    capture_velocity = 500.0  # m/s
    target_velocity = 50.0  # m/s
    eta_phase = 0.95  # use almost the full available acceleration
    s0_phase = s0_design  # higher saturation for stronger slowing force
    accel_max = (
        hbar * params.k_l * params.gamma / (2.0 * params.mass)
        * s0_phase
        / (1.0 + s0_phase)
    )
    accel_target = eta_phase * accel_max
    length_phase = (
        capture_velocity * capture_velocity
        - target_velocity * target_velocity
    ) / (2.0 * accel_target)
    z_phase = np.linspace(0.0, length_phase, 400)
    vel_profile = np.sqrt(
        np.maximum(
            target_velocity * target_velocity,
            capture_velocity * capture_velocity
            - 2.0 * accel_target * z_phase,
        )
    )
    B0_phase = 250.0  # Gauss
    delta_laser = (
        params.g_jp * params.mu_b * B0_phase / hbar
        - params.k_l * capture_velocity
    )
    B_phase_gauss = (
        hbar
        / (params.g_jp * params.mu_b)
        * (delta_laser + params.k_l * vel_profile)
    )
    vel_axis = np.linspace(target_velocity * 0.5, capture_velocity * 1.05, 360)
    force_grid = scattering_force_grid(
        B_phase_gauss, vel_axis, params, s0_phase, delta_laser
    )

    fig_phase, ax_phase = plt.subplots(figsize=(8, 4))
    im = ax_phase.imshow(
        force_grid.T,
        extent=[z_phase[0], z_phase[-1], vel_axis[0], vel_axis[-1]],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    cbar = fig_phase.colorbar(im, ax=ax_phase)
    cbar.set_label("Scattering force (N)")

    initial_velocities = [480.0, 440.0, 400.0, 360.0]
    for v_init in initial_velocities:
        traj = integrate_trajectory(
            z_phase, B_phase_gauss, v_init, params, s0_phase, delta_laser
        )
        ax_phase.plot(z_phase, traj, label=f"v0={v_init:.0f} m/s")

    ax_phase.set_xlabel("z (m)")
    ax_phase.set_ylabel("v (m/s)")
    ax_phase.set_title(
        "Phase-space trajectories with background scattering force"
    )
    ax_phase.legend()

    plt.xlabel("z (m)")
    plt.ylabel("B_y (G)")
    plt.legend()
    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python port of ZS_RD_commented.m"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip interactive plot display (useful for CI/testing)",
    )
    parsed = parser.parse_args()
    main(show_plots=not parsed.no_plots)
