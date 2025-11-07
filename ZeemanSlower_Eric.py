"""
ZeemanSlower_Eric.py

Compute and plot a Zeeman-slower magnetic field profile for neutral
Calcium (Ca). Includes an optional interactive matplotlib widget
that lets you vary key parameters with sliders.

Formulae:
- B(z)=B0(1-z/L)^{1/2}+Bbias.


The script is configurable via command-line arguments and saves a PNG.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np


# Physical constants
hh = 6.62607015e-34  # J*s (Planck constant)
hbar = hh / (2.0 * math.pi)
mu_B = 9.274009994e-24  # J/T (Bohr magneton)
amu = 1.66053906660e-27  # kg


def compute_profile(
    mass_amu=40.078,
    wavelength=423.0e-9,
    gamma=2 * math.pi * 34.6e6,
    g_eff=1.0,
    v0=500.0,
    vf=50.0,
    s0=5.0,
    eta=0.5,
    n_points=500,
):
    """Compute z array, velocity v(z) and magnetic field B(z).

    Returns: (z, v, B, L) where B is in Tesla and z in meters.
    """
    m = mass_amu * amu
    k = 2 * math.pi / wavelength

    # Approximate maximum scattering force
    F_max = hbar * k * gamma / 2.0 * (s0 / (1.0 + s0))
    a_max = F_max / m
    a = eta * a_max

    if a <= 0:
        raise ValueError("Computed deceleration <= 0; check s0 and eta")

    if v0 <= vf:
        raise ValueError("Initial velocity v0 must be > final velocity vf")

    # Required length (constant deceleration)
    L = (v0 * v0 - vf * vf) / (2.0 * a)

    z = np.linspace(0.0, L, n_points)
    vz = np.sqrt(np.maximum(0.0, v0 * v0 - 2.0 * a * z))

    # Choose detuning so B(L) = 0 => delta = - k * vf
    # B(z) = hbar * k * (v(z) - vf) / (mu_B * g_eff)
    Bz = (hbar * k * (vz - vf)) / (mu_B * g_eff)

    return z, vz, Bz, L


def plot_profile(z, v, B, L, outpath=None, show=True):
    """Plot B(z) in Gauss and v(z) on a twin axis.

    outpath: optional Path or filename to save PNG.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("z (m)")
    ax1.set_ylabel("B (G)", color=color)
    ax1.plot(z, B * 1e4, color=color, lw=2, label="B (G)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("v (m/s)", color=color)
    ax2.plot(z, v, color=color, lw=2, label="v (m/s)")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Zeeman-slower profile (L = {L:.3f} m)")
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Saved plot to {outpath}")

    if show:
        plt.show()

    plt.close(fig)


def interactive_profile(initial_args):
    """Create an interactive matplotlib figure with sliders.

    initial_args must provide: v0, vf, eta, s0, g, mass, wavelength,
    gamma, points (namespace from argparse is fine).
    """
    z, v, B, L = compute_profile(
        mass_amu=initial_args.mass,
        wavelength=initial_args.wavelength,
        gamma=initial_args.gamma,
        g_eff=initial_args.g,
        v0=initial_args.v0,
        vf=initial_args.vf,
        s0=initial_args.s0,
        eta=initial_args.eta,
        n_points=initial_args.points,
    )

    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(left=0.12, bottom=0.35)

    color = "tab:blue"
    lB, = ax1.plot(z, B * 1e4, color=color, lw=2)
    ax1.set_xlabel("z (m)")
    ax1.set_ylabel("B (G)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    color = "tab:orange"
    lv, = ax2.plot(z, v, color=color, lw=2)
    ax2.set_ylabel("v (m/s)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    title = ax1.set_title(f"Zeeman-slower profile (L = {L:.3f} m)")

    # slider axes
    ax_v0 = plt.axes([0.12, 0.25, 0.75, 0.03])
    ax_vf = plt.axes([0.12, 0.20, 0.75, 0.03])
    ax_eta = plt.axes([0.12, 0.15, 0.75, 0.03])
    ax_s0 = plt.axes([0.12, 0.10, 0.75, 0.03])
    ax_g = plt.axes([0.12, 0.05, 0.75, 0.03])

    s_v0 = Slider(ax_v0, 'v0 (m/s)', 0.0, 1000.0, valinit=initial_args.v0)
    s_vf = Slider(ax_vf, 'vf (m/s)', 0.0, 600.0, valinit=initial_args.vf)
    s_eta = Slider(ax_eta, 'eta', 0.01, 1.0, valinit=initial_args.eta)
    s_s0 = Slider(ax_s0, 's0', 0.1, 50.0, valinit=initial_args.s0)
    s_g = Slider(ax_g, 'g_eff', 0.1, 2.0, valinit=initial_args.g)

    def update(val):
        v0 = s_v0.val
        vf = s_vf.val
        eta = s_eta.val
        s0 = s_s0.val
        g_eff = s_g.val

        try:
            znew, vnew, Bnew, Lnew = compute_profile(
                mass_amu=initial_args.mass,
                wavelength=initial_args.wavelength,
                gamma=initial_args.gamma,
                g_eff=g_eff,
                v0=v0,
                vf=vf,
                s0=s0,
                eta=eta,
                n_points=initial_args.points,
            )
        except Exception as e:
            print("Update skipped due to error:", e)
            return

        lB.set_xdata(znew)
        lB.set_ydata(Bnew * 1e4)
        lv.set_xdata(znew)
        lv.set_ydata(vnew)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        title.set_text(f"Zeeman-slower profile (L = {Lnew:.3f} m)")
        fig.canvas.draw_idle()

    for s in (s_v0, s_vf, s_eta, s_s0, s_g):
        s.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        s_v0.reset()
        s_vf.reset()
        s_eta.reset()
        s_s0.reset()
        s_g.reset()

    button.on_clicked(reset)

    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Zeeman slower magnetic profile for Ca"
    )
    parser.add_argument("--v0", type=float, default=500.0,
                        help="initial speed (m/s)")
    parser.add_argument("--vf", type=float, default=50.0,
                        help="final speed at exit (m/s)")
    parser.add_argument("--eta", type=float, default=0.5,
                        help="fraction of max deceleration (0-1)")
    parser.add_argument("--s0", type=float, default=5.0,
                        help="on-resonance saturation parameter")
    parser.add_argument("--g", type=float, default=1.0,
                        help="effective g-factor of excited state")
    parser.add_argument("--mass", type=float, default=40.078,
                        help="atomic mass (amu)")
    parser.add_argument("--wavelength", type=float, default=423.0e-9,
                        help="wavelength (m)")
    parser.add_argument("--gamma", type=float, default=2 * math.pi * 34.6e6,
                        help="natural linewidth (rad/s)")
    parser.add_argument("--points", type=int, default=800,
                        help="number of points in z")
    parser.add_argument("--out", type=str, default="zeeman_profile_ca.png",
                        help="output image file")
    parser.add_argument("--interactive", action="store_true",
                        help="open interactive slider GUI")
    args = parser.parse_args(argv)

    if args.interactive:
        interactive_profile(args)
        return

    z, v, B, L = compute_profile(
        mass_amu=args.mass,
        wavelength=args.wavelength,
        gamma=args.gamma,
        g_eff=args.g,
        v0=args.v0,
        vf=args.vf,
        s0=args.s0,
        eta=args.eta,
        n_points=args.points,
    )

    Bmax = np.max(B)
    print(f"Computed slower length L = {L:.4f} m")
    print(f"Peak magnetic field B_max = {Bmax*1e4:.2f} G ({Bmax:.4f} T)")

    outpath = Path(args.out).resolve()
    plot_profile(z, v, B, L, outpath=outpath, show=True)


if __name__ == "__main__":
    main()

