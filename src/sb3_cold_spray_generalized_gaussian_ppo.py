from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
@dataclass
class Slicer1DConfig:
    # Geometry / discretization
    n_bins: int = 256
    x_max: float = 1.0
    n_steps: int = 256  # number of arc-length steps (also number of control points)

    # Process params
    feed_rate: float = 2.0
    sigma: float = 0.05  # Gaussian kernel width

    # Velocity bounds
    v_min: float = 1e-3
    v_max: float = 0.5

    # Slicer iteration (simple fixed-point refinement)
    n_refine: int = 20
    step_size_eta: float = 1e-3  #learning rate

    # Numerical safety
    eps: float = 1e-12

    # ---- Target randomization (for "0.9": harder, non-trivial targets)
    target_seed: Optional[int] = None  # None => random each run
    base_height: float = 0.002

    # random bumps
    n_bumps_min: int = 2
    n_bumps_max: int = 6
    bump_amp_range: Tuple[float, float] = (0.0006, 0.0022)
    bump_sigma_range: Tuple[float, float] = (0.02, 0.12)
    bump_mu_margin: float = 0.05  # keep bumps away from boundaries

    # sinusoidal components
    use_sin: bool = True
    n_sin_min: int = 1
    n_sin_max: int = 3
    sin_amp_range: Tuple[float, float] = (0.0001, 0.0008)
    sin_freq_range: Tuple[float, float] = (1.0, 10.0)  # cycles over [0, x_max]

     # ---- Shape-primitive bumps (rect / triangle / trapezoid / semicircle / cosine-arc)
    use_shapes: bool = True

    n_shapes_min: int = 2
    n_shapes_max: int = 8

    shape_amp_range: Tuple[float, float] = (0.0004, 0.0025)

    # width in x-units (not sigma). Ensure > dx to be representable.
    shape_width_range: Tuple[float, float] = (0.03, 0.25)
    shape_mu_margin: float = 0.05  # keep centers away from boundaries

    # probabilities for each primitive (should sum ~ 1.0; doesn't have to be exact)
    p_rect: float = 0.25
    p_triangle: float = 0.25
    p_trapezoid: float = 0.20
    p_semicircle: float = 0.15
    p_cosine: float = 0.15

    # Optional smoothing for "rect" edges (0 => hard step, >0 => softer edges)
    rect_edge_softness: float = 0.0  # try 0.005..0.02 if you want softer steps

# -----------------------------
# Kernel + Deposition Simulator
# -----------------------------
def kernel_gaussian(x_grid: np.ndarray, x0: float, sigma: float, dx: float) -> np.ndarray:
    """
    Discrete Gaussian kernel weights for a point at x0.
    Uses continuous Gaussian PDF sampled on the grid, multiplied by dx.
    Note: not renormalized near boundaries (mass spills outside domain).
    """
    z = (x_grid - x0) / sigma
    w = np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * np.pi))
    return (w * dx).astype(np.float32)

def simulate_deposition_1d(
    cfg: Slicer1DConfig,
    v_profile: np.ndarray,
    h0: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Forward simulator:
    - Fixed arc-length stepping ds = x_max / n_steps
    - At step i: dt_i = ds_eff / v_i, mass m_i = feed_rate * dt_i
    - Deposit using midpoint quadrature along the traversed segment (same spirit as mentor env)
    Returns:
      h_final, metrics dict
    """
    assert v_profile.ndim == 1
    assert h0.ndim == 1
    assert len(h0) == cfg.n_bins
    assert len(v_profile) == cfg.n_steps

    x_grid = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
    dx = float(x_grid[1] - x_grid[0])
    ds = float(cfg.x_max / cfg.n_steps)

    h = h0.astype(np.float32).copy()
    nozzle_x = 0.0

    total_mass = 0.0

    for i in range(cfg.n_steps):
        v = float(np.clip(v_profile[i], cfg.v_min, cfg.v_max))

        x_prev = nozzle_x
        ds_eff = float(min(ds, cfg.x_max - nozzle_x))
        nozzle_x = float(min(cfg.x_max, nozzle_x + ds_eff))

        if ds_eff <= 0.0:
            break

        dt_step = ds_eff / max(v, cfg.eps)
        m_step = cfg.feed_rate * dt_step
        total_mass += m_step

        # Midpoint quadrature along segment [x_prev, x_prev+ds_eff]
        L = ds_eff
        n_q = max(2, int(np.ceil(L / dx))) if L > 0.0 else 2

        k_acc = np.zeros_like(h, dtype=np.float32)
        for j in range(n_q):
            s_j = (j + 0.5) * L / n_q
            x_j = x_prev + s_j
            k_acc += kernel_gaussian(x_grid, x_j, cfg.sigma, dx)
        k_seg = k_acc / float(n_q)

        h += (m_step * k_seg).astype(np.float32)

    info = {
        "total_mass": float(total_mass),
        "final_x": float(nozzle_x),
    }
    return h, info

# -----------------------------
# Slicer (compute v_base)
# -----------------------------
def make_target_example(cfg: Slicer1DConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A simple demo generator:
      - h0 = 0
      - h_target = a smooth base + Gaussian bump
    You can replace this with your real h0/target later.
    """
    x = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)

    h0 = np.zeros_like(x, dtype=np.float32)

    # -----------------------------
    # Randomized target: multi-bumps + sinusoids (NOT hard-coded)
    # -----------------------------
    rng = np.random.default_rng(cfg.target_seed)

    # Base
    h_target = (cfg.base_height * np.ones_like(x, dtype=np.float32))

    dx = float(x[1] - x[0])

    def _clamp01(a: np.ndarray) -> np.ndarray:
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    def rect_bump(mu: float, width: float, amp: float) -> np.ndarray:
        left = mu - 0.5 * width
        right = mu + 0.5 * width
        if getattr(cfg, "rect_edge_softness", 0.0) > 0.0:
            s = float(cfg.rect_edge_softness)
            # smooth step edges using logistic
            return (amp * (1.0 / (1.0 + np.exp(-(x - left) / s)) - 1.0 / (1.0 + np.exp(-(x - right) / s)))).astype(np.float32)
        else:
            return (amp * ((x >= left) & (x <= right)).astype(np.float32))

    def triangle_bump(mu: float, width: float, amp: float) -> np.ndarray:
        left = mu - 0.5 * width
        right = mu + 0.5 * width
        t = 1.0 - np.abs((x - mu) / (0.5 * width + 1e-12))
        return (amp * _clamp01(t))

    def trapezoid_bump(mu: float, width: float, amp: float, plateau_frac: float) -> np.ndarray:
        # plateau_frac in (0,1): fraction of width that is flat on top
        plateau_frac = float(np.clip(plateau_frac, 0.05, 0.95))
        half = 0.5 * width
        half_top = 0.5 * width * plateau_frac
        left = mu - half
        right = mu + half
        left_top = mu - half_top
        right_top = mu + half_top

        y = np.zeros_like(x, dtype=np.float32)

        # rising edge
        rise = (x >= left) & (x < left_top)
        y[rise] = ((x[rise] - left) / (left_top - left + 1e-12)).astype(np.float32)

        # top plateau
        top = (x >= left_top) & (x <= right_top)
        y[top] = 1.0

        # falling edge
        fall = (x > right_top) & (x <= right)
        y[fall] = ((right - x[fall]) / (right - right_top + 1e-12)).astype(np.float32)

        return (amp * _clamp01(y))

    def semicircle_bump(mu: float, width: float, amp: float) -> np.ndarray:
        # y = amp * sqrt(1 - ((x-mu)/r)^2) on |x-mu|<=r
        r = 0.5 * width
        z = (x - mu) / (r + 1e-12)
        inside = np.abs(z) <= 1.0
        y = np.zeros_like(x, dtype=np.float32)
        y[inside] = np.sqrt(np.maximum(0.0, 1.0 - z[inside] * z[inside])).astype(np.float32)
        return (amp * y)

    def cosine_arc_bump(mu: float, width: float, amp: float) -> np.ndarray:
        # smooth dome: 0 at edges, 1 at center
        half = 0.5 * width
        z = (x - mu) / (half + 1e-12)  # in [-1,1]
        inside = np.abs(z) <= 1.0
        y = np.zeros_like(x, dtype=np.float32)
        # cos(pi*z) gives 1 at z=0, -1 at z=±1 => use 0.5*(1+cos(pi*z))
        y[inside] = (0.5 * (1.0 + np.cos(np.pi * z[inside]))).astype(np.float32)
        return (amp * y)

    # ---- Random shape primitives
    if getattr(cfg, "use_shapes", True):
        n_shapes = int(rng.integers(cfg.n_shapes_min, cfg.n_shapes_max + 1))

        # categorical distribution over shapes
        probs = np.array([
            cfg.p_rect, cfg.p_triangle, cfg.p_trapezoid, cfg.p_semicircle, cfg.p_cosine
        ], dtype=np.float64)
        probs = probs / probs.sum()

        for _ in range(n_shapes):
            width = float(rng.uniform(cfg.shape_width_range[0], cfg.shape_width_range[1]))
            # ensure width isn't smaller than a few grid steps
            width = max(width, 3.0 * dx)

            mu = float(rng.uniform(cfg.shape_mu_margin * cfg.x_max,
                                   (1.0 - cfg.shape_mu_margin) * cfg.x_max))
            amp = float(rng.uniform(cfg.shape_amp_range[0], cfg.shape_amp_range[1]))

            shape_id = int(rng.choice(5, p=probs))

            if shape_id == 0:
                h_target += rect_bump(mu, width, amp)
            elif shape_id == 1:
                h_target += triangle_bump(mu, width, amp)
            elif shape_id == 2:
                plateau_frac = float(rng.uniform(0.2, 0.8))
                h_target += trapezoid_bump(mu, width, amp, plateau_frac)
            elif shape_id == 3:
                h_target += semicircle_bump(mu, width, amp)
            else:
                h_target += cosine_arc_bump(mu, width, amp)

    # Helper gaussian
    def g(mu: float, sig: float, amp: float) -> np.ndarray:
        return (amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)).astype(np.float32)

    # Random number of bumps
    n_bumps = int(rng.integers(cfg.n_bumps_min, cfg.n_bumps_max + 1))
    for _ in range(n_bumps):
        mu = float(rng.uniform(cfg.bump_mu_margin * cfg.x_max,
                               (1.0 - cfg.bump_mu_margin) * cfg.x_max))
        sig = float(rng.uniform(cfg.bump_sigma_range[0], cfg.bump_sigma_range[1]))
        amp = float(rng.uniform(cfg.bump_amp_range[0], cfg.bump_amp_range[1]))
        h_target += g(mu, sig, amp)

    # Smooth window to avoid edgy boundaries (Hann window)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * x / cfg.x_max)).astype(np.float32)

    # Random sinusoids
    if getattr(cfg, "use_sin", True):
        n_sin = int(rng.integers(cfg.n_sin_min, cfg.n_sin_max + 1))
        for _ in range(n_sin):
            amp = float(rng.uniform(cfg.sin_amp_range[0], cfg.sin_amp_range[1]))
            freq = float(rng.uniform(cfg.sin_freq_range[0], cfg.sin_freq_range[1]))  # cycles
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            h_target += (amp * window * np.sin(2.0 * np.pi * freq * x / cfg.x_max + phase)).astype(np.float32)

    # Keep non-negative
    h_target = np.maximum(h_target, 0.0).astype(np.float32)
    return x, h0, h_target.astype(np.float32)

def initial_guess_v_from_delta_h(cfg: Slicer1DConfig, h0: np.ndarray, h_target: np.ndarray) -> np.ndarray:
    """
    Heuristic initialization:
      delta_h = max(h_target - h0, 0)
      map delta_h -> v such that bigger delta_h => smaller v
    This is NOT the full optimization in theory; it is a strong baseline start.
    """
    delta = np.maximum(h_target - h0, 0.0).astype(np.float32)

    # Normalize delta to [0, 1]
    dmax = float(np.max(delta))
    if dmax < cfg.eps:
        # Nothing to add: just go fast everywhere
        return np.full(cfg.n_steps, cfg.v_max, dtype=np.float32)

    dnorm = (delta / dmax).astype(np.float32)

    # Convert spatial delta (n_bins) -> control points (n_steps) by interpolation
    x_bins = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
    x_steps = np.linspace(0.0, cfg.x_max, cfg.n_steps, dtype=np.float32)
    d_steps = np.interp(x_steps, x_bins, dnorm).astype(np.float32)

    # Simple monotone mapping: v = v_max - (v_max - v_min) * d_steps^p
    p = 1.0
    v = cfg.v_max - (cfg.v_max - cfg.v_min) * (d_steps ** p)
    return np.clip(v, cfg.v_min, cfg.v_max).astype(np.float32)

def compute_v_base_refine(
    cfg: Slicer1DConfig,
    h0: np.ndarray,
    h_target: np.ndarray,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    A simple "slicer" that approximates:
        min_v || h0 + Deposition(v) - h_target ||  (overview 0.4.2)  :contentReference[oaicite:4]{index=4}

    We do a lightweight fixed-point refinement:
    - Start from a monotone heuristic v0(delta_h)
    - Loop:
        h_pred = simulate(v)
        error e = h_target - h_pred
        update v(x) <- clip( v(x) - eta * e_interp(x), vmin, vmax )
      (If we are below target (e>0), reduce v to deposit more; if above target, increase v.)

    This is meant to be:
    - easy to implement
    - stable to demo
    - good enough for "Step 1 baseline" discussion with mentor
    """
    v = initial_guess_v_from_delta_h(cfg, h0, h_target)

    x_bins = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
    x_steps = np.linspace(0.0, cfg.x_max, cfg.n_steps, dtype=np.float32)

    last_mse = None
    for it in range(cfg.n_refine):
        h_pred, _ = simulate_deposition_1d(cfg, v, h0)

        err_bins = (h_target - h_pred).astype(np.float32)  # positive => need more material
        mse = float(np.mean(err_bins * err_bins))
        last_mse = mse

        # Map error to step/control grid
        err_steps = np.interp(x_steps, x_bins, err_bins).astype(np.float32)

        # Update: below target => decrease v; above target => increase v
        # Scale update to velocity range to keep it numerically tame
        v_range = (cfg.v_max - cfg.v_min)
        v = v - cfg.step_size_eta * v_range * err_steps

        v = np.clip(v, cfg.v_min, cfg.v_max).astype(np.float32)

        if verbose:
            print(f"[refine {it+1:02d}/{cfg.n_refine}] mse={mse:.6e}, "
                  f"v_min={float(v.min()):.3g}, v_max={float(v.max()):.3g}")

    info = {"final_mse": float(last_mse) if last_mse is not None else float("nan")}
    return v, info

# -----------------------------
# Metrics + Plot
# -----------------------------
def compute_metrics(h_pred: np.ndarray, h_target: np.ndarray) -> Dict[str, float]:
    err = (h_pred - h_target).astype(np.float32)
    mse = float(np.mean(err * err))
    overshoot = np.maximum(err, 0.0)
    deficit = np.maximum(-err, 0.0)
    return {
        "mse": mse,
        "l1_deficit": float(np.sum(deficit)),
        "max_overshoot": float(np.max(overshoot)),
        "l1_overshoot": float(np.sum(overshoot)),}

def plot_result(
    x: np.ndarray,
    h0: np.ndarray,
    h_target: np.ndarray,
    h_pred: np.ndarray,
    v_base: np.ndarray,
    cfg: Slicer1DConfig,
    metrics: Dict[str, float],
):
    x_steps = np.linspace(0.0, cfg.x_max, cfg.n_steps, dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    ax = axes[0]
    ax.plot(x, h0, linewidth=2, color="red", label="h0 (initial)")
    ax.plot(x, h_target, linewidth=2, alpha=0.8, color="green", label="h_target")
    ax.plot(x, h_pred, linewidth=2, alpha=0.8, color="orange", label="h_pred (with v_base)")
    ax.set_title(
        "STEP 1 Slicer baseline: h0 + Deposition(v_base) ≈ h_target\n"
        f"MSE={metrics['mse']:.3e} | L1_def={metrics['l1_deficit']:.3e} | "
        f"max_ov={metrics['max_overshoot']:.3e}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x_steps, v_base, linewidth=2, label="v_base(x)")
    ax2.set_title("Base velocity profile computed by Slicer (output of Step 1)")
    ax2.set_xlabel("x (control points)")
    ax2.set_ylabel("velocity")
    ax2.set_ylim(cfg.v_min * 0.9, cfg.v_max * 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cfg = Slicer1DConfig(
        n_bins=256,
        x_max=1.0,
        n_steps=256,
        feed_rate=0.01,
        sigma=0.05,
        v_min=1e-3,
        v_max=2.0,
        n_refine=3000,
        step_size_eta=1e-1,
        n_shapes_min=4, n_shapes_max=12,
        p_rect=0.35, p_triangle=0.20, p_trapezoid=0.25, p_semicircle=0.10, p_cosine=0.10,
        rect_edge_softness=0.0,
        target_seed=None,)

    # Example data (replace with your real h0 / h_target)
    x, h0, h_target = make_target_example(cfg)

    # STEP 1: compute v_base
    v_base, slicer_info = compute_v_base_refine(cfg, h0, h_target, verbose=True)
    print("Slicer info:", slicer_info)

    # Forward simulate with v_base
    h_pred, sim_info = simulate_deposition_1d(cfg, v_base, h0)
    print("Sim info:", sim_info)

    # Metrics + plot
    metrics = compute_metrics(h_pred, h_target)
    print("Metrics:", metrics)

    plot_result(x, h0, h_target, h_pred, v_base, cfg, metrics)