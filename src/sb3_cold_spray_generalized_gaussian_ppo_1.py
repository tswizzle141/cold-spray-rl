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

    base_height = float(np.random.uniform(0.001, 0.005))
    base = base_height * np.ones_like(x, dtype=np.float32)

    # ---- RANDOM bump (no hard-coded center/width/amplitude)
    rng = np.random.default_rng()  # random each run
    mu = float(rng.uniform(0.15 * cfg.x_max, 0.85 * cfg.x_max))          # center
    sigma = float(rng.uniform(0.03 * cfg.x_max, 0.12 * cfg.x_max))       # width
    amp = float(rng.uniform(0.0008, 0.0022))                              # amplitude

    bump = (amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)).astype(np.float32)
    h_target = base + bump
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
        n_refine=5000,
        step_size_eta=1e-1,
    )

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