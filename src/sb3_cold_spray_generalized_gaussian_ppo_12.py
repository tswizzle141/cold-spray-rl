from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from math import gamma
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SAVE_DIR = Path("/netscratch/nham/logs/pics-ppo-12")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("/netscratch/nham/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ColdSpray2DConfig:
    # -------------------------------------------------------------------------
    # 1) Grid / domain
    # -------------------------------------------------------------------------
    grid_n: int = 64
    x_max: float = 1.0
    y_max: float = 1.0

    # Snake path settings copied in spirit from trial_target_2d.py
    spacing_px: int = 1
    ds: float = 0.02  # arc-length per RL step along the snake centerline

    # -------------------------------------------------------------------------
    # 2) Deposition model
    # -------------------------------------------------------------------------
    sigma: float = 0.02
    beta: float = 1.0
    eta: float = 0.4
    rho_m: float = 6500.0
    samples_per_pixel: float = 1.0
    A_amp: float = 1.0
    feed_rate: float = 2.0

    # -------------------------------------------------------------------------
    # 3) Velocity / offset constraints
    # -------------------------------------------------------------------------
    v_max: float = 2.0
    v_min: float = 1e-2
    base_v: float = 0.5
    delta_v_max: float = 1.0

    # IMPORTANT CHANGE:
    # There is NO offset_base_i anymore.
    # Only the RL action can shift the nozzle laterally.
    off_max: float = 0.05
    offset_limit: float = 0.10

    # -------------------------------------------------------------------------
    # 4) Target construction for the simplified experiment
    # -------------------------------------------------------------------------
    # IMPORTANT CHANGE:
    # We TEMPORARILY remove the whole "build v_base_i from a GP" logic.
    #
    # New simplified target definition:
    #   1) target_base      = a fixed rectangular box over the full domain
    #                         with constant height = rectangular_target_height.
    #   2) target_modified  = target_base + random Gaussian bumps.
    #
    # This makes the nominal target extremely easy to understand before we go
    # back to the more complicated v_base_i construction later.
    rectangular_target_height: float = 0.05

    # ---------------------------------------------------------------------
    # CHANGED: no random robust training anymore.
    # We now use EXACTLY TWO fixed Gaussian bumps with fixed shape and
    # fixed position on top of the flat target_base.
    #
    # target_modified = target_base + bump_1 + bump_2
    # ---------------------------------------------------------------------
    bump1_x0: float = 0.30
    bump1_y0: float = 0.35
    bump1_sigma_x: float = 0.06
    bump1_sigma_y: float = 0.10
    bump1_amp: float = 0.10
    bump1_theta: float = 0.0

    bump2_x0: float = 0.72
    bump2_y0: float = 0.68
    bump2_sigma_x: float = 0.08
    bump2_sigma_y: float = 0.07
    bump2_amp: float = 0.12
    bump2_theta: float = 0.0

    n_envs: int = 4

    action_penalty_v: float = 0.10
    action_penalty_off: float = 0.10
    smoothness_penalty_v: float = 10.0
    smoothness_penalty_off: float = 10.0
    overshoot_weight: float = 10.0
    reward_scale: float = 10.0

    ppo_n_steps: int = 1024
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_learning_rate: float = 1e-4
    ppo_clip_range: float = 0.2
    ppo_target_kl: float = 0.2
    ppo_n_epochs: int = 4
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_log_std_init: float = -5.0
    total_timesteps: int = 500000000

    # -------------------------------------------------------------------------
    # 8) Resume-training config
    # -------------------------------------------------------------------------
    resume: bool = True
    resume_model_path: str = str(CHECKPOINT_DIR / "best_model_12.zip")
    resume_vecnormalize_path: str = str(CHECKPOINT_DIR / "best_model_12_vecnormalize.pkl")
    save_freq: int = 50000

    # ---------------------------------------------------------------------
    # NEW: random bumps config
    # ---------------------------------------------------------------------
    min_n_bumps: int = 1
    max_n_bumps: int = 4

    # amplitude nhỏ (bump thấp)
    bump_amp_min: float = 0.01
    bump_amp_max: float = 0.05

    # sigma lớn (bump rộng, không nhọn)
    bump_sigma_min: float = 0.10
    bump_sigma_max: float = 0.40

    # tránh sát biên
    bump_margin: float = 0.1

def _save_fig(fig, filename: str, dpi: int = 160) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _make_rectangular_target_base(cfg: ColdSpray2DConfig) -> np.ndarray:
    """
    Build the new simplified base target.

    IMPORTANT CHANGE:
    -----------------
    In the previous file, target_base was generated indirectly from
    v_base_i + snake-path deposition physics.

    In THIS simplified version, we intentionally make target_base trivial:
        target_base(x, y) = constant height over the whole 1x1 domain.

    Since the simulation domain itself is normalized to [0, 1] x [0, 1],
    the phrase "1 x 1 x 0.5 rectangular box" means:
        - full width  = 1
        - full height = 1
        - constant deposited height value = 0.5

    On the discrete pixel grid this is simply a constant matrix filled with
    cfg.rectangular_target_height.
    """
    return np.full(
        (cfg.grid_n, cfg.grid_n),
        fill_value=float(cfg.rectangular_target_height),
        dtype=np.float32,)

def _make_random_gaussian_bumps(cfg: ColdSpray2DConfig, rng: np.random.Generator) -> np.ndarray:
    """
    NEW:
    ----
    Random number of bumps (1 → 4), random position, random shape
    BUT constrained to be:
        - low amplitude
        - wide (not sharp)
    """

    n = int(cfg.grid_n)
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    Y, X = np.meshgrid(grid, grid, indexing="ij")

    bump_map = np.zeros((n, n), dtype=np.float64)

    # 🔥 RANDOM number of bumps
    n_bumps = rng.integers(cfg.min_n_bumps, cfg.max_n_bumps + 1)

    for _ in range(n_bumps):
        # 🔥 RANDOM position (avoid edges)
        x0 = rng.uniform(cfg.bump_margin, 1.0 - cfg.bump_margin)
        y0 = rng.uniform(cfg.bump_margin, 1.0 - cfg.bump_margin)

        # 🔥 RANDOM shape (wide bumps)
        sigma_x = rng.uniform(cfg.bump_sigma_min, cfg.bump_sigma_max)
        sigma_y = rng.uniform(cfg.bump_sigma_min, cfg.bump_sigma_max)

        # 🔥 RANDOM amplitude (low bumps)
        amp = rng.uniform(cfg.bump_amp_min, cfg.bump_amp_max)

        # 🔥 random rotation
        theta = rng.uniform(0.0, np.pi)

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        dx = X - x0
        dy = Y - y0

        xr = c * dx + s * dy
        yr = -s * dx + c * dy

        bump = amp * np.exp(
            -0.5 * (
                (xr / max(sigma_x, 1e-12)) ** 2
                + (yr / max(sigma_y, 1e-12)) ** 2
            )
        )

        bump_map += bump

    return bump_map.astype(np.float32)

class DepositionModel2D:
    """
    Converts one path segment into a 2D deposited-height basis map.

    This is the same physical idea as in trial_target_2d.py, but here we use it
    directly inside the RL environment instead of solving an LSQ inverse problem.
    """

    def __init__(
        self,
        grid_n: int,
        sigma: float,
        beta: float,
        eta: float,
        rho_m: float,
        samples_per_pixel: float,
        A_amp: float,
    ) -> None:
        self.n = int(grid_n)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.eta = float(eta)
        self.rho_m = float(rho_m)
        self.spp = float(samples_per_pixel)
        self.A_amp = float(A_amp)

        self._pix_area = (1.0 / self.n) ** 2
        self._C = self.beta * self._pix_area / (
            2.0 * np.pi * (self.sigma ** 2) * gamma(2.0 / self.beta)
        )

        grid = (np.arange(self.n, dtype=np.float64) + 0.5) / self.n
        self.Y, self.X = np.meshgrid(grid, grid, indexing="ij")

    def _kernel(self, r: np.ndarray) -> np.ndarray:
        return self.A_amp * self._C * np.exp(-((r / self.sigma) ** self.beta))

    def basis_for_segment(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        L = float(np.hypot(dx, dy))

        if L < 1e-12:
            r = np.hypot(self.X - x0, self.Y - y0)
            k = self._kernel(r)
            return (
                k / (self.rho_m * self._pix_area * (float(np.sum(k)) + 1e-16))
            ).astype(np.float32, copy=False)

        ds_target = (1.0 / self.n) / max(self.spp, 1e-6)
        S = max(1, int(np.ceil(L / ds_target)))

        s = (np.arange(S, dtype=np.float64) + 0.5) * (L / S)
        xs = x0 + (dx / L) * s
        ys = y0 + (dy / L) * s

        acc = np.zeros((self.n, self.n), dtype=np.float64)
        for k in range(S):
            r = np.hypot(self.X - xs[k], self.Y - ys[k])
            acc += self._kernel(r)

        basis = (self.eta / (self.rho_m * self._pix_area)) * acc * (1.0 / S)
        return basis.astype(np.float32, copy=False)


# =============================================================================
# Snake pathway (adapted from trial_target_2d.py)
# =============================================================================

class SnakePath2D:
    """
    Build one horizontal snake pathway over the full domain [0,1] x [0,1].
    """

    def __init__(self, cfg: ColdSpray2DConfig, grid_shape: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.grid_shape = grid_shape

    @staticmethod
    def _rebuild_cumlen(pts: np.ndarray) -> np.ndarray:
        dif = pts[1:] - pts[:-1]
        seg = np.sqrt(np.sum(dif * dif, axis=1))
        return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float64)

    def build_single_snake_polyline(self) -> Tuple[np.ndarray, np.ndarray]:
        N = self.grid_shape[0]
        ix0, ix1 = 0, N - 1
        iy0, iy1 = 0, N - 1

        def pix2xy(px: int, py: int) -> Tuple[float, float]:
            return (px + 0.5) / N, (py + 0.5) / N

        pts: List[List[float]] = []
        direction = 1
        y = iy0

        while y <= iy1:
            x_start, x_end = (ix0, ix1) if direction == 1 else (ix1, ix0)
            p0 = pix2xy(x_start, y)
            p1 = pix2xy(x_end, y)

            if not pts:
                pts.append([p0[0], p0[1]])
            elif abs(pts[-1][0] - p0[0]) > 1e-12 or abs(pts[-1][1] - p0[1]) > 1e-12:
                pts.append([p0[0], p0[1]])

            pts.append([p1[0], p1[1]])

            y_next = y + self.cfg.spacing_px
            if y_next <= iy1:
                pv0 = pix2xy(x_end, y)
                pv1 = pix2xy(x_end, y_next)
                if abs(pts[-1][0] - pv0[0]) > 1e-12 or abs(pts[-1][1] - pv0[1]) > 1e-12:
                    pts.append([pv0[0], pv0[1]])
                pts.append([pv1[0], pv1[1]])

            direction *= -1
            y = y_next

        pts_np = np.asarray(pts, dtype=np.float64)
        return pts_np, self._rebuild_cumlen(pts_np)

    @staticmethod
    def pose_at_s(pts: np.ndarray, cumlen: np.ndarray, s: float) -> Tuple[float, float, float, float]:
        s_clamped = float(np.clip(s, 0.0, float(cumlen[-1])))
        i = int(np.searchsorted(cumlen, s_clamped, side="right") - 1)
        i = int(np.clip(i, 0, len(cumlen) - 2))

        s0 = float(cumlen[i])
        s1 = float(cumlen[i + 1])
        p0 = pts[i]
        p1 = pts[i + 1]

        alpha = 0.0 if (s1 - s0) < 1e-12 else (s_clamped - s0) / (s1 - s0)
        b = (1.0 - alpha) * p0 + alpha * p1

        d = p1 - p0
        L = float(np.linalg.norm(d) + 1e-12)
        tx = float(d[0] / L)
        ty = float(d[1] / L)
        return float(b[0]), float(b[1]), tx, ty


# =============================================================================
# Simplified target construction: rectangular base + random Gaussian bumps
# =============================================================================

def build_fixed_steps(cfg: ColdSpray2DConfig) -> List[dict]:
    snake = SnakePath2D(cfg, (cfg.grid_n, cfg.grid_n))
    pts, cumlen = snake.build_single_snake_polyline()
    total_len = float(cumlen[-1])

    s_values = np.arange(0.0, total_len, cfg.ds, dtype=np.float64)
    if len(s_values) == 0 or s_values[-1] < total_len:
        s_values = np.append(s_values, total_len).astype(np.float64)

    steps: List[dict] = []
    for i in range(len(s_values) - 1):
        s0 = float(s_values[i])
        s1 = float(s_values[i + 1])

        x0, y0, tx0, ty0 = SnakePath2D.pose_at_s(pts, cumlen, s0)
        x1, y1, tx1, ty1 = SnakePath2D.pose_at_s(pts, cumlen, s1)

        tx = 0.5 * (tx0 + tx1)
        ty = 0.5 * (ty0 + ty1)
        L = float(np.hypot(tx, ty) + 1e-12)
        tx, ty = tx / L, ty / L
        nx, ny = -ty, tx

        steps.append(
            {
                "s0": s0,
                "s1": s1,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "tx": tx,
                "ty": ty,
                "nx": nx,
                "ny": ny,
                "length": max(s1 - s0, 1e-8),
            }
        )
    return steps


def _clip_segment_to_domain(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float, float, float]:
    return (
        float(np.clip(x0, 0.0, 1.0)),
        float(np.clip(y0, 0.0, 1.0)),
        float(np.clip(x1, 0.0, 1.0)),
        float(np.clip(y1, 0.0, 1.0)),
    )


def _step_local_base_velocity(v_base_vec: np.ndarray, step_idx: int) -> float:
    return float(v_base_vec[int(np.clip(step_idx, 0, len(v_base_vec) - 1))])


def _sample_map_at_xy_bilinear(field: np.ndarray, x: float, y: float) -> float:
    # Bilinear interpolation on the pixel-center grid.
    #
    # IMPORTANT CHANGE:
    # We use this to read the SPATIAL GP g_base(x, y) at the center of each
    # snake segment. That gives one local scalar g_base_i per segment without
    # reverting to an index-based GP.
    n_y, n_x = field.shape
    gx = np.clip(x * n_x - 0.5, 0.0, n_x - 1.0)
    gy = np.clip(y * n_y - 0.5, 0.0, n_y - 1.0)

    x0 = int(np.floor(gx))
    x1 = min(x0 + 1, n_x - 1)
    y0 = int(np.floor(gy))
    y1 = min(y0 + 1, n_y - 1)

    wx = float(gx - x0)
    wy = float(gy - y0)

    v00 = float(field[y0, x0])
    v01 = float(field[y0, x1])
    v10 = float(field[y1, x0])
    v11 = float(field[y1, x1])

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )


def _sample_rect_base_and_bump_modified_target(
    rng: np.random.Generator,
    cfg: ColdSpray2DConfig,
    steps: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the new simplified target pair:

        target_base      = fixed rectangular box of height 0.05
        target_modified  = target_base + exactly two fixed Gaussian bumps

    IMPORTANT CHANGE:
    -----------------
    Compared to the previous fully-spatial GP version, we intentionally REMOVE:
      - spatial GP g_base(x, y)
      - v_base_i generated from g_base
      - spatial GP target_modulation

    Instead we use:
      - a very simple constant base target
      - a deterministic additive field with exactly two fixed bumps

    Why do we still return v_base_vec?
    ---------------------------------
    The environment step() still expects one local base velocity value so that
    the action can remain in the same form:

        v = local_base_v + dv

    Since we are NOT studying v_base_i generation right now, we simply keep
    a constant base velocity vector:

        v_base_vec[i] = cfg.base_v   for all steps i

    This keeps the control interface unchanged while making target generation
    much easier to understand.
    """
    n_steps = len(steps)

    # 1) Keep a constant base velocity profile.
    v_base_vec = np.full(n_steps, fill_value=float(cfg.base_v), dtype=np.float32)

    # 2) target_base = simple constant rectangular box over the full domain.
    target_base = _make_rectangular_target_base(cfg)

    # 3) target_modification = exactly two fixed Gaussian bumps.
    target_modulation = _make_random_gaussian_bumps(cfg=cfg, rng=rng)

    # 4) target_modified = target_base + bumps.
    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    # We no longer have a real g_base map in this simplified version.
    # Keep a zero-map placeholder so the rest of the code stays compatible.
    g_base_placeholder = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)

    return v_base_vec, target_base, target, g_base_placeholder, target_modulation


# =============================================================================
# 2D RL environment
# =============================================================================

class ColdSpray2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: ColdSpray2DConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.deposition = DepositionModel2D(
            grid_n=cfg.grid_n,
            sigma=cfg.sigma,
            beta=cfg.beta,
            eta=cfg.eta,
            rho_m=cfg.rho_m,
            samples_per_pixel=cfg.samples_per_pixel,
            A_amp=cfg.A_amp,
        )
        self.steps = build_fixed_steps(cfg)
        self.n_steps = len(self.steps)

        self.height = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.target = np.zeros_like(self.height)
        self.target_base = np.zeros_like(self.height)
        self.target_modulation = np.zeros_like(self.height)

        self.v_base_vec = np.full(self.n_steps, fill_value=cfg.base_v, dtype=np.float32)
        # IMPORTANT CHANGE:
        # In this simplified rectangular-target experiment, we do NOT generate
        # a real g_base map anymore. We keep this array only as a placeholder so
        # logging / plotting code does not break.
        self.g_base = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)

        self.step_idx = 0
        self.velocity = float(cfg.base_v)
        self.offset = 0.0
        self._prev_velocity = float(cfg.base_v)
        self._prev_offset = 0.0

        # IMPORTANT CHANGE:
        # We removed off_base_i from the observation because there is no
        # offset_base_i anymore.
        # Observation now is:
        #   [height_map, height-target, progress, remaining,
        #    v_current, v_base_i, off_current]
        #
        # Note: v_base_i is still present in the observation, but in THIS
        # simplified version it is simply constant and equal to cfg.base_v at
        # every step. We keep it so the action/dynamics interface stays the
        # same as before.
        obs_dim = 2 * cfg.grid_n * cfg.grid_n + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action = [dv_action, abweichung_action]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _current_step(self) -> dict:
        idx = int(np.clip(self.step_idx, 0, self.n_steps - 1))
        return self.steps[idx]

    def _current_base_velocity(self) -> float:
        return _step_local_base_velocity(self.v_base_vec, self.step_idx)

    def _get_obs(self) -> np.ndarray:
        progress = np.float32(self.step_idx / max(self.n_steps, 1))
        remaining = np.float32(1.0 - progress)
        v_base_i = np.float32(self._current_base_velocity())

        obs = np.concatenate(
            [
                self.height.reshape(-1),
                (self.height - self.target).reshape(-1),
                np.array(
                    [
                        progress,
                        remaining,
                        np.float32(self.velocity),
                        v_base_i,
                        np.float32(self.offset),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        (
            self.v_base_vec,
            self.target_base,
            self.target,
            self.g_base,
            self.target_modulation,
        ) = _sample_rect_base_and_bump_modified_target(
            rng=self.np_random,
            cfg=self.cfg,
            steps=self.steps,
        )

        self.height.fill(0.0)
        self.step_idx = 0
        self.velocity = self._current_base_velocity()
        self.offset = 0.0
        self._prev_velocity = self.velocity
        self._prev_offset = self.offset
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        st = self._current_step()
        local_base_v = self._current_base_velocity()

        dv = float(np.clip(action[0], -1.0, 1.0)) * cfg.delta_v_max
        off_delta = float(np.clip(action[1], -1.0, 1.0)) * cfg.off_max

        v = float(np.clip(local_base_v + dv, cfg.v_min, cfg.v_max))

        # IMPORTANT CHANGE:
        # Offset now comes ONLY from the action. There is no local_base_off.
        off = float(np.clip(self.offset + off_delta, -cfg.offset_limit, cfg.offset_limit))

        self.velocity = v
        self.offset = off

        x0 = st["x0"] + off * st["nx"]
        y0 = st["y0"] + off * st["ny"]
        x1 = st["x1"] + off * st["nx"]
        y1 = st["y1"] + off * st["ny"]
        x0, y0, x1, y1 = _clip_segment_to_domain(x0, y0, x1, y1)

        dt_step = float(st["length"]) / max(v, 1e-12)
        m_step = cfg.feed_rate * dt_step
        basis = self.deposition.basis_for_segment(x0, y0, x1, y1)
        delta_h = (m_step * basis).astype(np.float32)

        h_old = self.height.copy()
        h_new = h_old + delta_h
        self.height = h_new

        def_old = np.maximum(self.target - h_old, 0.0)
        def_new = np.maximum(self.target - h_new, 0.0)
        ov_old = np.maximum(h_old - self.target, 0.0)
        ov_new = np.maximum(h_new - self.target, 0.0)

        # Keep the same potential-difference reward logic as the 1D code.
        loss_old = float(np.sum(def_old) + cfg.overshoot_weight * np.sum(ov_old))
        loss_new = float(np.sum(def_new) + cfg.overshoot_weight * np.sum(ov_new))
        reward = (loss_old - loss_new) * cfg.reward_scale

        reward -= cfg.action_penalty_v * (dv / max(cfg.delta_v_max, 1e-12)) ** 2
        reward -= cfg.action_penalty_off * (off_delta / max(cfg.off_max, 1e-12)) ** 2

        dv_smooth = (v - self._prev_velocity) / max(cfg.v_max, 1e-12)
        doff_smooth = (off - self._prev_offset) / max(cfg.offset_limit, 1e-12)
        reward -= cfg.smoothness_penalty_v * (dv_smooth ** 2)
        reward -= cfg.smoothness_penalty_off * (doff_smooth ** 2)

        self._prev_velocity = v
        self._prev_offset = off

        err = self.height - self.target
        mse = float(np.mean(err * err))
        ov_max = float(np.max(np.maximum(err, 0.0)))

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.n_steps

        info: Dict[str, Any] = {
            "step_idx": int(self.step_idx),
            "velocity": float(v),
            "local_base_velocity": float(local_base_v),
            "offset": float(off),
            "dv": float(dv),
            "abweichung": float(off_delta),
            "dt_step": float(dt_step),
            "step_mass": float(m_step),
            "g_base_mean": float(np.mean(self.g_base)),  # placeholder = 0 here
            "g_base_std": float(np.std(self.g_base)),
            "target_mod_mean": float(np.mean(self.target_modulation)),
            "target_mod_std": float(np.std(self.target_modulation)),
            "mse": float(mse),
            "max_overshoot": float(ov_max),
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# =============================================================================
# SB3 helpers
# =============================================================================

def make_venv(cfg: ColdSpray2DConfig, vecnormalize_path: Optional[str] = None) -> VecNormalize:
    def make_env():
        return Monitor(ColdSpray2DEnv(cfg))

    base_venv = DummyVecEnv([make_env for _ in range(cfg.n_envs)])

    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        venv = VecNormalize.load(vecnormalize_path, base_venv)
        venv.training = True
        venv.norm_reward = True
        print(f"[make_venv] Loaded VecNormalize from: {vecnormalize_path}")
        return venv

    return VecNormalize(base_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def make_ppo(cfg: ColdSpray2DConfig, venv: VecNormalize, seed: int = 0) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs={"log_std_init": cfg.ppo_log_std_init},
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_gae_lambda,
        learning_rate=cfg.ppo_learning_rate,
        clip_range=cfg.ppo_clip_range,
        target_kl=cfg.ppo_target_kl,
        n_epochs=cfg.ppo_n_epochs,
        ent_coef=cfg.ppo_ent_coef,
        vf_coef=cfg.ppo_vf_coef,
        max_grad_norm=cfg.ppo_max_grad_norm,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )


def load_or_create_model(cfg: ColdSpray2DConfig, venv: VecNormalize, seed: int = 0) -> PPO:
    model_path = Path(cfg.resume_model_path)
    if cfg.resume and model_path.exists():
        print(f"[load_or_create_model] Resuming from: {model_path}")
        return PPO.load(model_path, env=venv, device="cuda" if torch.cuda.is_available() else "cpu")
    print("[load_or_create_model] No checkpoint found -> train from scratch")
    return make_ppo(cfg, venv, seed=seed)


class SaveBestMSECallback(BaseCallback):
    def __init__(
        self,
        venv: VecNormalize,
        cfg: ColdSpray2DConfig,
        check_freq: int = 10000,
        save_prefix: str = "best_model_12",
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.venv = venv
        self.cfg = cfg
        self.check_freq = int(check_freq)
        self.save_prefix = save_prefix
        self.deterministic = deterministic
        self.best_mse = np.inf

    def _evaluate_once(self) -> float:
        data = rollout_model(self.model, self.venv, self.cfg, deterministic=self.deterministic)
        mses = data["mses"]
        return float(mses[-1]) if len(mses) > 0 else np.inf

    def _save_best(self, mse: float):
        model_path = CHECKPOINT_DIR / self.save_prefix
        vecnorm_path = CHECKPOINT_DIR / f"{self.save_prefix}_vecnormalize.pkl"
        self.model.save(model_path)
        self.venv.save(vecnorm_path)
        if self.verbose > 0:
            print(f"[SaveBestMSECallback] New best MSE = {mse:.8f} at timestep = {self.num_timesteps}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        mse = self._evaluate_once()
        if self.verbose > 0:
            print(f"[SaveBestMSECallback] Eval at timestep {self.num_timesteps}, MSE = {mse:.8f}, best = {self.best_mse:.8f}")
        if mse < self.best_mse:
            self.best_mse = mse
            self._save_best(mse)
        return True



class VisualizeEveryNEpisodesCallback(BaseCallback):
    """
    Save one rollout summary figure every `every_episodes` completed episodes
    across all vectorized environments.

    We rely on the `episode` key added by `Monitor` when an environment finishes
    an episode (terminated or truncated).
    """

    def __init__(
        self,
        venv: VecNormalize,
        cfg: ColdSpray2DConfig,
        every_episodes: int = 500,
        deterministic: bool = True,
        start_at: int = 0,
        max_plots: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.venv = venv
        self.cfg = cfg
        self.every_episodes = int(every_episodes)
        self.deterministic = deterministic
        self.start_at = int(start_at)
        self.max_plots = None if max_plots is None else int(max_plots)

        if self.every_episodes <= 0:
            raise ValueError("every_episodes must be > 0")

        self.episode_count = 0
        self.plots_done = 0
        self.next_trigger = self._compute_first_trigger()

    def _compute_first_trigger(self) -> int:
        if self.start_at <= 0:
            return self.every_episodes
        k = (self.start_at + self.every_episodes - 1) // self.every_episodes
        return k * self.every_episodes

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            return True

        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and ("episode" in info):
                self.episode_count += 1

                if self.episode_count < self.next_trigger:
                    continue

                if self.max_plots is not None and self.plots_done >= self.max_plots:
                    self.next_trigger += self.every_episodes
                    continue

                data = rollout_model(
                    self.model,
                    self.venv,
                    self.cfg,
                    deterministic=self.deterministic,
                )
                out = plot_rollout_summary(
                    data,
                    save_name=f"train_rollout_after_ep_{self.episode_count:08d}.png",
                )

                self.plots_done += 1
                self.next_trigger += self.every_episodes

                if self.verbose > 0:
                    print(
                        f"[VisualizeEveryNEpisodesCallback] Saved rollout plot at episode {self.episode_count}: {out}"
                    )

        return True

def train_ppo(model: PPO, total_timesteps: int, callback=None) -> PPO:
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
    return model


# =============================================================================
# Rollout / diagnostics
# =============================================================================

def rollout_model(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray2DConfig,
    deterministic: bool,
    max_steps: Optional[int] = None,
):
    env = ColdSpray2DEnv(cfg)
    obs, _ = env.reset()
    T = env.n_steps if max_steps is None else min(max_steps, env.n_steps)

    heights = [env.height.copy()]
    targets = env.target.copy()
    target_base = env.target_base.copy()
    target_modulation = env.target_modulation.copy()
    v_base_vec = env.v_base_vec.copy()

    velocities = [float(env.velocity)]
    offsets = [float(env.offset)]
    mses: List[float] = []
    rewards: List[float] = []
    segments: List[Tuple[float, float, float, float]] = []

    for _ in range(T):
        obs_norm = venv.normalize_obs(obs[None, :].astype(np.float32))
        act, _ = model.predict(obs_norm, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(act[0])

        heights.append(env.height.copy())
        velocities.append(float(info["velocity"]))
        offsets.append(float(info["offset"]))
        mses.append(float(info["mse"]))
        rewards.append(float(r))
        segments.append((info["x0"], info["y0"], info["x1"], info["y1"]))

        if terminated or truncated:
            break

    return {
        "heights": np.stack(heights, axis=0),
        "target": targets,
        "target_base": target_base,
        "target_modulation": target_modulation,
        "v_base_vec": v_base_vec,
        "velocities": np.array(velocities, dtype=np.float32),
        "offsets": np.array(offsets, dtype=np.float32),
        "mses": np.array(mses, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "segments": np.array(segments, dtype=np.float32),
    }


def plot_rollout_summary(data: Dict[str, Any], save_name: Optional[str] = None) -> str:
    target = data["target"]
    target_base = data["target_base"]
    target_modulation = data["target_modulation"]
    height = data["heights"][-1]
    velocities = data["velocities"]
    offsets = data["offsets"]
    v_base_vec = data["v_base_vec"]
    rewards = data["rewards"]
    mses = data["mses"]
    segments = data["segments"]

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(target_base, origin="lower")
    ax1.set_title("target_base (fixed rectangle)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(target, origin="lower")
    ax2.set_title("target_modified (random bumps)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(height, origin="lower")
    ax3.set_title(f"final height | final mse={mses[-1]:.3e}" if len(mses) else "final height")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    t_v = np.arange(len(velocities))
    t_b = np.arange(len(v_base_vec))
    ax4.plot(t_b, v_base_vec, label="v_base_i")
    ax4.plot(t_v, velocities, label="velocity")
    ax4.set_title("velocity tracking")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="best")

    ax5 = fig.add_subplot(2, 3, 5)
    t_o = np.arange(len(offsets))
    ax5.plot(t_o, offsets, label="offset")
    ax5.set_title("offset / abweichung tracking")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="best")

    ax6 = fig.add_subplot(2, 3, 6)
    if len(segments) > 0:
        for x0, y0, x1, y1 in segments:
            ax6.plot([x0, x1], [y0, y1], linewidth=1.0)
    ax6.set_xlim(0.0, 1.0)
    ax6.set_ylim(0.0, 1.0)
    ax6.set_aspect("equal")
    ax6.grid(alpha=0.3)
    ax6.set_title("realized 2D path")

    # Keep target_modulation referenced so it is easy to inspect later if needed.
    _ = target_modulation

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_name is None:
        save_name = f"rollout_12d_snake_{stamp}.png"
    return _save_fig(fig, save_name)


# =============================================================================
# Main example
# =============================================================================

if __name__ == "__main__":
    cfg = ColdSpray2DConfig(
        grid_n=16,
        spacing_px=1,
        ds=0.05,
        sigma=0.02,
        beta=1.0,
        eta=0.4,
        rho_m=6500.0,
        samples_per_pixel=1.0,
        A_amp=1.0,
        feed_rate=2.0,
        v_max=2.0,
        v_min=0.01,
        base_v=0.5,
        delta_v_max=1.0,
        off_max=0.03,
        offset_limit=0.08,
        rectangular_target_height=0.05,
        bump1_x0=0.30,
        bump1_y0=0.35,
        bump1_sigma_x=0.06,
        bump1_sigma_y=0.10,
        bump1_amp=0.10,
        bump1_theta=0.0,
        bump2_x0=0.72,
        bump2_y0=0.68,
        bump2_sigma_x=0.08,
        bump2_sigma_y=0.07,
        bump2_amp=0.12,
        bump2_theta=0.0,
        n_envs=4,
        total_timesteps=500000000,
    )

    venv = make_venv(cfg, vecnormalize_path=cfg.resume_vecnormalize_path if cfg.resume else None)
    model = load_or_create_model(cfg, venv, seed=0)

    best_callback = SaveBestMSECallback(
        venv=venv,
        cfg=cfg,
        check_freq=50000,
        save_prefix="best_model_12",
        deterministic=True,
        verbose=1,
    )

    visualize_callback = VisualizeEveryNEpisodesCallback(
        venv=venv,
        cfg=cfg,
        every_episodes=50000,
        deterministic=True,
        verbose=1,
    )

    callback = CallbackList([best_callback, visualize_callback])

    model = train_ppo(model, total_timesteps=cfg.total_timesteps, callback=callback)

    det = rollout_model(model, venv, cfg, deterministic=True)
    out_png = plot_rollout_summary(det, save_name="final_rollout_12d_snake.png")
    print(f"Saved summary plot to: {out_png}")
