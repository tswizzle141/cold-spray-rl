#!pip install "stable-baselines3[extra]" torch gymnasium

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import shutil
from google.colab import files

@dataclass
class ColdSpray1DConfig:
    n_bins: int = 256
    x_max: float = 1.0
    v_max: float = 2.0
    v_min: float = 1e-2
    base_v: float = 0.5
    delta_v_max: float = 1.0

    feed_rate: float = 2.0
    sigma: float = 0.02
    n_steps: int = 64
    n_envs: int = 1

    # =========================
    # GP-related params
    # =========================
    v_base_gp_amplitude: float = 0.05
    target_mod_gp_amplitude: float = 1e-2
    gp_std: float = 1.0
    gp_lengthscale: float = 0.25
    gp_jitter: float = 1e-8
    gp_resample_every_reset: bool = True

    action_penalty: float = 0.1
    smoothness_penalty: float = 10.0
    overshoot_weight: float = 10.0
    reward_scale: float = 10.0

    # ==========================================================
    # NEW:
    # bump_type dùng để chọn cách sinh target:
    # - "gaussian": giữ nguyên logic GP hiện tại
    # - "rect": v_base constant + target_base + 1 rectangular bump
    # - "circular": v_base constant + target_base + 1 circular-cap bump
    # ==========================================================
    bump_type: str = "gaussian"   # "gaussian"/"rect"/"circular"/"triangle"

    # ==========================================================
    # OLD rect fields:
    # Mình vẫn giữ lại để backward compatibility.
    # Nếu bạn muốn dùng fixed rect thì có thể tận dụng tiếp.
    # ==========================================================
    rect_amp: float = 0.05
    rect_width: float = 0.2

    # ==========================================================
    # NEW rect random-shape params:
    # Mỗi reset ở mode "rect", bump sẽ random:
    # - amplitude trong [rect_amp_min, rect_amp_max]
    # - width     trong [rect_width_min, rect_width_max]
    # - center    random sao cho bump vẫn nằm trong [0, x_max]
    # ==========================================================
    rect_amp_min: float = 0.005
    rect_amp_max: float = 0.03
    rect_width_min: float = 0.30
    rect_width_max: float = 0.60

    # ==========================================================
    # NEW circular bump params:
    # Dùng khi bump_type == "circular"
    # Ý tưởng: tạo 1 circular-cap bump đối xứng, biên tròn mượt hơn rect.
    # - amplitude random trong [circ_amp_min, circ_amp_max]
    # - width random trong [circ_width_min, circ_width_max]
    # - center random sao cho bump nằm trọn trong domain [0, x_max]
    #
    # Công thức profile:
    #   bump(x) = amp * sqrt(max(0, 1 - ((x-center)/radius)^2))
    # với radius = width / 2.
    # => bump cao nhất ở tâm, về 0 tại 2 biên, hình dạng là "nắp tròn".
    # ==========================================================
    circ_amp_min: float = 0.005
    circ_amp_max: float = 0.03
    circ_width_min: float = 0.30
    circ_width_max: float = 0.60

    # ==========================================================
    # NEW triangle bump params:
    # ==========================================================
    tri_amp_min: float = 0.005
    tri_amp_max: float = 0.03
    tri_width_min: float = 0.30
    tri_width_max: float = 0.60

    # kept here for compatibility with training distribution/config structure
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_learning_rate: float = 1e-4
    ppo_clip_range: float = 0.1
    ppo_target_kl: float = 0.05
    ppo_n_epochs: int = 4
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_log_std_init: float = -5.0
    total_timesteps: int = 500000000
    resume: bool = False


def _kernel_gaussian(x_grid: np.ndarray, x0: float, sigma: float, dx: float) -> np.ndarray:
    z = (x_grid - x0) / sigma
    w = np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * np.pi))
    w = (w * dx).astype(np.float32)
    return w


def _rbf_covariance(x: np.ndarray, std: float, lengthscale: float, jitter: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    diff = x[:, None] - x[None, :]
    K = (std ** 2) * np.exp(-0.5 * (diff / max(lengthscale, 1e-12)) ** 2)
    K += jitter * np.eye(len(x), dtype=np.float64)
    return K


def _sample_zero_mean_gp(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    std: float,
    lengthscale: float,
    jitter: float = 1e-8,
) -> np.ndarray:
    K = _rbf_covariance(x_grid, std=std, lengthscale=lengthscale, jitter=jitter)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(len(x_grid))
    gp = L @ z
    return gp.astype(np.float32)


# ======================================================================
# GIỮ NGUYÊN NHÁNH GAUSSIAN:
# Hàm này giữ nguyên logic cũ của bạn:
# - v_base_vec lấy từ GP zero-mean
# - target = target_base + GP modulation
# ======================================================================
def _sample_gp_based_target_base_and_target(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g_base = _sample_zero_mean_gp(
        rng=rng,
        x_grid=x_grid,
        std=cfg.gp_std,
        lengthscale=cfg.gp_lengthscale,
        jitter=cfg.gp_jitter,
    )

    v_mean = float(cfg.base_v)
    a = float(cfg.v_base_gp_amplitude)
    v_base_raw = v_mean + a * g_base.astype(np.float64)
    v_base_vec = np.clip(v_base_raw, cfg.v_min, cfg.v_max).astype(np.float32)

    target_base = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)

    g_mod = _sample_zero_mean_gp(
        rng=rng,
        x_grid=x_grid,
        std=cfg.gp_std,
        lengthscale=cfg.gp_lengthscale,
        jitter=cfg.gp_jitter,
    )

    b = float(cfg.target_mod_gp_amplitude)
    target_modulation = (b * g_mod).astype(np.float32)
    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    return v_base_vec, target_base, target, target_modulation, g_base, g_mod

def _rectangular_bump(
    x_grid: np.ndarray,
    center: float,
    width: float,
    amp: float,
) -> np.ndarray:
    half_w = 0.5 * width
    left = center - half_w
    right = center + half_w
    mask = (x_grid >= left) & (x_grid <= right)
    bump = np.zeros_like(x_grid, dtype=np.float32)
    bump[mask] = np.float32(amp)
    return bump


def _circular_bump(
    x_grid: np.ndarray,
    center: float,
    width: float,
    amp: float,
) -> np.ndarray:
    # ==========================================================
    # NEW:
    # Circular bump dạng "nắp tròn".
    # - width là bề rộng toàn phần của bump
    # - radius = width / 2
    # - trong vùng |x-center| <= radius:
    #       y = amp * sqrt(1 - (d/radius)^2)
    # - ngoài vùng đó: y = 0
    #
    # Như vậy bump:
    # - cực đại = amp tại center
    # - chạm 0 mượt ở hai mép
    # - khác rect ở chỗ không có đỉnh phẳng
    # ==========================================================
    radius = 0.5 * width
    if radius <= 0.0:
        return np.zeros_like(x_grid, dtype=np.float32)

    d = (x_grid - center) / radius
    inside = np.maximum(0.0, 1.0 - d * d)
    bump = amp * np.sqrt(inside)

    # ép ngoài support về 0 rõ ràng để tránh sai số số học rất nhỏ
    bump[np.abs(x_grid - center) > radius] = 0.0
    return bump.astype(np.float32)

def _triangular_bump(
    x_grid: np.ndarray,
    center: float,
    width: float,
    amp: float,
) -> np.ndarray:
    # ==========================================================
    # Triangle bump:
    # - đỉnh tại center với giá trị amp
    # - tuyến tính giảm về 0 ở 2 mép
    # - support: |x-center| <= width/2
    #
    # formula:
    #   bump(x) = amp * max(0, 1 - |x-center| / (width/2))
    # ==========================================================
    half_w = 0.5 * width
    if half_w <= 0.0:
        return np.zeros_like(x_grid, dtype=np.float32)

    d = np.abs(x_grid - center)
    bump = amp * np.maximum(0.0, 1.0 - d / half_w)

    bump[d > half_w] = 0.0
    return bump.astype(np.float32)

def _sample_rect_based_target_base_and_target(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ------------------------------------------------------------
    # (1) v_base_vec constant theo yêu cầu của bạn
    # ------------------------------------------------------------
    v_base_vec = np.full_like(x_grid, fill_value=cfg.base_v, dtype=np.float32)

    # ------------------------------------------------------------
    # (2) target_base sinh từ base_v constant
    #     target_base = feed_rate * dx / v_base
    # ------------------------------------------------------------
    target_base = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)

    # ------------------------------------------------------------
    # (3) Random rectangular bump shape
    #     - amplitude random
    #     - width random
    #     - center random sao cho bump nằm trong domain [0, x_max]
    # ------------------------------------------------------------
    amp = float(rng.uniform(cfg.rect_amp_min, cfg.rect_amp_max))
    width = float(rng.uniform(cfg.rect_width_min, cfg.rect_width_max))

    # Nếu width quá lớn thì chặn lại để không vượt domain
    width = min(width, cfg.x_max)

    half_w = 0.5 * width
    center = float(rng.uniform(half_w, cfg.x_max - half_w)) if width < cfg.x_max else float(0.5 * cfg.x_max)

    target_modulation = _rectangular_bump(
        x_grid=x_grid,
        center=center,
        width=width,
        amp=amp,
    ).astype(np.float32)

    # ------------------------------------------------------------
    # (4) target_modified = target_base + rect bump
    # ------------------------------------------------------------
    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    # ------------------------------------------------------------
    # (5) Để giữ API return giống nhánh gaussian:
    #     - g_base = zeros vì mode rect không dùng GP cho base velocity
    #     - g_mod  = zeros vì mode rect không dùng GP modulation
    # ------------------------------------------------------------
    g_base = np.zeros_like(x_grid, dtype=np.float32)
    g_mod = np.zeros_like(x_grid, dtype=np.float32)

    return v_base_vec, target_base, target, target_modulation, g_base, g_mod


def _sample_circular_based_target_base_and_target(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ------------------------------------------------------------
    # NEW circular branch
    # (1) v_base_vec constant, GIỮ cùng triết lý như mode rect
    # ------------------------------------------------------------
    v_base_vec = np.full_like(x_grid, fill_value=cfg.base_v, dtype=np.float32)

    # ------------------------------------------------------------
    # (2) target_base sinh từ base_v constant
    # ------------------------------------------------------------
    target_base = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)

    # ------------------------------------------------------------
    # (3) Random circular bump shape
    #     - amplitude random
    #     - width random
    #     - center random sao cho bump nằm trọn trong domain
    # ------------------------------------------------------------
    amp = float(rng.uniform(cfg.circ_amp_min, cfg.circ_amp_max))
    width = float(rng.uniform(cfg.circ_width_min, cfg.circ_width_max))

    # chặn width không vượt domain
    width = min(width, cfg.x_max)

    half_w = 0.5 * width
    center = float(rng.uniform(half_w, cfg.x_max - half_w)) if width < cfg.x_max else float(0.5 * cfg.x_max)

    target_modulation = _circular_bump(
        x_grid=x_grid,
        center=center,
        width=width,
        amp=amp,
    ).astype(np.float32)

    # ------------------------------------------------------------
    # (4) target_modified = target_base + circular bump
    # ------------------------------------------------------------
    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    # ------------------------------------------------------------
    # (5) Giữ API return giống 2 mode còn lại
    # ------------------------------------------------------------
    g_base = np.zeros_like(x_grid, dtype=np.float32)
    g_mod = np.zeros_like(x_grid, dtype=np.float32)

    return v_base_vec, target_base, target, target_modulation, g_base, g_mod

def _sample_triangle_based_target_base_and_target(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # (1) base velocity constant
    v_base_vec = np.full_like(x_grid, fill_value=cfg.base_v, dtype=np.float32)

    # (2) base target
    target_base = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)

    # (3) random triangle params
    amp = float(rng.uniform(cfg.tri_amp_min, cfg.tri_amp_max))
    width = float(rng.uniform(cfg.tri_width_min, cfg.tri_width_max))
    width = min(width, cfg.x_max)

    half_w = 0.5 * width
    center = (
        float(rng.uniform(half_w, cfg.x_max - half_w))
        if width < cfg.x_max
        else float(0.5 * cfg.x_max)
    )

    target_modulation = _triangular_bump(
        x_grid=x_grid,
        center=center,
        width=width,
        amp=amp,
    )

    # (4) final target
    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    # (5) giữ API
    g_base = np.zeros_like(x_grid, dtype=np.float32)
    g_mod = np.zeros_like(x_grid, dtype=np.float32)

    return v_base_vec, target_base, target, target_modulation, g_base, g_mod

class ColdSpray1DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: ColdSpray1DConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.x_grid = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.ds = float(cfg.x_max / cfg.n_steps)

        self.v_base_vec = np.full_like(self.x_grid, fill_value=cfg.base_v, dtype=np.float32)
        self.target_base = ((cfg.feed_rate * self.dx) / np.maximum(self.v_base_vec, 1e-12)).astype(np.float32)
        self.base_profile = self.target_base.copy()
        self.target = self.target_base.copy()
        self.g_base = np.zeros_like(self.x_grid, dtype=np.float32)
        self.target_modulation = np.zeros_like(self.x_grid, dtype=np.float32)
        self.g_mod = np.zeros_like(self.x_grid, dtype=np.float32)

        obs_dim = 2 * cfg.n_bins + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.height = np.zeros(cfg.n_bins, dtype=np.float32)
        self.nozzle_x = 0.0
        self.velocity = self.cfg.base_v
        self.step_count = 0
        self._prev_velocity = self.cfg.base_v

    # NEW:
    # Lưu lại toàn bộ "episode target setup" để có thể replay
    # cùng target/base profile cho deterministic=False.
    # ==========================================================
    def _export_episode_setup(self) -> Dict[str, np.ndarray]:
        return {
            "v_base_vec": self.v_base_vec.copy(),
            "target_base": self.target_base.copy(),
            "target": self.target.copy(),
            "target_modulation": self.target_modulation.copy(),
            "g_base": self.g_base.copy(),
            "g_mod": self.g_mod.copy(),
        }

    # ==========================================================
    # NEW:
    # Nạp lại đúng setup target/base profile đã lưu.
    # Dùng để deterministic=True và deterministic=False chạy
    # trên cùng một target.
    # ==========================================================
    def _load_episode_setup(self, setup: Dict[str, np.ndarray]) -> None:
        self.v_base_vec = setup["v_base_vec"].copy().astype(np.float32)
        self.target_base = setup["target_base"].copy().astype(np.float32)
        self.target = setup["target"].copy().astype(np.float32)
        self.target_modulation = setup["target_modulation"].copy().astype(np.float32)
        self.g_base = setup["g_base"].copy().astype(np.float32)
        self.g_mod = setup["g_mod"].copy().astype(np.float32)
        self.base_profile = self.target_base.copy()

    def _x_to_idx(self, x: float) -> int:
        return int(np.clip(np.searchsorted(self.x_grid, x, side="left"), 0, self.cfg.n_bins - 1))

    def _base_velocity_at(self, x: float) -> float:
        return float(self.v_base_vec[self._x_to_idx(x)])

    def _kernel_at(self, x0: float) -> np.ndarray:
        return _kernel_gaussian(self.x_grid, x0, self.cfg.sigma, self.dx)

    def _get_obs(self) -> np.ndarray:
        delta = (self.height - self.target).astype(np.float32)
        x_norm = np.float32(self.nozzle_x / self.cfg.x_max)
        remaining = np.float32(1.0 - x_norm)
        local_base_v = np.float32(self._base_velocity_at(self.nozzle_x))
        return np.concatenate(
            [
                self.height,
                delta,
                np.array([x_norm, remaining, self.velocity, local_base_v], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        # ==========================================================
        # NEW:
        # Nếu có preset_episode_setup thì không random target nữa,
        # mà load lại đúng target/base profile cũ.
        # ==========================================================
        preset_episode_setup = options.get("preset_episode_setup", None)

        if preset_episode_setup is not None:
            self._load_episode_setup(preset_episode_setup)

        else:
            # ==========================================================
            # Logic cũ: random target theo bump_type
            # ==========================================================
            if self.cfg.bump_type == "gaussian":
                (
                    self.v_base_vec,
                    self.target_base,
                    self.target,
                    self.target_modulation,
                    self.g_base,
                    self.g_mod,
                ) = _sample_gp_based_target_base_and_target(
                    rng=self.np_random,
                    x_grid=self.x_grid,
                    cfg=self.cfg,
                    dx=self.dx,
                )

            elif self.cfg.bump_type == "rect":
                (
                    self.v_base_vec,
                    self.target_base,
                    self.target,
                    self.target_modulation,
                    self.g_base,
                    self.g_mod,
                ) = _sample_rect_based_target_base_and_target(
                    rng=self.np_random,
                    x_grid=self.x_grid,
                    cfg=self.cfg,
                    dx=self.dx,
                )

            elif self.cfg.bump_type == "circular":
                (
                    self.v_base_vec,
                    self.target_base,
                    self.target,
                    self.target_modulation,
                    self.g_base,
                    self.g_mod,
                ) = _sample_circular_based_target_base_and_target(
                    rng=self.np_random,
                    x_grid=self.x_grid,
                    cfg=self.cfg,
                    dx=self.dx,
                )

            elif self.cfg.bump_type == "triangle":
                (
                    self.v_base_vec,
                    self.target_base,
                    self.target,
                    self.target_modulation,
                    self.g_base,
                    self.g_mod,
                ) = _sample_triangle_based_target_base_and_target(
                    rng=self.np_random,
                    x_grid=self.x_grid,
                    cfg=self.cfg,
                    dx=self.dx,
                )

            else:
                raise ValueError(
                    f"Unsupported bump_type='{self.cfg.bump_type}'. "
                    f"Expected one of ['gaussian', 'rect', 'circular', 'triangle']."
                )

            self.base_profile = self.target_base.copy()

        self.height[:] = 0.0
        self.nozzle_x = 0.0
        self.velocity = self._base_velocity_at(0.0)
        self.step_count = 0
        self._prev_velocity = self.velocity
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        self.step_count += 1

        local_base_v = self._base_velocity_at(self.nozzle_x)
        dv = float(np.clip(action[0], -1.0, 1.0)) * cfg.delta_v_max
        v = float(np.clip(local_base_v + dv, cfg.v_min, cfg.v_max))
        self.velocity = v

        x_prev = self.nozzle_x
        ds_eff = float(min(self.ds, cfg.x_max - self.nozzle_x))
        self.nozzle_x = float(min(cfg.x_max, self.nozzle_x + ds_eff))

        dt_step = ds_eff / v if ds_eff > 0.0 else 0.0
        m_step = cfg.feed_rate * dt_step

        L = ds_eff
        n_q = max(2, int(np.ceil(L / self.dx))) if L > 0.0 else 2
        k_acc = np.zeros_like(self.height)
        for j in range(n_q):
            s_j = (j + 0.5) * L / n_q
            x_j = x_prev + s_j
            k_acc += self._kernel_at(x_j)
        k_seg = k_acc / float(n_q)
        delta_h = (m_step * k_seg).astype(np.float32)

        h_old = self.height.copy()
        h_new = h_old + delta_h
        self.height = h_new

        def_old = np.maximum(self.target - h_old, 0.0)
        def_new = np.maximum(self.target - h_new, 0.0)
        ov_old = np.maximum(h_old - self.target, 0.0)
        ov_new = np.maximum(h_new - self.target, 0.0)

        loss_old = float(np.sum(def_old) + cfg.overshoot_weight * np.sum(ov_old))
        loss_new = float(np.sum(def_new) + cfg.overshoot_weight * np.sum(ov_new))
        reward = (loss_old - loss_new) * cfg.reward_scale

        reward -= cfg.action_penalty * (dv / cfg.delta_v_max) ** 2
        dv_smooth = (v - self._prev_velocity) / (cfg.v_max + 1e-12)
        reward -= cfg.smoothness_penalty * (dv_smooth * dv_smooth)
        self._prev_velocity = v

        err = self.height - self.target
        mse = float(np.mean(err * err))
        ov_max = float(np.max(np.maximum(err, 0.0)))

        terminated = False
        truncated = False
        term_reason: Optional[str] = None
        if self.step_count >= cfg.n_steps:
            truncated = True

        # ==========================================================
        # Mình giữ nguyên structure info.
        # Chỉ thêm bump_type để debug biết episode này đang ở mode nào.
        # Với rect:
        # - g_base_mean/std và g_mod_mean/std sẽ ~0 vì không dùng GP
        # ==========================================================
        info: Dict[str, Any] = {
            "bump_type": str(self.cfg.bump_type),
            "nozzle_x": float(self.nozzle_x),
            "velocity": float(self.velocity),
            "local_base_velocity": float(local_base_v),
            "dv": float(dv),
            "ds_eff": float(ds_eff),
            "dt_step": float(dt_step),
            "step_mass": float(m_step),
            "g_base_mean": float(np.mean(self.g_base)),
            "g_base_std": float(np.std(self.g_base)),
            "g_mod_mean": float(np.mean(self.g_mod)),
            "g_mod_std": float(np.std(self.g_mod)),
            "target_mod_mean": float(np.mean(self.target_modulation)),
            "target_mod_std": float(np.std(self.target_modulation)),
            "v_base_gp_amplitude_a": float(self.cfg.v_base_gp_amplitude),
            "target_mod_gp_amplitude_b": float(self.cfg.target_mod_gp_amplitude),
            "gp_std": float(self.cfg.gp_std),
            "gp_lengthscale": float(self.cfg.gp_lengthscale),
            "mse": float(mse),
            "max_overshoot": float(ov_max),
            "termination_reason": term_reason,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def make_venv(cfg: ColdSpray1DConfig, vecnormalize_path: str) -> VecNormalize:
    def make_env():
        return Monitor(ColdSpray1DEnv(cfg))

    base_venv = DummyVecEnv([make_env for _ in range(cfg.n_envs)])
    venv = VecNormalize.load(vecnormalize_path, base_venv)
    venv.training = False
    venv.norm_reward = False
    return venv

def rollout_model(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    deterministic: bool,
    max_steps: Optional[int] = None,
    preset_episode_setup: Optional[Dict[str, np.ndarray]] = None,
):
    env = ColdSpray1DEnv(cfg)

    # ==========================================================
    # NEW:
    # Nếu có preset_episode_setup thì reset bằng setup đó
    # để 2 rollout dùng cùng target.
    # ==========================================================
    if preset_episode_setup is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(options={"preset_episode_setup": preset_episode_setup})

    n = cfg.n_bins
    T = cfg.n_steps if max_steps is None else max_steps

    heights: List[np.ndarray] = [obs[:n].copy()]
    xs: List[float] = [0.0]
    vs: List[float] = [float(obs[2 * n + 2])]
    mses: List[float] = []
    ov_maxs: List[float] = []
    losses: List[float] = []
    rewards: List[float] = []
    reasons: List[str] = [""]

    for _ in range(T):
        obs_norm = venv.normalize_obs(obs[None, :].astype(np.float32))
        act, _ = model.predict(obs_norm, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(act[0])

        heights.append(obs[:n].copy())
        xs.append(float(info["nozzle_x"]))
        vs.append(float(info["velocity"]))
        mses.append(float(info["mse"]))
        ov_maxs.append(float(info["max_overshoot"]))
        losses.append(float(info["mse"]))
        rewards.append(float(r))
        reasons.append(str(info.get("termination_reason", "") or ""))

        if terminated or truncated:
            break

    return {
        "episode_setup": env._export_episode_setup(),   # NEW
        "x_grid": env.x_grid.copy(),
        "target": env.target.copy(),
        "target_base": env.target_base.copy(),
        "v_base_vec": env.v_base_vec.copy(),
        "base_profile": env.base_profile.copy(),
        "g_base": env.g_base.copy(),
        "g_mod": env.g_mod.copy(),
        "target_modulation": env.target_modulation.copy(),
        "heights": np.stack(heights, axis=0),
        "xs": np.array(xs, dtype=np.float32),
        "vs": np.array(vs, dtype=np.float32),
        "mses": np.array(mses, dtype=np.float32),
        "ov_maxs": np.array(ov_maxs, dtype=np.float32),
        "losses": np.array(losses, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "reasons": reasons,
    }


def compute_run_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    final_height = data["heights"][-1]
    target = data["target"]
    err = final_height - target

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    max_overshoot = float(np.max(np.maximum(err, 0.0)))
    max_deficit = float(np.max(np.maximum(-err, 0.0)))
    total_reward = float(np.sum(data["rewards"])) if len(data["rewards"]) else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "max_overshoot": max_overshoot,
        "max_deficit": max_deficit,
        "total_reward": total_reward,
    }

def save_run_plot(
    data_det: Dict[str, Any],
    metrics_det: Dict[str, float],
    data_stoch: Dict[str, Any],
    metrics_stoch: Dict[str, float],
    save_path: Path,
    title: str,
) -> None:
    x = data_det["x_grid"]

    target = data_det["target"]
    target_base = data_det.get("target_base", None)

    final_height_det = data_det["heights"][-1]
    final_height_stoch = data_stoch["heights"][-1]

    # ==========================================================
    # NEW:
    # 1 hàng, 4 cột
    # [0] deterministic=True
    # [1] deterministic=False
    # [2] Velocity
    # [3] Rewards
    # ==========================================================
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    # ----------------------------------------------------------
    # [0] deterministic=True
    # ----------------------------------------------------------
    axes[0].plot(x, target, label="target_modified", color="orange", linewidth=2)
    if target_base is not None:
        axes[0].plot(x, target_base, label="target_base", color="green", alpha=0.8)
    axes[0].plot(x, final_height_det, label="final_height", color="blue", linewidth=2)
    axes[0].set_title(f"{title} | deterministic=True")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("height")
    axes[0].legend()

    # ----------------------------------------------------------
    # [1] deterministic=False
    # ----------------------------------------------------------
    axes[1].plot(x, target, label="target_modified", color="orange", linewidth=2)
    if target_base is not None:
        axes[1].plot(x, target_base, label="target_base", color="green", alpha=0.8)
    axes[1].plot(x, final_height_stoch, label="final_height", color="blue", linewidth=2)
    axes[1].set_title(
        f"deterministic=False\nMSE={metrics_stoch['mse']:.6f}, MAE={metrics_stoch['mae']:.6f}, Ov={metrics_stoch['max_overshoot']:.6f}"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("height")
    axes[1].legend()

    # ----------------------------------------------------------
    # [2] Velocity của deterministic=True
    # ----------------------------------------------------------
    t_v = np.arange(len(data_det["vs"]))
    axes[2].plot(t_v, data_det["vs"])
    axes[2].set_title("Velocity")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("v")

    # ----------------------------------------------------------
    # [3] Rewards của deterministic=True
    # ----------------------------------------------------------
    t_r = np.arange(len(data_det["rewards"]))
    axes[3].plot(t_r, data_det["rewards"])
    axes[3].set_title(
        f"Rewards\nMSE={metrics_det['mse']:.6f}, MAE={metrics_det['mae']:.6f}, Ov={metrics_det['max_overshoot']:.6f}"
    )
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("reward")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _draw_run_plot_on_axes(axes, data: Dict[str, Any], metrics: Dict[str, float], title: str) -> None:
    # ==========================================================
    # NEW helper:
    # Tách phần vẽ 1 run ra thành helper riêng để có thể tái sử dụng
    # cho cả plot đơn lẻ (hàm cũ) và plot so sánh True/False.
    # Logic visualize cũ được giữ nguyên, chỉ chuyển sang helper.
    # ==========================================================
    x = data["x_grid"]
    target = data["target"]
    target_base = data.get("target_base", None)
    final_height = data["heights"][-1]

    axes[0].plot(x, target, label="target_modified", color="orange", linewidth=2)
    if target_base is not None:
        axes[0].plot(x, target_base, label="target_base", color="green", alpha=0.8)
    axes[0].plot(x, final_height, label="final_height", color="blue", linewidth=2)
    axes[0].set_title(title)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("height")
    axes[0].legend()

    t_v = np.arange(len(data["vs"]))
    axes[1].plot(t_v, data["vs"])
    axes[1].set_title("Velocity")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("v")

    t_r = np.arange(len(data["rewards"]))
    axes[2].plot(t_r, data["rewards"])
    axes[2].set_title(
        f"Rewards\nMSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, Ov={metrics['max_overshoot']:.6f}"
    )
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("reward")


def save_compare_plot_true_false(
    data_true: Dict[str, Any],
    metrics_true: Dict[str, float],
    data_false: Dict[str, Any],
    metrics_false: Dict[str, float],
    save_path: Path,
    title_prefix: str,
) -> None:
    # ==========================================================
    # NEW:
    # Tạo 1 hình SO SÁNH đặt deterministic=False bên cạnh deterministic=True
    # đúng theo yêu cầu của bạn.
    #
    # Bố cục: 1 hàng, 6 subplot
    # - 3 subplot bên trái: deterministic=True
    # - 3 subplot bên phải: deterministic=False
    # ==========================================================
    fig, axes = plt.subplots(1, 6, figsize=(32, 4))

    _draw_run_plot_on_axes(
        axes[0:3],
        data_true,
        metrics_true,
        title=f"{title_prefix} | deterministic=True",
    )
    _draw_run_plot_on_axes(
        axes[3:6],
        data_false,
        metrics_false,
        title=f"{title_prefix} | deterministic=False",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_compare_deterministic_pairs(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    n_compare_runs: int,
    output_dir: Path,
) -> List[Dict[str, float]]:
    # ==========================================================
    # NEW:
    # Sinh N hình compare. Mỗi hình dùng cùng reset_seed cho cả 2 mode
    # deterministic=True và deterministic=False để target giống hệt nhau.
    # ==========================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_metrics: List[Dict[str, float]] = []
    for run_idx in range(n_compare_runs):
        reset_seed = 100000 + run_idx

        data_true = rollout_model(
            model=model,
            venv=venv,
            cfg=cfg,
            deterministic=True,
            reset_seed=reset_seed,
        )
        metrics_true = compute_run_metrics(data_true)

        data_false = rollout_model(
            model=model,
            venv=venv,
            cfg=cfg,
            deterministic=False,
            reset_seed=reset_seed,
        )
        metrics_false = compute_run_metrics(data_false)

        save_compare_plot_true_false(
            data_true=data_true,
            metrics_true=metrics_true,
            data_false=data_false,
            metrics_false=metrics_false,
            save_path=output_dir / f"compare_run_{run_idx:03d}.png",
            title_prefix=f"Run {run_idx:03d}",
        )

        compare_metrics.append({
            "run_idx": run_idx,
            "reset_seed": reset_seed,
            "true_mse": float(metrics_true["mse"]),
            "true_mae": float(metrics_true["mae"]),
            "true_max_overshoot": float(metrics_true["max_overshoot"]),
            "true_total_reward": float(metrics_true["total_reward"]),
            "false_mse": float(metrics_false["mse"]),
            "false_mae": float(metrics_false["mae"]),
            "false_max_overshoot": float(metrics_false["max_overshoot"]),
            "false_total_reward": float(metrics_false["total_reward"]),
        })

    return compare_metrics

def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["mse", "mae", "max_overshoot", "max_deficit", "total_reward"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([m[key] for m in metrics_list], dtype=np.float64)
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
        out[f"min_{key}"] = float(np.min(vals))
        out[f"max_{key}"] = float(np.max(vals))
    return out


def load_trained_model(cfg: ColdSpray1DConfig, model_path: str, vecnormalize_path: str):
    venv = make_venv(cfg, vecnormalize_path)
    model = PPO.load(model_path, env=venv)
    return model, venv

def run_many_tests(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    n_runs: int,
    deterministic: bool,
    output_dir: Path,
) -> List[Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    for run_idx in range(n_runs):
        # ------------------------------------------------------
        # Rollout chính: theo deterministic truyền vào
        # ------------------------------------------------------
        data_main = rollout_model(model, venv, cfg, deterministic=True)
        metrics_main = compute_run_metrics(data_main)
        metrics_main["run_idx"] = run_idx
        all_metrics.append(metrics_main)

        # ------------------------------------------------------
        # NEW:
        # Rollout phụ: deterministic=False
        # nhưng dùng cùng episode_setup để cùng target
        # ------------------------------------------------------
        shared_setup = data_main["episode_setup"]

        data_stoch = rollout_model(
            model=model,
            venv=venv,
            cfg=cfg,
            deterministic=False,
            preset_episode_setup=shared_setup,
        )
        metrics_stoch = compute_run_metrics(data_stoch)

        save_run_plot(
            data_det=data_main,
            metrics_det=metrics_main,
            data_stoch=data_stoch,
            metrics_stoch=metrics_stoch,
            save_path=output_dir / f"run_{run_idx:03d}.png",
            title=f"Run {run_idx:03d}",
        )

    return all_metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Load trained PPO model and test on random targets from the same training distribution.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .zip model")
    parser.add_argument("--vecnormalize-path", type=str, required=True, help="Path to VecNormalize .pkl")
    parser.add_argument("--n-runs", type=int, default=20, help="Number of random test episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--output-dir", type=str, default="./test_outputs", help="Directory to save plots and summary")

    class Args:
        model_path = "/content/drive/MyDrive/thesis/checkpoint/best_model_9.zip"
        vecnormalize_path = "/content/drive/MyDrive/thesis/checkpoint/best_model_9_vecnormalize.pkl"
        n_runs = 50
        deterministic = True
        output_dir = "test_outputs"

        # ==========================================================
        # NEW:
        # compare_both_deterministic_modes=True sẽ tạo thêm 4 hình compare:
        # mỗi hình đặt deterministic=False bên cạnh deterministic=True.
        # Bạn có thể đổi n_compare_runs nếu muốn nhiều / ít hơn 4 hình.
        # ==========================================================
        compare_both_deterministic_modes = True
        n_compare_runs = 4

    args = Args()

    cfg = ColdSpray1DConfig()
    output_dir = Path(args.output_dir)

    model, venv = load_trained_model(cfg, args.model_path, args.vecnormalize_path)
    metrics_list = run_many_tests(
        model=model,
        venv=venv,
        cfg=cfg,
        n_runs=args.n_runs,
        deterministic=args.deterministic,
        output_dir=output_dir,
    )

    summary = summarize_metrics(metrics_list)

    with open(output_dir / "metrics_per_run.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TEST SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.8f}")
    print(f"\nSaved outputs to: {output_dir.resolve()}")

    zip_path = output_dir.with_suffix(".zip")

    # tạo file zip
    shutil.make_archive(base_name=str(output_dir), format='zip', root_dir=output_dir)

    print(f"Zipped results to: {zip_path}")

    # download về máy
    files.download(str(zip_path))


if __name__ == "__main__":
    main()
