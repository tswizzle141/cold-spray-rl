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
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import *

SAVE_DIR = Path("/netscratch/nham/logs/pics-ppo-2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("/netscratch/nham/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_RNG = np.random.default_rng()

def sample_env_seed() -> int:
    return int(GLOBAL_RNG.integers(0, int(1e7)))

class ColdSpray2DMultiInputExtractor(BaseFeaturesExtractor):
    """
    ======== CHANGED: extractor mới cho MultiInputPolicy ========

    Ý tưởng:
    - nhánh "map": dùng CNN để đọc cấu trúc không gian 2D
        + channel 0 = height
        + channel 1 = height - target
    - nhánh "state": dùng MLP nhỏ để đọc 5 scalar global
        + progress, remaining, velocity, v_base_i, offset

    Sau đó concatenate 2 nhánh lại thành 1 feature vector chung cho actor/critic.
    Đây chính là điểm khác biệt cốt lõi so với MlpPolicy cũ.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,
        state_output_dim: int = 32,) -> None:
        
        super().__init__(observation_space, features_dim=cnn_output_dim + state_output_dim)

        map_space = observation_space["map"]
        state_space = observation_space["state"]

        if not isinstance(map_space, spaces.Box):
            raise TypeError("observation_space['map'] must be a Box")
        if not isinstance(state_space, spaces.Box):
            raise TypeError("observation_space['state'] must be a Box")

        n_map_channels = int(map_space.shape[0])
        n_state = int(state_space.shape[0])

        # ======== CHANGED: CNN branch cho map 2D ========
        # Grid hiện tại là 16x16 nên conv stride=1 + AdaptiveAvgPool2d(1) là an toàn,
        # không phụ thuộc chặt vào đúng 16x16.
        self.map_cnn = nn.Sequential(
            nn.Conv2d(n_map_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Suy ra số chiều thật của CNN branch bằng 1 forward giả.
        with torch.no_grad():
            sample_map = torch.as_tensor(map_space.sample()[None]).float()
            n_flatten = int(self.map_cnn(sample_map).shape[1])

        self.map_head = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU(),
        )

        # ======== CHANGED: MLP branch cho scalar/global state ========
        self.state_mlp = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, state_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_tensor = observations["map"].float()
        state_tensor = observations["state"].float()

        map_features = self.map_head(self.map_cnn(map_tensor))
        state_features = self.state_mlp(state_tensor)

        # ======== CHANGED: ghép CNN features + scalar features ========
        return torch.cat([map_features, state_features], dim=1)

@dataclass
class ColdSpray2DConfig:
    # -------------------------------------------------------------------------
    # 1) Grid / domain
    # -------------------------------------------------------------------------
    grid_n: int = 16
    x_max: float = 1.0
    y_max: float = 1.0
    # Snake path settings copied in spirit from trial_target_2d.py
    spacing_px: int = 1
    ds: float = 0.05  # arc-length per RL step along the snake centerline

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
    v_max: float = 1.5
    v_min: float = 1e-2
    base_v: float = 0.5
    delta_v_max: float = 1.0

    # IMPORTANT CHANGE:
    # There is NO offset_base_i anymore.
    # Only the RL action can shift the nozzle laterally.
    off_max: float = 0.05
    offset_limit: float = 0.10

    # -------------------------------------------------------------------------
    # 4) Target construction with 2D GPs
    # -------------------------------------------------------------------------
    # ======== CHANGED: target_base không còn là mặt phẳng nữa.
    # Mỗi episode sẽ sample:
    #   - 1 mean height trong [0.05, 0.10]
    #   - 1 zero-mean GP 2D cho target_base
    #   - 1 zero-mean GP 2D khác cho target_modulation (noise)
    # rồi build:
    #   target_base = mean_height * (1 + rel_amp_base * g_base)
    #   target      = target_base + rel_amp_noise * target_base * g_mod
    #
    # Nhờ vậy:
    #   - mean(target_base) thay đổi giữa 0.05 và 0.10 theo từng episode
    #   - noise GP có mean đúng bằng 0
    target_base_mean_height_min: float = 0.05
    target_base_mean_height_max: float = 0.10
    target_base_relative_gp_amplitude: float = 0.10
    target_mod_relative_amplitude_min: float = 0.05
    target_mod_relative_amplitude_max: float = 0.10
    gp_std: float = 1.0
    gp_lengthscale_x: float = 0.20
    gp_lengthscale_y: float = 0.20
    gp_jitter: float = 1e-8
    # ======== CHANGED: mode switch giống file 1D ========
    # True  -> mỗi reset()/episode sẽ sample target GP mới
    # False -> dùng đúng 1 shared target GP + shared noise GP cho tất cả episode
    gp_resample_every_reset: bool = True
    n_envs: int = 2

    # ======== CHANGED: thêm đúng ý tưởng file 1D ========
    # GP g_base 2D bây giờ KHÔNG sinh target_base trực tiếp nữa.
    # Thay vào đó:
    #   1) sample g_base 2D
    #   2) đọc g_base local tại từng snake segment
    #   3) suy ra v_base_i = base_v + v_base_gp_amplitude * g_base_i
    #   4) dựng target_base bằng chính mô hình deposition khi chạy với v_base_i đó
    #
    # Nghĩa là pipeline mới là:
    #   g_base(2D) -> v_base_i (theo segment) -> target_base(2D)
    #
    # Đây là phiên bản 2D gần nhất với ý tưởng file 1D.
    v_base_gp_amplitude: float = 0.25

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
    ppo_target_kl: float = 0.1
    ppo_n_epochs: int = 4
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_log_std_init: float = -5.0
    total_timesteps: int = 500000000

    resume: bool = True
    resume_model_path: str = str(CHECKPOINT_DIR / "best_model_2.zip")
    resume_vecnormalize_path: str = str(CHECKPOINT_DIR / "best_model_2_vecnormalize.pkl")
    save_freq: int = 50000


@dataclass
class GPTarget2DBundle:
    """
    Bundle target 2D hoàn chỉnh để có thể:
    - resample mỗi episode
    - hoặc giữ cố định cho mọi episode khi debug policy
    """
    v_base_vec: np.ndarray
    target_base: np.ndarray
    target: np.ndarray
    g_base: np.ndarray
    g_mod: np.ndarray
    target_modulation: np.ndarray
    sampled_target_base_mean_height: float
    sampled_target_mod_relative_amplitude: float


# =============================================================================
# Small utilities
# =============================================================================

def _save_fig(fig, filename: str, dpi: int = 160) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ======== CHANGED: 2D GP helpers ========
def _rbf_covariance_1d(x: np.ndarray, std: float, lengthscale: float, jitter: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    diff = x[:, None] - x[None, :]
    K = (std ** 2) * np.exp(-0.5 * (diff / max(lengthscale, 1e-12)) ** 2)
    K += jitter * np.eye(len(x), dtype=np.float64)
    return K

def _sample_zero_mean_gp_2d_separable(
    rng: np.random.Generator,
    grid_n: int,
    std: float,
    lengthscale_x: float,
    lengthscale_y: float,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    ======== CHANGED: sample GP 2D zero-mean bằng covariance tách được ========
    Nếu Z ~ N(0, I), G = Ly @ Z @ Lx^T thì vec(G) có covariance K_y ⊗ K_x.
    Cách này rẻ hơn rất nhiều so với build ma trận covariance 2D khổng lồ.
    """
    grid = (np.arange(grid_n, dtype=np.float64) + 0.5) / grid_n
    Kx = _rbf_covariance_1d(grid, std=std, lengthscale=lengthscale_x, jitter=jitter)
    Ky = _rbf_covariance_1d(grid, std=std, lengthscale=lengthscale_y, jitter=jitter)
    Lx = np.linalg.cholesky(Kx)
    Ly = np.linalg.cholesky(Ky)

    Z = rng.standard_normal((grid_n, grid_n))
    G = Ly @ Z @ Lx.T
    G = G - np.mean(G)
    return G.astype(np.float32)


class DepositionModel2D:
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

class SnakePath2D:
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


def _build_target_base_from_vbase_2d(
    cfg: ColdSpray2DConfig,
    steps: List[dict],
    v_base_vec: np.ndarray,
) -> np.ndarray:
    """
    ======== CHANGED: dựng target_base từ chính v_base_i, giống tinh thần file 1D ========

    File 1D làm theo chuỗi:
        g_base -> v_base(x) -> target_base(x)

    Ở bản 2D mới, ta làm tương đương nhưng phải qua snake segments:
        g_base(2D)
        -> sample local g_base_i tại từng segment
        -> v_base_i cho từng segment
        -> chạy mô hình deposition với v_base_i đó
        -> tích lũy toàn bộ để được target_base(2D)

    Nhờ vậy target_base bây giờ thực sự là "kết quả lắng đọng baseline"
    sinh ra bởi baseline velocity profile, thay vì là một ảnh GP được vẽ trực tiếp.
    """
    deposition = DepositionModel2D(
        grid_n=cfg.grid_n,
        sigma=cfg.sigma,
        beta=cfg.beta,
        eta=cfg.eta,
        rho_m=cfg.rho_m,
        samples_per_pixel=cfg.samples_per_pixel,
        A_amp=cfg.A_amp,
    )

    target_base = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)

    for step_idx, st in enumerate(steps):
        v_i = float(v_base_vec[int(np.clip(step_idx, 0, len(v_base_vec) - 1))])

        x0 = float(st["x0"])
        y0 = float(st["y0"])
        x1 = float(st["x1"])
        y1 = float(st["y1"])

        dt_step = float(st["length"]) / max(v_i, 1e-12)
        m_step = float(cfg.feed_rate) * dt_step

        basis = deposition.basis_for_segment(x0, y0, x1, y1)
        target_base += (m_step * basis).astype(np.float32)

    return target_base.astype(np.float32)


def build_gp_target_2d_bundle(
    rng: np.random.Generator,
    cfg: ColdSpray2DConfig,
    steps: List[dict],
) -> GPTarget2DBundle:
    """
    ======== CHANGED: build target 2D theo đúng tinh thần file 1D ========

    Trước đây bản 2D cũ làm:
        g_base(2D) -> target_base(2D) trực tiếp
        v_base_vec = constant = base_v

    Bây giờ sửa lại thành:
        1) sample GP g_base 2D
        2) tại mỗi snake segment, đọc ra local g_base_i bằng bilinear interpolation
        3) suy ra v_base_i = base_v + v_base_gp_amplitude * g_base_i
        4) clip về [v_min, v_max]
        5) chạy mô hình deposition với chính v_base_i đó để dựng target_base(2D)
        6) sau đó mới cộng thêm GP noise để tạo target cuối

    Đây là phiên bản 2D gần nhất với file 1D:
        g_base -> v_base_i -> target_base -> target

    Lưu ý:
    - sampled_target_base_mean_height bây giờ KHÔNG còn là một số random độc lập
      dùng để "vẽ" target_base nữa.
    - Nó được ghi lại là mean(target_base) THỰC TẾ sau khi target_base đã được
      dựng từ baseline deposition.
    """
    n_steps = len(steps)

    # ======== CHANGED: sample GP 2D nền ========
    g_base = _sample_zero_mean_gp_2d_separable(
        rng=rng,
        grid_n=cfg.grid_n,
        std=cfg.gp_std,
        lengthscale_x=cfg.gp_lengthscale_x,
        lengthscale_y=cfg.gp_lengthscale_y,
        jitter=cfg.gp_jitter,
    )

    # ======== CHANGED: từ g_base(2D) -> local g_base_i theo từng snake segment ========
    # Ta lấy giá trị tại trung điểm của segment.
    # Đây là scalar đại diện cho mức "baseline GP" local của segment i.
    local_g_base_vec = np.zeros(n_steps, dtype=np.float32)
    for step_idx, st in enumerate(steps):
        x_mid = 0.5 * (float(st["x0"]) + float(st["x1"]))
        y_mid = 0.5 * (float(st["y0"]) + float(st["y1"]))
        local_g_base_vec[step_idx] = np.float32(
            _sample_map_at_xy_bilinear(g_base, x_mid, y_mid)
        )

    # ======== CHANGED: đúng ý tưởng file 1D ========
    # v_base_i = base_v + amplitude * local_g_base_i
    v_base_raw = float(cfg.base_v) + float(cfg.v_base_gp_amplitude) * local_g_base_vec.astype(np.float64)
    v_base_vec = np.clip(v_base_raw, cfg.v_min, cfg.v_max).astype(np.float32)

    # ======== CHANGED: target_base bây giờ được DỰNG từ chính baseline velocity profile ========
    target_base = _build_target_base_from_vbase_2d(
        cfg=cfg,
        steps=steps,
        v_base_vec=v_base_vec,
    )
    target_base = np.maximum(target_base, 0.0).astype(np.float32)

    # ======== CHANGED: chỉ để log/visualize mean base height thực tế ========
    sampled_target_base_mean_height = float(np.mean(target_base.astype(np.float64)))

    # ======== GIỮ NGUYÊN Ý TƯỞNG noise GP như trước ========
    g_mod = _sample_zero_mean_gp_2d_separable(
        rng=rng,
        grid_n=cfg.grid_n,
        std=cfg.gp_std,
        lengthscale_x=cfg.gp_lengthscale_x,
        lengthscale_y=cfg.gp_lengthscale_y,
        jitter=cfg.gp_jitter,
    )

    sampled_target_mod_relative_amplitude = float(
        rng.uniform(
            cfg.target_mod_relative_amplitude_min,
            cfg.target_mod_relative_amplitude_max,
        )
    )

    target_modulation = (
        sampled_target_mod_relative_amplitude
        * target_base.astype(np.float64)
        * g_mod.astype(np.float64)
    ).astype(np.float32)
    target_modulation = (target_modulation - np.mean(target_modulation)).astype(np.float32)

    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    return GPTarget2DBundle(
        v_base_vec=v_base_vec.copy(),
        target_base=target_base.copy(),
        target=target.copy(),
        g_base=g_base.copy(),
        g_mod=g_mod.copy(),
        target_modulation=target_modulation.copy(),
        sampled_target_base_mean_height=float(sampled_target_base_mean_height),
        sampled_target_mod_relative_amplitude=float(sampled_target_mod_relative_amplitude),
    )


def build_shared_gp_target_2d_bundle(cfg: ColdSpray2DConfig, steps: List[dict], seed: int = 0) -> GPTarget2DBundle:
    """
    ======== CHANGED: tạo đúng 1 shared 2D target bundle ========
    Chỉ dùng khi gp_resample_every_reset == False.
    """
    rng = np.random.default_rng(seed)
    return build_gp_target_2d_bundle(rng=rng, cfg=cfg, steps=steps)


def compute_relative_mse_mean_target(
    final_height: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    ======== CHANGED: relative MSE giống file 1D ========
    Robust metric dùng để save best weight:
    - chuẩn hóa MSE theo mean(target)^2
    - nhờ vậy so sánh công bằng hơn giữa các episode có mean target khác nhau
    """
    err = final_height.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(err ** 2))
    mean_target = float(np.mean(target.astype(np.float64)))
    denom = max(mean_target * mean_target, eps)
    return float(mse / denom)


# =============================================================================
# 2D RL environment
# =============================================================================

class ColdSpray2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: ColdSpray2DConfig,
        render_mode: Optional[str] = None,
        shared_target_bundle: Optional[GPTarget2DBundle] = None,
        env_seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.shared_target_bundle = shared_target_bundle
        self._rng = np.random.default_rng(env_seed)

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

        if (not self.cfg.gp_resample_every_reset) and self.shared_target_bundle is None:
            raise ValueError(
                "gp_resample_every_reset=False requires shared_target_bundle. "
                "You must create one shared target bundle for all episodes."
            )

        self.height = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.target = np.zeros_like(self.height)
        self.target_base = np.zeros_like(self.height)
        self.target_modulation = np.zeros_like(self.height)

        self.v_base_vec = np.full(self.n_steps, fill_value=cfg.base_v, dtype=np.float32)
        # ======== CHANGED: g_base / g_mod là GP 2D thật của target ========
        self.g_base = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.g_mod = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.sampled_target_base_mean_height = 0.0
        self.sampled_target_mod_relative_amplitude = 0.0
        self.episode_index = 0

        self.step_idx = 0
        self.velocity = float(cfg.base_v)
        self.offset = 0.0
        self._prev_velocity = float(cfg.base_v)
        self._prev_offset = 0.0

        # ======== CHANGED: set target bundle ngay từ lúc khởi tạo env ========
        self._set_target_bundle_for_current_episode(initial=True)

        # ======== CHANGED: đổi observation từ vector phẳng sang spaces.Dict ========
        # Trước đây MlpPolicy nhận 1 vector 1D đã flatten toàn bộ map.
        # Bây giờ ta tách rõ:
        #   - "map"  : tensor (2, H, W) cho CNN
        #       + channel 0 = height
        #       + channel 1 = height - target
        #   - "state": vector 5 chiều cho MLP nhỏ
        #       + progress, remaining, velocity, v_base_i, offset
        # Đây là format tự nhiên nhất để dùng MultiInputPolicy + CNN extractor.
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2, cfg.grid_n, cfg.grid_n),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        # Action = [dv_action, abweichung_action]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _set_target_bundle_for_current_episode(self, initial: bool = False) -> None:
        """
        ======== CHANGED: mode switch thật sự ========
        - gp_resample_every_reset=True  -> mỗi episode sample target mới
        - gp_resample_every_reset=False -> reuse đúng shared target bundle
        """
        if self.cfg.gp_resample_every_reset:
            bundle = build_gp_target_2d_bundle(self._rng, self.cfg, self.steps)
        else:
            bundle = self.shared_target_bundle
            if bundle is None:
                raise RuntimeError("shared_target_bundle is required in shared-target mode.")

        self.v_base_vec = bundle.v_base_vec.copy()
        self.target_base = bundle.target_base.copy()
        self.target = bundle.target.copy()
        self.g_base = bundle.g_base.copy()
        self.g_mod = bundle.g_mod.copy()
        self.target_modulation = bundle.target_modulation.copy()
        self.sampled_target_base_mean_height = float(bundle.sampled_target_base_mean_height)
        self.sampled_target_mod_relative_amplitude = float(bundle.sampled_target_mod_relative_amplitude)

        if not initial:
            self.episode_index += 1

    def _current_step(self) -> dict:
        idx = int(np.clip(self.step_idx, 0, self.n_steps - 1))
        return self.steps[idx]

    def _current_base_velocity(self) -> float:
        return _step_local_base_velocity(self.v_base_vec, self.step_idx)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        progress = np.float32(self.step_idx / max(self.n_steps, 1))
        remaining = np.float32(1.0 - progress)
        v_base_i = np.float32(self._current_base_velocity())

        # ======== CHANGED: map branch cho CNN ========
        map_obs = np.stack(
            [
                self.height,
                self.height - self.target,
            ],
            axis=0,
        ).astype(np.float32)

        # ======== CHANGED: scalar branch cho MLP ========
        state_obs = np.array(
            [
                progress,
                remaining,
                np.float32(self.velocity),
                v_base_i,
                np.float32(self.offset),
            ],
            dtype=np.float32,
        )

        return {
            "map": map_obs,
            "state": state_obs,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ======== CHANGED: nếu True thì resample, nếu False thì reuse shared target ========
        self._set_target_bundle_for_current_episode(initial=False)

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
            "gp_resample_every_reset": bool(self.cfg.gp_resample_every_reset),
            "episode_index": int(self.episode_index),
            "g_base_mean": float(np.mean(self.g_base)),
            "g_base_std": float(np.std(self.g_base)),
            "g_mod_mean": float(np.mean(self.g_mod)),
            "g_mod_std": float(np.std(self.g_mod)),
            "target_mod_mean": float(np.mean(self.target_modulation)),
            "target_mod_std": float(np.std(self.target_modulation)),
            # ======== CHANGED: log thêm mean height base + relative noise amp ========
            "sampled_target_base_mean_height": float(self.sampled_target_base_mean_height),
            "sampled_target_mod_relative_amplitude": float(self.sampled_target_mod_relative_amplitude),
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

def make_venv(
    cfg: ColdSpray2DConfig,
    shared_target_bundle: Optional[GPTarget2DBundle],
    vecnormalize_path: Optional[str] = None,
) -> VecNormalize:
    def make_env(rank: int):
        def _factory():
            env_seed = sample_env_seed()
            return Monitor(
                ColdSpray2DEnv(
                    cfg,
                    shared_target_bundle=shared_target_bundle,
                    env_seed=env_seed,
                )
            )
        return _factory

    base_venv = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)]) #DummyVecEnv

    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        venv = VecNormalize.load(vecnormalize_path, base_venv)
        venv.training = True
        venv.norm_reward = True
        print(f"[make_venv] Loaded VecNormalize from: {vecnormalize_path}")
        return venv

    return VecNormalize(base_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def make_ppo(cfg: ColdSpray2DConfig, venv: VecNormalize, seed: int = 0) -> PPO:
    # ======== CHANGED: đổi từ MlpPolicy sang MultiInputPolicy ========
    # Trước đây observation bị flatten hết nên dùng MLP.
    # Bây giờ observation là Dict(map, state), nên phải dùng MultiInputPolicy.
    # map sẽ đi qua CNN extractor, còn state sẽ đi qua MLP nhỏ bên trong extractor.
    return PPO(
        policy="MultiInputPolicy",
        env=venv,
        policy_kwargs={
            "log_std_init": cfg.ppo_log_std_init,
            "features_extractor_class": ColdSpray2DMultiInputExtractor,
            "features_extractor_kwargs": {
                "cnn_output_dim": 128,
                "state_output_dim": 32,
            },
        },
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
        shared_target_bundle: Optional[GPTarget2DBundle],
        check_freq: int = 10000,
        save_prefix: str = "best_model_2",
        deterministic: bool = True,
        rollout_seed: int = 654321,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.venv = venv
        self.cfg = cfg
        self.shared_target_bundle = shared_target_bundle
        self.check_freq = int(check_freq)
        self.save_prefix = save_prefix
        self.deterministic = deterministic
        self.rollout_seed = int(rollout_seed)

        self.best_mse = np.inf
        self.best_step = 0

        # ======== CHANGED: fixed evaluation seeds giống file 1D ========
        # Dùng nhiều seed cố định để đo robust/generalization thay vì chỉ 1 rollout.
        self.eval_seeds = [101, 202, 303, 404, 505, 606, 707, 808]

    # ======== CHANGED: evaluate trên nhiều rollout + relative MSE ========
    def _evaluate_once(self) -> tuple[float, float]:
        values = []

        for seed in self.eval_seeds:
            data = rollout_model(
                self.model,
                self.venv,
                self.cfg,
                self.shared_target_bundle,
                deterministic=self.deterministic,
                env_seed=seed,
            )

            final_height = data["heights"][-1]
            target = data["target"]

            relative_mse = compute_relative_mse_mean_target(
                final_height=final_height,
                target=target,
            )
            values.append(float(relative_mse))

        arr = np.array(values, dtype=np.float64)

        # ======== CHANGED: return mean + std để log độ robust ========
        return float(np.mean(arr)), float(np.std(arr))

    def _save_best(self, relative_mse: float):
        model_path = CHECKPOINT_DIR / self.save_prefix
        vecnorm_path = CHECKPOINT_DIR / f"{self.save_prefix}_vecnormalize.pkl"
        self.model.save(model_path)
        self.venv.save(vecnorm_path)
        if self.verbose > 0:
            print(
                f"[SaveBestMSECallback] New BEST mean relative MSE = {relative_mse:.8f} "
                f"at timestep = {self.num_timesteps}"
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # ======== CHANGED: save best theo mean(final_relative_mse) trên nhiều seed ========
        mean_mse, std_mse = self._evaluate_once()

        if self.verbose > 0:
            print(
                f"[SaveBestMSECallback] Eval at timestep {self.num_timesteps}, "
                f"mean relative MSE = {mean_mse:.8f}, std = {std_mse:.8f}, "
                f"best = {self.best_mse:.8f}"
            )

        if mean_mse < self.best_mse:
            self.best_mse = mean_mse
            self.best_step = self.num_timesteps
            self._save_best(mean_mse)
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
        shared_target_bundle: Optional[GPTarget2DBundle],
        every_episodes: int = 500,
        deterministic: bool = True,
        start_at: int = 0,
        max_plots: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.venv = venv
        self.cfg = cfg
        self.shared_target_bundle = shared_target_bundle
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
                    self.shared_target_bundle,
                    deterministic=self.deterministic,
                )
                stochastic_data = rollout_model(
                    self.model,
                    self.venv,
                    self.cfg,
                    self.shared_target_bundle,
                    deterministic=False,
                )
                data["stochastic_rollout"] = stochastic_data
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


def _add_batch_dim_to_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    ======== CHANGED: helper cho Dict observation ========

    rollout_model() dùng env thường (không vectorized), còn model lại đi kèm VecNormalize.
    Vì thế trước khi gọi venv.normalize_obs(...), ta cần thêm batch dimension
    cho từng key trong Dict observation.
    """
    return {key: value[None, ...].astype(np.float32) for key, value in obs.items()}


def rollout_model(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray2DConfig,
    shared_target_bundle: Optional[GPTarget2DBundle],
    deterministic: bool,
    max_steps: Optional[int] = None,
    env_seed: int = 0,
):
    env = ColdSpray2DEnv(
        cfg,
        shared_target_bundle=shared_target_bundle,
        env_seed=env_seed,
    )
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
        # ======== CHANGED: normalize Dict observation cho MultiInputPolicy ========
        obs_batched = _add_batch_dim_to_obs_dict(obs)
        obs_norm = venv.normalize_obs(obs_batched)
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
        "g_base": env.g_base.copy(),
        "g_mod": env.g_mod.copy(),
        "gp_resample_every_reset": bool(cfg.gp_resample_every_reset),
        # ======== CHANGED: expose sampled GP settings for plots/debug ========
        "sampled_target_base_mean_height": float(env.sampled_target_base_mean_height),
        "sampled_target_mod_relative_amplitude": float(env.sampled_target_mod_relative_amplitude),
        # ======== CHANGED: abs GP scale giống file 1D để hiện lên tem visualize ========
        "abs_gp_scale": float(
            env.sampled_target_mod_relative_amplitude
            * float(np.mean(env.target_base.astype(np.float64)))
        ),
        "v_base_vec": v_base_vec,
        "velocities": np.array(velocities, dtype=np.float32),
        "offsets": np.array(offsets, dtype=np.float32),
        "mses": np.array(mses, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "segments": np.array(segments, dtype=np.float32),
    }


# ======== CHANGED: helper tạo tem info giống file 1D cho deterministic / stochastic ========
def _build_rollout_stamp_text(data: Dict[str, Any]) -> str:
    sampled_rel_amp = float(data.get("sampled_target_mod_relative_amplitude", 0.0))
    abs_gp_scale = float(data.get("abs_gp_scale", 0.0))

    heights = data.get("heights", None)
    target = data.get("target", None)

    relative_mse = np.nan
    if heights is not None and target is not None and len(heights) > 0:
        final_height = heights[-1]
        relative_mse = compute_relative_mse_mean_target(
            final_height=final_height,
            target=target,
        )

    return (
        f"GP amplitude = {100.0 * sampled_rel_amp:.2f}% of mean target_base\n"
        f"abs GP scale = {abs_gp_scale:.6f}\n"
        f"relative MSE = {relative_mse:.6f}"
    )


def plot_rollout_summary(data: Dict[str, Any], save_name: Optional[str] = None) -> str:
    """
    Updated visualization:
    - remove realized 2D path
    - add deterministic/stochastic velocity + offset tracking separately
    - use the same color scale for target_modified and final_height
    """
    target = data["target"]
    target_base = data["target_base"]
    target_modulation = data["target_modulation"]
    height = data["heights"][-1]
    det_velocities = data["velocities"]
    det_offsets = data["offsets"]
    v_base_vec = data["v_base_vec"]
    mses = data["mses"]

    stochastic_data = data.get("stochastic_rollout")
    if stochastic_data is not None:
        sto_velocities = stochastic_data["velocities"]
        sto_offsets = stochastic_data["offsets"]
    else:
        sto_velocities = np.array([], dtype=np.float32)
        sto_offsets = np.array([], dtype=np.float32)

    # ======== CHANGED: build tem text cho deterministic và stochastic giống file 1D ========
    det_stamp_text = _build_rollout_stamp_text(data)
    sto_stamp_text = _build_rollout_stamp_text(stochastic_data) if stochastic_data is not None else det_stamp_text

    # Shared scale so target_modified and final_height are directly comparable.
    shared_vmin = float(min(np.min(target), np.min(height)))
    shared_vmax = float(max(np.max(target), np.max(height)))

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(target_base, origin="lower")
    ax1.set_title("target_base (2D GP)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(target, origin="lower", vmin=shared_vmin, vmax=shared_vmax)
    ax2.set_title("target_modified = target_base + GP noise")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ======== CHANGED: tem nhỏ để biết episode này sample GP nào ========
    sampled_base_mean = float(data.get("sampled_target_base_mean_height", 0.0))
    sampled_noise_rel_amp = float(data.get("sampled_target_mod_relative_amplitude", 0.0))
    rollout_mode = "resample/episode" if bool(data.get("gp_resample_every_reset", False)) else "shared target GP"
    ax2.text(
        0.02,
        0.98,
        f"mode = {rollout_mode}\nbase mean = {sampled_base_mean:.5f}\nnoise amp = {100.0 * sampled_noise_rel_amp:.2f}%",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(height, origin="lower", vmin=shared_vmin, vmax=shared_vmax)
    ax3.set_title(f"final height | final mse={mses[-1]:.3e}" if len(mses) else "final height")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[0, 3])
    t_det_v = np.arange(len(det_velocities))
    t_b = np.arange(len(v_base_vec))
    ax4.plot(t_b, v_base_vec, label="v_base_i")
    ax4.plot(t_det_v, det_velocities, label="deterministic velocity")
    ax4.set_title("velocity tracking (deterministic)")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="best")
    # ======== CHANGED: tem deterministic giống file 1D ========
    ax4.text(
        0.02,
        0.98,
        det_stamp_text,
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax5 = fig.add_subplot(gs[1, 0])
    t_sto_v = np.arange(len(sto_velocities))
    ax5.plot(t_b, v_base_vec, label="v_base_i")
    if len(sto_velocities) > 0:
        ax5.plot(t_sto_v, sto_velocities, label="stochastic velocity")
    ax5.set_title("velocity tracking (stochastic)")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="best")
    # ======== CHANGED: tem stochastic giống file 1D ========
    ax5.text(
        0.02,
        0.98,
        sto_stamp_text,
        transform=ax5.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax6 = fig.add_subplot(gs[1, 1])
    t_det_o = np.arange(len(det_offsets))
    ax6.plot(t_det_o, det_offsets, label="deterministic offset")
    ax6.set_title("offset tracking (deterministic)")
    ax6.grid(alpha=0.3)
    ax6.legend(loc="best")

    ax7 = fig.add_subplot(gs[1, 2])
    t_sto_o = np.arange(len(sto_offsets))
    if len(sto_offsets) > 0:
        ax7.plot(t_sto_o, sto_offsets, label="stochastic offset")
    ax7.set_title("offset tracking (stochastic)")
    ax7.grid(alpha=0.3)
    ax7.legend(loc="best")

    ax8 = fig.add_subplot(gs[1, 3])
    height_diff = height - target
    diff_abs = float(np.max(np.abs(height_diff))) if np.size(height_diff) else 1.0
    if diff_abs < 1e-12:
        diff_abs = 1.0
    im8 = ax8.imshow(height_diff, origin="lower", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm")
    ax8.set_title("final_height - target_modified")
    fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    _ = target_modulation

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_name is None:
        save_name = f"rollout_2d_snake_{stamp}.png"
    return _save_fig(fig, save_name)

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
        v_max=1.5,
        v_min=0.01,
        base_v=0.5,
        delta_v_max=1.0,
        v_base_gp_amplitude=0.05,  # ======== CHANGED: amplitude để GP quyết định baseline velocity theo segment ========
        off_max=0.03,
        offset_limit=0.08,
        target_base_mean_height_min=0.05,
        target_base_mean_height_max=0.10,
        target_base_relative_gp_amplitude=0.05,
        target_mod_relative_amplitude_min=0.05,
        target_mod_relative_amplitude_max=0.10,
        gp_std=1.0,
        gp_lengthscale_x=0.20,
        gp_lengthscale_y=0.20,
        gp_jitter=1e-8,
        gp_resample_every_reset=False,  # ======== CHANGED: False để debug trên đúng 1 target cố định ========
        n_envs=8,
        total_timesteps=500000000,
        resume=True,
    )

    # ======== CHANGED: shared bundle chỉ được tạo khi muốn giữ target cố định toàn bộ episode ========
    shared_target_bundle = None
    if not cfg.gp_resample_every_reset:
        shared_target_bundle = build_shared_gp_target_2d_bundle(cfg=cfg, steps=build_fixed_steps(cfg), seed=0)

    venv = make_venv(
        cfg,
        shared_target_bundle=shared_target_bundle,
        vecnormalize_path=cfg.resume_vecnormalize_path if cfg.resume else None,
    )
    model = load_or_create_model(cfg, venv, seed=0)

    best_callback = SaveBestMSECallback(
        venv=venv,
        cfg=cfg,
        shared_target_bundle=shared_target_bundle,
        check_freq=50000,
        save_prefix="best_model_2",
        deterministic=True,
        rollout_seed=333,  # ======== CHANGED: giữ style giống file 1D ========
        verbose=1,
    )

    visualize_callback = VisualizeEveryNEpisodesCallback(
        venv=venv,
        cfg=cfg,
        shared_target_bundle=shared_target_bundle,
        every_episodes=5000,
        deterministic=True,
        verbose=1,
    )

    callback = CallbackList([best_callback, visualize_callback])
    model = train_ppo(model, total_timesteps=cfg.total_timesteps, callback=callback)
    det = rollout_model(model, venv, cfg, shared_target_bundle, deterministic=True)
    sto = rollout_model(model, venv, cfg, shared_target_bundle, deterministic=False)
    det["stochastic_rollout"] = sto
    out_png = plot_rollout_summary(det, save_name="final_rollout_2d_snake.png")
    print(f"Saved summary plot to: {out_png}")
