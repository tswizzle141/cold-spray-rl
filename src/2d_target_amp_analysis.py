from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from math import gamma
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

SAVE_DIR = Path("/netscratch/nham/logs/pics-ppo-12")
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
    grid_n: int = 32
    x_max: float = 1.0
    y_max: float = 1.0
    # Snake path settings copied in spirit from trial_target_2d.py
    spacing_px: int = 2
    ds: float = 0.05  # arc-length per RL step along the snake centerline

    # -------------------------------------------------------------------------
    # 2) Deposition model
    # -------------------------------------------------------------------------
    sigma: float = 0.05
    beta: float = 1.0
    eta: float = 0.4
    rho_m: float = 6500.0
    samples_per_pixel: float = 1.0
    A_amp: float = 1.0
    feed_rate: float = 2.0

    # Must match sb3_cold_spray_generalized_gaussian_ppo_12(3).py.
    # Slicer/target construction is nominal; PPO execution is geometry-dependent.
    use_geometry_dependent_kernel: bool = True
    geometry_kernel_p: float = 1.0
    slope_smoothing_window: int = 9

    # -------------------------------------------------------------------------
    # 3) Velocity / offset constraints
    # -------------------------------------------------------------------------
    v_max: float = 1.5
    v_min: float = 1e-2
    base_v: float = 0.5
    delta_v_max: float = 1.0

    # Offset uses the same residual idea as 1D, but its baseline is fixed at 0.
    # The RL action directly produces Δoffset around 0.
    off_max: float = 0.05
    offset_limit: float = 0.10

    # -------------------------------------------------------------------------
    # 4) Target construction with 2D GPs
    # -------------------------------------------------------------------------
    # ======== CHANGED: đồng bộ với file 1D ========
    # Pipeline mới cho 2D:
    #   1) sample GP 2D zero-mean
    #   2) dựng target_base = target_base_mean + target_base_gp_amplitude * g_base
    #   3) solve least squares để recover baseline velocity theo từng snake segment
    #   4) sample GP modulation khác để tạo target cuối
    target_base_mean: float = 0.05
    target_base_gp_amplitude: float = 0.01
    target_mod_relative_amplitude_min: float = 0.00
    target_mod_relative_amplitude_max: float = 0.10
    gp_std: float = 1.0
    gp_lengthscale_x: float = 0.20
    gp_lengthscale_y: float = 0.20
    gp_jitter: float = 1e-8

    # ======== CHANGED: LSQ config đồng bộ với cold-spray-2d-random-bell ========
    n_offset_lanes: int = 7
    lambda_mass = 1e-5
    lambda_step_smooth = 1e-3
    lambda_lane_smooth = 5e-4
    positive_target_weight: float = 2.5
    chunk_cols: int = 128
    basis_dtype: str = "float32"
    scratch_dir: Optional[str] = None

    # ======== CHANGED: mode switch giống file 1D ========
    # True  -> mỗi reset()/episode sẽ sample target GP mới
    # False -> dùng đúng 1 shared target GP + shared noise GP cho tất cả episode
    gp_resample_every_reset: bool = True
    # Khi gp_resample_every_reset=False, luôn dùng đúng 1 shared bundle cố định
    # cho mọi env / mọi episode / mọi rollout diagnostic.
    shared_bundle_seed: int = 0
    n_envs: int = 8

    # ======== CHANGED: dùng trực tiếp trong công thức slide v_base = v0 + a*g_base ========
    # base_v đóng vai trò v0, còn v_base_gp_amplitude đóng vai trò a.
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
    ppo_clip_range: float = 0.1
    ppo_target_kl: float = 0.2
    ppo_n_epochs: int = 4
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_log_std_init: float = -4.0
    total_timesteps: int = 500000000

    resume: bool = True
    resume_model_path: str = str(CHECKPOINT_DIR / "best_model_12.zip")
    resume_vecnormalize_path: str = str(CHECKPOINT_DIR / "best_model_12_vecnormalize.pkl")
    save_freq: int = 50000


@dataclass
class GPTarget2DBundle:
    """
    Bundle target 2D hoàn chỉnh để có thể:
    - resample mỗi episode
    - hoặc giữ cố định cho mọi episode khi debug policy
    """
    v_base_vec: np.ndarray
    offset_base_vec: np.ndarray
    target_base: np.ndarray
    lsq_reconstructed_target_base: np.ndarray
    lsq_lane_mass_matrix: np.ndarray
    lsq_lane_offsets: np.ndarray
    target: np.ndarray
    h0_modification: np.ndarray
    baseline_deposition: np.ndarray
    g_base: np.ndarray
    g_mod: np.ndarray
    g_h0: np.ndarray
    target_modulation: np.ndarray
    sampled_target_base_mean_height: float
    sampled_target_mod_relative_amplitude: float
    sampled_h0_mod_relative_amplitude: float
    lsq_cost: float
    lsq_success: bool
    lsq_status: int


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


def _smooth_2d_reflect(height: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return height.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    h = np.pad(height.astype(np.float64), ((pad, pad), (pad, pad)), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=h)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=tmp)
    return out.astype(np.float32)


def _surface_gradient_magnitude_2d(
    height: np.ndarray,
    dx: float,
    smoothing_window: int,
) -> np.ndarray:
    h_smooth = _smooth_2d_reflect(height, smoothing_window).astype(np.float64)
    dh_dy, dh_dx = np.gradient(h_smooth, float(dx), float(dx))
    return np.sqrt(dh_dx * dh_dx + dh_dy * dh_dy).astype(np.float32)


def _geometry_efficiency_from_gradmag_2d(grad_mag: np.ndarray, p: float) -> np.ndarray:
    g = grad_mag.astype(np.float64)
    return np.power(1.0 + g * g, -0.5 * float(p)).astype(np.float32)

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

    def basis_for_segment(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        height_for_slope: Optional[np.ndarray] = None,
        use_geometry_dependent_kernel: bool = False,
        geometry_kernel_p: float = 1.0,
        slope_smoothing_window: int = 9,
    ) -> np.ndarray:
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        L = float(np.hypot(dx, dy))

        if L < 1e-12:
            r = np.hypot(self.X - x0, self.Y - y0)
            k = self._kernel(r)
            basis = (
                k / (self.rho_m * self._pix_area * (float(np.sum(k)) + 1e-16))
            )
        else:
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

        if use_geometry_dependent_kernel and height_for_slope is not None:
            grad_mag = _surface_gradient_magnitude_2d(
                np.asarray(height_for_slope, dtype=np.float32),
                dx=1.0 / float(self.n),
                smoothing_window=slope_smoothing_window,
            )
            basis = (
                _geometry_efficiency_from_gradmag_2d(grad_mag, geometry_kernel_p).astype(np.float64)
                * basis
            )
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
        + wx * wy * v11)

def flatten_map(M: np.ndarray) -> np.ndarray:
    return np.asarray(M, dtype=np.float64).reshape(-1)


def make_pixel_weights(cfg: ColdSpray2DConfig, target_map: np.ndarray) -> np.ndarray:
    # ======== CHANGED: weighting giống cold-spray-2d-random-bell ========
    W = np.ones_like(target_map, dtype=np.float64)
    W[target_map > 1e-12] = float(cfg.positive_target_weight)
    return W.reshape(-1)

class BasisBankMemmap:
    def __init__(self, cfg: ColdSpray2DConfig, deposition: DepositionModel2D, meta: List[dict]) -> None:
        self.cfg = cfg
        self.deposition = deposition
        self.meta = meta
        self.n_vars = len(meta)
        self.n_pix = cfg.grid_n * cfg.grid_n
        self.dtype = np.float16 if cfg.basis_dtype == "float16" else np.float32

        if cfg.scratch_dir is None:
            self._tmpdir_ctx = TemporaryDirectory()
            self.scratch_dir = Path(self._tmpdir_ctx.name)
        else:
            self._tmpdir_ctx = None
            self.scratch_dir = Path(cfg.scratch_dir)
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.scratch_dir / "basis_bank.dat"
        self.mm = np.memmap(
            self.path,
            dtype=self.dtype,
            mode="w+",
            shape=(self.n_vars, self.n_pix),
        )

    def build(self) -> None:
        for j, m in enumerate(self.meta):
            basis = self.deposition.basis_for_segment(
                m["x0"], m["y0"], m["x1"], m["y1"],
                height_for_slope=None,
                use_geometry_dependent_kernel=False,
            )
            self.mm[j, :] = basis.reshape(-1).astype(self.dtype, copy=False)
        self.mm.flush()

    def get_chunk(self, j0: int, j1: int, out_dtype=np.float32) -> np.ndarray:
        return np.asarray(self.mm[j0:j1, :], dtype=out_dtype)

    def close(self) -> None:
        try:
            if hasattr(self, "mm") and self.mm is not None:
                self.mm.flush()
                del self.mm
        finally:
            if self._tmpdir_ctx is not None:
                self._tmpdir_ctx.cleanup()


class LeastSquaresSlicer2DForSB3:
    """
    ======== CHANGED: LSQ formulation giống cold-spray-2d-random-bell ========
    - biến là mass w(step, lane)
    - có pixel weighting
    - có lambda_mass, lambda_step_smooth, lambda_lane_smooth
    - solve bằng LinearOperator + chunking

    Sau khi solve xong, ta collapse nghiệm mass về:
    - v_base_vec(step)
    - offset_base_vec(step)
    để phần RL hiện có vẫn dùng được.
    """
    def __init__(self, cfg: ColdSpray2DConfig, steps: List[dict]) -> None:
        self.cfg = cfg
        self.deposition = DepositionModel2D(
            grid_n=cfg.grid_n,
            sigma=cfg.sigma,
            beta=cfg.beta,
            eta=cfg.eta,
            rho_m=cfg.rho_m,
            samples_per_pixel=cfg.samples_per_pixel,
            A_amp=cfg.A_amp,
        )
        self.steps = steps
        self.offset_lanes = np.linspace(-cfg.off_max, cfg.off_max, cfg.n_offset_lanes, dtype=np.float64)

    def build_meta(self) -> List[dict]:
        meta: List[dict] = []
        for step_idx, st in enumerate(self.steps):
            for lane_idx, lane_off in enumerate(self.offset_lanes):
                x0 = float(np.clip(st["x0"] + lane_off * st["nx"], 0.0, 1.0))
                y0 = float(np.clip(st["y0"] + lane_off * st["ny"], 0.0, 1.0))
                x1 = float(np.clip(st["x1"] + lane_off * st["nx"], 0.0, 1.0))
                y1 = float(np.clip(st["y1"] + lane_off * st["ny"], 0.0, 1.0))
                meta.append({
                    "step_idx": step_idx,
                    "lane_idx": lane_idx,
                    "lane_off": float(lane_off),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "length": float(st["length"]),
                })
        return meta

    def _regularization_diag_counts(self, n_steps: int, lane_count: int) -> np.ndarray:
        n_vars = n_steps * lane_count
        reg_diag = np.zeros(n_vars, dtype=np.float64)

        if self.cfg.lambda_mass > 0.0:
            reg_diag += float(self.cfg.lambda_mass)

        if self.cfg.lambda_step_smooth > 0.0:
            lam = float(self.cfg.lambda_step_smooth)
            for t in range(n_steps):
                for lane in range(lane_count):
                    j = t * lane_count + lane
                    count = 0
                    if t > 0:
                        count += 1
                    if t < n_steps - 1:
                        count += 1
                    reg_diag[j] += lam * count

        if self.cfg.lambda_lane_smooth > 0.0:
            lam = float(self.cfg.lambda_lane_smooth)
            for t in range(n_steps):
                for lane in range(lane_count):
                    j = t * lane_count + lane
                    count = 0
                    if lane > 0:
                        count += 1
                    if lane < lane_count - 1:
                        count += 1
                    reg_diag[j] += lam * count

        return reg_diag

    def _compute_data_col_sq(
        self,
        basis_bank: BasisBankMemmap,
        pix_w: np.ndarray,
        chunk_cols: int,
    ) -> np.ndarray:
        n_vars = basis_bank.n_vars
        out = np.zeros(n_vars, dtype=np.float64)
        for j0 in range(0, n_vars, chunk_cols):
            j1 = min(j0 + chunk_cols, n_vars)
            chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)
            weighted = chunk * pix_w[None, :]
            out[j0:j1] = np.sum(weighted.astype(np.float64) ** 2, axis=1)
        return out

    def build_operator_and_rhs(
        self,
        basis_bank: BasisBankMemmap,
        meta: List[dict],
        target_map: np.ndarray,
    ) -> tuple[LinearOperator, np.ndarray, np.ndarray]:
        lane_count = self.cfg.n_offset_lanes
        n_steps = max(m["step_idx"] for m in meta) + 1
        n_vars = basis_bank.n_vars
        n_pix = basis_bank.n_pix
        chunk_cols = max(1, int(self.cfg.chunk_cols))

        pix_w = np.sqrt(make_pixel_weights(self.cfg, target_map)).astype(np.float64)
        b = flatten_map(target_map).astype(np.float64)
        b_data = b * pix_w

        n_mass = n_vars if self.cfg.lambda_mass > 0.0 else 0
        n_step = (n_steps - 1) * lane_count if self.cfg.lambda_step_smooth > 0.0 else 0
        n_lane = n_steps * (lane_count - 1) if self.cfg.lambda_lane_smooth > 0.0 else 0
        n_total = n_pix + n_mass + n_step + n_lane

        data_col_sq = self._compute_data_col_sq(basis_bank, pix_w, chunk_cols)
        reg_diag = self._regularization_diag_counts(n_steps, lane_count)
        col_scale = np.sqrt(np.maximum(data_col_sq + reg_diag, 1e-12))

        sqrt_lm = np.sqrt(self.cfg.lambda_mass) if self.cfg.lambda_mass > 0.0 else 0.0
        sqrt_ls = np.sqrt(self.cfg.lambda_step_smooth) if self.cfg.lambda_step_smooth > 0.0 else 0.0
        sqrt_ll = np.sqrt(self.cfg.lambda_lane_smooth) if self.cfg.lambda_lane_smooth > 0.0 else 0.0

        def matvec(z: np.ndarray) -> np.ndarray:
            w = z / col_scale
            parts: List[np.ndarray] = []
            y_data = np.zeros(n_pix, dtype=np.float64)
            for j0 in range(0, n_vars, chunk_cols):
                j1 = min(j0 + chunk_cols, n_vars)
                chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)
                y_data += chunk.T @ w[j0:j1]
            y_data *= pix_w
            parts.append(y_data)

            if n_mass > 0:
                parts.append(sqrt_lm * w)

            if n_step > 0:
                w2 = w.reshape(n_steps, lane_count)
                step_diff = (w2[:-1, :] - w2[1:, :]).reshape(-1)
                parts.append(sqrt_ls * step_diff)

            if n_lane > 0:
                w2 = w.reshape(n_steps, lane_count)
                lane_diff = (w2[:, :-1] - w2[:, 1:]).reshape(-1)
                parts.append(sqrt_ll * lane_diff)

            return np.concatenate(parts).astype(np.float64)

        def rmatvec(y: np.ndarray) -> np.ndarray:
            out = np.zeros(n_vars, dtype=np.float64)
            pos = 0

            y_data = y[pos:pos + n_pix]
            pos += n_pix
            y_data_weighted = y_data * pix_w
            for j0 in range(0, n_vars, chunk_cols):
                j1 = min(j0 + chunk_cols, n_vars)
                chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)
                out[j0:j1] += chunk @ y_data_weighted

            if n_mass > 0:
                y_mass = y[pos:pos + n_mass]
                pos += n_mass
                out += sqrt_lm * y_mass

            if n_step > 0:
                y_step = y[pos:pos + n_step].reshape(n_steps - 1, lane_count)
                pos += n_step
                acc = np.zeros((n_steps, lane_count), dtype=np.float64)
                acc[:-1, :] += y_step
                acc[1:, :] -= y_step
                out += sqrt_ls * acc.reshape(-1)

            if n_lane > 0:
                y_lane = y[pos:pos + n_lane].reshape(n_steps, lane_count - 1)
                pos += n_lane
                acc = np.zeros((n_steps, lane_count), dtype=np.float64)
                acc[:, :-1] += y_lane
                acc[:, 1:] -= y_lane
                out += sqrt_ll * acc.reshape(-1)

            return out / col_scale

        Aop = LinearOperator(
            shape=(n_total, n_vars),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        rhs_parts = [b_data]
        if n_mass > 0:
            rhs_parts.append(np.zeros(n_mass, dtype=np.float64))
        if n_step > 0:
            rhs_parts.append(np.zeros(n_step, dtype=np.float64))
        if n_lane > 0:
            rhs_parts.append(np.zeros(n_lane, dtype=np.float64))
        b_aug = np.concatenate(rhs_parts).astype(np.float64)

        return Aop, b_aug, col_scale

    def reconstruct_height_from_weights(
        self,
        basis_bank: BasisBankMemmap,
        w: np.ndarray,
    ) -> np.ndarray:
        n_pix = basis_bank.n_pix
        n_vars = basis_bank.n_vars
        chunk_cols = max(1, int(self.cfg.chunk_cols))

        h = np.zeros(n_pix, dtype=np.float64)
        for j0 in range(0, n_vars, chunk_cols):
            j1 = min(j0 + chunk_cols, n_vars)
            chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)
            h += chunk.T @ w[j0:j1]
        return h.reshape((self.cfg.grid_n, self.cfg.grid_n)).astype(np.float64)

    def solve(self, target_map: np.ndarray) -> Dict[str, Any]:
        meta = self.build_meta()
        basis_bank = BasisBankMemmap(self.cfg, self.deposition, meta)
        try:
            basis_bank.build()
            Aop, b_aug, col_scale = self.build_operator_and_rhs(basis_bank, meta, target_map)
            sol = lsq_linear(
                Aop,
                b_aug,
                bounds=(0.0, np.inf),
                lsmr_tol="auto",
                method="trf",
                max_iter=200,
            )
            z = sol.x.astype(np.float64)
            w = z / col_scale
            height = self.reconstruct_height_from_weights(basis_bank, w)
        finally:
            basis_bank.close()

        return {
            "w": w,
            "meta": meta,
            "height": height,
            "lsq_cost": float(sol.cost),
            "lsq_status": int(sol.status),
            "lsq_success": bool(sol.success),
        }


def _collapse_mass_solution_to_baseline_controls(
    cfg: ColdSpray2DConfig,
    steps: List[dict],
    w: np.ndarray,
    offset_lanes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ======== CHANGED: quy đổi nghiệm mass(step,lane) -> baseline controls dùng trong RL ========
    n_steps = len(steps)
    lane_count = len(offset_lanes)
    W = np.asarray(w, dtype=np.float64).reshape(n_steps, lane_count)
    lane_mass_sum = np.sum(W, axis=1)

    v_base_vec = np.zeros(n_steps, dtype=np.float32)
    offset_base_vec = np.zeros(n_steps, dtype=np.float32)

    for i, st in enumerate(steps):
        total_mass_i = float(lane_mass_sum[i])
        if total_mass_i <= 1e-12:
            v_i = float(cfg.v_max)
            off_i = 0.0
        else:
            dt_i = total_mass_i / max(float(cfg.feed_rate), 1e-12)
            v_i = float(st["length"]) / max(dt_i, 1e-12)
            best_lane = int(np.argmax(W[i]))
            off_i = float(offset_lanes[best_lane])

        v_base_vec[i] = np.float32(np.clip(v_i, cfg.v_min, cfg.v_max))
        offset_base_vec[i] = np.float32(np.clip(off_i, -cfg.off_max, cfg.off_max))

    return v_base_vec, offset_base_vec, W.astype(np.float32)


def recover_baseline_controls_from_target_base_lsq_2d(
    cfg: ColdSpray2DConfig,
    steps: List[dict],
    target_base: np.ndarray,
    h0_modification: np.ndarray,
) -> Dict[str, Any]:
    """
    Legacy inverse helper kept for compatibility only.

    The active 2D target pipeline below follows the 1D idea:
    h0 GP -> g_base -> v_base -> forward deposition -> target_base.
    It does not call this LSQ helper.
    """
    target_deficit_deposition = np.maximum(
        np.asarray(target_base, dtype=np.float64) - np.asarray(h0_modification, dtype=np.float64),
        0.0,
    )

    slicer = LeastSquaresSlicer2DForSB3(cfg=cfg, steps=steps)
    sol = slicer.solve(target_map=target_deficit_deposition)
    v_base_vec, offset_base_vec, lane_mass_matrix = _collapse_mass_solution_to_baseline_controls(
        cfg=cfg,
        steps=steps,
        w=np.asarray(sol["w"], dtype=np.float64),
        offset_lanes=slicer.offset_lanes,
    )
    return {
        "v_base_vec": v_base_vec,
        "offset_base_vec": offset_base_vec,
        "lsq_reconstructed_target_base": np.asarray(sol["height"], dtype=np.float32),
        "target_deficit_deposition": np.asarray(target_deficit_deposition, dtype=np.float32),
        "lsq_lane_mass_matrix": lane_mass_matrix,
        "lsq_lane_offsets": slicer.offset_lanes.astype(np.float32),
        "lsq_cost": float(sol["lsq_cost"]),
        "lsq_success": bool(sol["lsq_success"]),
        "lsq_status": int(sol["lsq_status"]),
    }


def generate_v_base_from_modified_h0_gp_2d(
    cfg: ColdSpray2DConfig,
    steps: List[dict],
    g_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct baseline generation following the slide idea:
        v_base(i) = v0 + a * g_base(x_i, y_i)

    Here g_base is a spatial 2D zero-mean GP. For each snake segment we sample
    the GP at the segment midpoint, then map that value to the local baseline
    velocity. No least-squares recovery is used.
    """
    v_base_vec = np.zeros(len(steps), dtype=np.float32)
    offset_base_vec = np.zeros(len(steps), dtype=np.float32)

    v0 = float(cfg.base_v)
    a = float(cfg.v_base_gp_amplitude)

    for i, st in enumerate(steps):
        xc = 0.5 * (float(st["x0"]) + float(st["x1"]))
        yc = 0.5 * (float(st["y0"]) + float(st["y1"]))
        g_i = _sample_map_at_xy_bilinear(g_base, xc, yc)
        v_i = v0 + a * g_i
        v_base_vec[i] = np.float32(np.clip(v_i, cfg.v_min, cfg.v_max))

    return v_base_vec, offset_base_vec


def simulate_target_base_from_modified_h0_2d(
    cfg: ColdSpray2DConfig,
    steps: List[dict],
    h0_modification: np.ndarray,
    v_base_vec: np.ndarray,
    offset_base_vec: np.ndarray,
) -> Dict[str, Any]:
    """
    Build target_base by simulating the nominal deposition process starting from
    the modified initial surface h0_modification, not from zero.
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

    height = np.asarray(h0_modification, dtype=np.float32).copy()

    for i, st in enumerate(steps):
        local_base_v = float(np.clip(v_base_vec[i], cfg.v_min, cfg.v_max))
        local_base_off = float(np.clip(offset_base_vec[i], -cfg.off_max, cfg.off_max))

        x0 = st["x0"] + local_base_off * st["nx"]
        y0 = st["y0"] + local_base_off * st["ny"]
        x1 = st["x1"] + local_base_off * st["nx"]
        y1 = st["y1"] + local_base_off * st["ny"]
        x0, y0, x1, y1 = _clip_segment_to_domain(x0, y0, x1, y1)

        dt_step = float(st["length"]) / max(local_base_v, 1e-12)
        m_step = float(cfg.feed_rate) * dt_step
        basis = deposition.basis_for_segment(
            x0, y0, x1, y1,
            height_for_slope=None,
            use_geometry_dependent_kernel=False,
        )
        height = height + (m_step * basis).astype(np.float32)

    target_base = np.asarray(height, dtype=np.float32)
    baseline_deposition = (
        target_base.astype(np.float64) - np.asarray(h0_modification, dtype=np.float64)
    ).astype(np.float32)

    return {
        "target_base": target_base,
        "baseline_deposition": baseline_deposition,
        "lsq_reconstructed_target_base": baseline_deposition.copy(),
        "lsq_lane_mass_matrix": np.zeros((len(steps), cfg.n_offset_lanes), dtype=np.float32),
        "lsq_lane_offsets": np.linspace(-cfg.off_max, cfg.off_max, cfg.n_offset_lanes, dtype=np.float32),
        "lsq_cost": 0.0,
        "lsq_success": False,
        "lsq_status": -1,
    }


def build_gp_target_2d_bundle(
    rng: np.random.Generator,
    cfg: ColdSpray2DConfig,
    steps: List[dict],
) -> GPTarget2DBundle:
    """
    New pipeline requested by the user:
        1) start from h0 = 0
        2) add zero-mean GP to get h0_modification
        3) sample another zero-mean GP g_base
        4) generate v_base directly with v_base = v0 + a * g_base
        5) simulate target_base starting from h0_modification
        6) add zero-mean GP on top of target_base to get the final RL target

    No least-squares recovery is used for v_base anymore.
    """
    # Step 1-2: modified initial surface.
    g_h0 = _sample_zero_mean_gp_2d_separable(
        rng=rng,
        grid_n=cfg.grid_n,
        std=cfg.gp_std,
        lengthscale_x=cfg.gp_lengthscale_x,
        lengthscale_y=cfg.gp_lengthscale_y,
        jitter=cfg.gp_jitter,
    )
    sampled_h0_mod_relative_amplitude = 0 #float(rng.uniform(cfg.target_mod_relative_amplitude_min, cfg.target_mod_relative_amplitude_max,))

    h0_modification = (
        sampled_h0_mod_relative_amplitude
        * float(cfg.target_base_mean)
        * g_h0.astype(np.float64)
    ).astype(np.float32)
    h0_modification = (h0_modification - np.mean(h0_modification)).astype(np.float32)

    # Step 3-4: baseline GP and direct baseline velocity generation.
    g_base = _sample_zero_mean_gp_2d_separable(
        rng=rng,
        grid_n=cfg.grid_n,
        std=cfg.gp_std,
        lengthscale_x=cfg.gp_lengthscale_x,
        lengthscale_y=cfg.gp_lengthscale_y,
        jitter=cfg.gp_jitter,
    )
    v_base_vec, offset_base_vec = generate_v_base_from_modified_h0_gp_2d(
        cfg=cfg,
        steps=steps,
        g_base=g_base,
    )

    # Step 5: simulate baseline target from the modified initial surface.
    base_sol = simulate_target_base_from_modified_h0_2d(
        cfg=cfg,
        steps=steps,
        h0_modification=h0_modification,
        v_base_vec=v_base_vec,
        offset_base_vec=offset_base_vec,
    )
    target_base = base_sol["target_base"]
    sampled_target_base_mean_height = float(np.mean(target_base.astype(np.float64)))

    # Step 6: final GP modification added on top of target_base.
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
        offset_base_vec=offset_base_vec.copy(),
        target_base=target_base.copy(),
        lsq_reconstructed_target_base=base_sol["lsq_reconstructed_target_base"].copy(),
        lsq_lane_mass_matrix=base_sol["lsq_lane_mass_matrix"].copy(),
        lsq_lane_offsets=base_sol["lsq_lane_offsets"].copy(),
        target=target.copy(),
        h0_modification=h0_modification.copy(),
        baseline_deposition=base_sol["baseline_deposition"].copy(),
        g_base=g_base.copy(),
        g_mod=g_mod.copy(),
        g_h0=g_h0.copy(),
        target_modulation=target_modulation.copy(),
        sampled_target_base_mean_height=float(sampled_target_base_mean_height),
        sampled_target_mod_relative_amplitude=float(sampled_target_mod_relative_amplitude),
        sampled_h0_mod_relative_amplitude=float(sampled_h0_mod_relative_amplitude),
        lsq_cost=float(base_sol["lsq_cost"]),
        lsq_success=bool(base_sol["lsq_success"]),
        lsq_status=int(base_sol["lsq_status"]),
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
        # ======== CHANGED: h0_modification là initial height GP zero-mean ========
        self.h0_modification = np.zeros_like(self.height)
        self.baseline_deposition = np.zeros_like(self.height)

        self.v_base_vec = np.full(self.n_steps, fill_value=cfg.base_v, dtype=np.float32)
        self.offset_base_vec = np.zeros(self.n_steps, dtype=np.float32)
        self.lsq_reconstructed_target_base = np.zeros_like(self.height)
        self.lsq_lane_mass_matrix = np.zeros((self.n_steps, cfg.n_offset_lanes), dtype=np.float32)
        self.lsq_lane_offsets = np.linspace(-cfg.off_max, cfg.off_max, cfg.n_offset_lanes, dtype=np.float32)
        # ======== CHANGED: g_base / g_mod là GP 2D thật của target ========
        self.g_base = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.g_mod = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.g_h0 = np.zeros((cfg.grid_n, cfg.grid_n), dtype=np.float32)
        self.sampled_target_base_mean_height = 0.0
        self.sampled_target_mod_relative_amplitude = 0.0
        self.sampled_h0_mod_relative_amplitude = 0.0
        self.lsq_cost = 0.0
        self.lsq_success = False
        self.lsq_status = 0
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

        # Action = [dv_action, offset_action]; v = v_base + Δv, offset = 0 + Δoffset
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
        self.offset_base_vec = bundle.offset_base_vec.copy()
        self.target_base = bundle.target_base.copy()
        self.lsq_reconstructed_target_base = bundle.lsq_reconstructed_target_base.copy()
        self.lsq_lane_mass_matrix = bundle.lsq_lane_mass_matrix.copy()
        self.lsq_lane_offsets = bundle.lsq_lane_offsets.copy()
        self.target = bundle.target.copy()
        self.h0_modification = bundle.h0_modification.copy()
        self.baseline_deposition = bundle.baseline_deposition.copy()
        self.g_base = bundle.g_base.copy()
        self.g_mod = bundle.g_mod.copy()
        self.g_h0 = bundle.g_h0.copy()
        self.target_modulation = bundle.target_modulation.copy()
        self.sampled_target_base_mean_height = float(bundle.sampled_target_base_mean_height)
        self.sampled_target_mod_relative_amplitude = float(bundle.sampled_target_mod_relative_amplitude)
        self.sampled_h0_mod_relative_amplitude = float(bundle.sampled_h0_mod_relative_amplitude)
        self.lsq_cost = float(bundle.lsq_cost)
        self.lsq_success = bool(bundle.lsq_success)
        self.lsq_status = int(bundle.lsq_status)

        if not initial:
            self.episode_index += 1

    def _current_step(self) -> dict:
        idx = int(np.clip(self.step_idx, 0, self.n_steps - 1))
        return self.steps[idx]

    def _current_base_velocity(self) -> float:
        return _step_local_base_velocity(self.v_base_vec, self.step_idx)

    def _current_base_offset(self) -> float:
        return float(self.offset_base_vec[int(np.clip(self.step_idx, 0, len(self.offset_base_vec) - 1))])

    def _get_obs(self) -> Dict[str, np.ndarray]:
        progress = np.float32(self.step_idx / max(self.n_steps, 1))
        remaining = np.float32(1.0 - progress)
        v_base_i = np.float32(self._current_base_velocity())
        total_offset = np.float32(self._current_base_offset() + self.offset)

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
                total_offset,
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

        # ======== CHANGED: initial height bây giờ là h0_modification thay vì 0 ========
        self.height = self.h0_modification.copy()
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
        local_base_off = self._current_base_offset()

        dv = float(np.clip(action[0], -1.0, 1.0)) * cfg.delta_v_max
        off_delta = float(np.clip(action[1], -1.0, 1.0)) * cfg.off_max

        v = float(np.clip(local_base_v + dv, cfg.v_min, cfg.v_max))

        # ======== CHANGED: 2D control now matches the 1D residual idea more closely.
        # Velocity remains residual around v_base: v = v_base + Δv.
        # Offset has zero baseline, so the action directly gives Δoffset around 0.
        # No accumulated offset drift: each step can choose a fresh offset in [-off_max, off_max].
        local_base_off = 0.0
        total_off = float(np.clip(local_base_off + off_delta, -cfg.offset_limit, cfg.offset_limit))
        self.offset = total_off

        self.velocity = v

        x0 = st["x0"] + total_off * st["nx"]
        y0 = st["y0"] + total_off * st["ny"]
        x1 = st["x1"] + total_off * st["nx"]
        y1 = st["y1"] + total_off * st["ny"]
        x0, y0, x1, y1 = _clip_segment_to_domain(x0, y0, x1, y1)

        dt_step = float(st["length"]) / max(v, 1e-12)
        m_step = cfg.feed_rate * dt_step

        h_old = self.height.copy()
        basis = self.deposition.basis_for_segment(
            x0, y0, x1, y1,
            height_for_slope=h_old,
            use_geometry_dependent_kernel=bool(cfg.use_geometry_dependent_kernel),
            geometry_kernel_p=float(cfg.geometry_kernel_p),
            slope_smoothing_window=int(cfg.slope_smoothing_window),
        )
        delta_h = (m_step * basis).astype(np.float32)

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
        doff_smooth = (total_off - self._prev_offset) / max(cfg.offset_limit, 1e-12)
        reward -= cfg.smoothness_penalty_v * (dv_smooth ** 2)
        reward -= cfg.smoothness_penalty_off * (doff_smooth ** 2)

        self._prev_velocity = v
        self._prev_offset = total_off

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
            "offset": float(total_off),
            "local_base_offset": float(local_base_off),
            "offset_correction": float(self.offset),
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
            "h0_mod_mean": float(np.mean(self.h0_modification)),
            "h0_mod_std": float(np.std(self.h0_modification)),
            # ======== CHANGED: log thêm mean height base + relative noise amp ========
            "sampled_target_base_mean_height": float(self.sampled_target_base_mean_height),
            "sampled_target_mod_relative_amplitude": float(self.sampled_target_mod_relative_amplitude),
            "lsq_cost": float(self.lsq_cost),
            "lsq_success": bool(self.lsq_success),
            "lsq_status": int(self.lsq_status),
            "baseline_reconstruction_mse": float(np.mean((self.lsq_reconstructed_target_base - self.baseline_deposition) ** 2)),
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
    steps = build_fixed_steps(cfg)
    shared_target_bundle = _resolve_shared_target_bundle(
        cfg=cfg,
        shared_target_bundle=shared_target_bundle,
        steps=steps,
    )

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
        save_prefix: str = "best_model_12",
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

                master_seed = sample_env_seed()
                data = rollout_model(
                    self.model,
                    self.venv,
                    self.cfg,
                    self.shared_target_bundle,
                    deterministic=self.deterministic,
                    env_seed=master_seed,
                )
                stochastic_data = rollout_model(
                    self.model,
                    self.venv,
                    self.cfg,
                    self.shared_target_bundle,
                    deterministic=False,
                    env_seed=master_seed,
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
    steps = build_fixed_steps(cfg)
    shared_target_bundle = _resolve_shared_target_bundle(
        cfg=cfg,
        shared_target_bundle=shared_target_bundle,
        steps=steps,
    )

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
    h0_modification = env.h0_modification.copy()
    baseline_deposition = env.baseline_deposition.copy()
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
        "h0_modification": h0_modification,
        "baseline_deposition": baseline_deposition,
        "g_base": env.g_base.copy(),
        "g_mod": env.g_mod.copy(),
        "g_h0": env.g_h0.copy(),
        "gp_resample_every_reset": bool(cfg.gp_resample_every_reset),
        # ======== CHANGED: expose sampled GP settings for plots/debug ========
        "sampled_target_base_mean_height": float(env.sampled_target_base_mean_height),
        "sampled_target_mod_relative_amplitude": float(env.sampled_target_mod_relative_amplitude),
        "sampled_h0_mod_relative_amplitude": float(env.sampled_h0_mod_relative_amplitude),
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
    sampled_h0_rel_amp = float(data.get("sampled_h0_mod_relative_amplitude", 0.0))
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
    - show h0_modification explicitly with a diverging colormap centered at 0
    - keep target / final height comparison
    - keep deterministic/stochastic velocity + offset tracking
    """
    target = data["target"]
    target_base = data["target_base"]
    target_modulation = data["target_modulation"]
    h0_modification = data["h0_modification"]
    baseline_deposition = data["baseline_deposition"]
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

    # ======== CHANGED: dùng scale đối xứng quanh 0 để nhìn rõ h0_modification âm/dương ========
    h0_abs = float(np.max(np.abs(h0_modification))) if np.size(h0_modification) else 1.0
    if h0_abs < 1e-12:
        h0_abs = 1.0
    h0_norm = TwoSlopeNorm(vmin=-h0_abs, vcenter=0.0, vmax=h0_abs)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(target_base, origin="lower")
    ax1.set_title("target_base (2D GP, before target noise)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(target, origin="lower", vmin=shared_vmin, vmax=shared_vmax)
    ax2.set_title("target_modified = target_base + GP noise")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ======== CHANGED: tem nhỏ để biết episode này sample GP nào ========
    sampled_base_mean = float(data.get("sampled_target_base_mean_height", 0.0))
    sampled_noise_rel_amp = float(data.get("sampled_target_mod_relative_amplitude", 0.0))
    sampled_h0_rel_amp = float(data.get("sampled_h0_mod_relative_amplitude", 0.0))
    rollout_mode = "resample/episode" if bool(data.get("gp_resample_every_reset", False)) else "shared target GP"
    ax2.text(
        0.02,
        0.98,
        f"mode = {rollout_mode}\nbase mean = {sampled_base_mean:.5f}\nnoise amp = {100.0 * sampled_noise_rel_amp:.2f}%\nh0 amp = {100.0 * sampled_h0_rel_amp:.2f}%",
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
    # ======== CHANGED: visualize h0_modification với colormap phân kỳ tâm 0 ========
    im4 = ax4.imshow(h0_modification, origin="lower", cmap="coolwarm", norm=h0_norm)
    ax4.set_title("h0_modification (GP on initial height, centered at 0)")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.text(
        0.02,
        0.98,
        f"min = {float(np.min(h0_modification)):.5f}\nmax = {float(np.max(h0_modification)):.5f}\nmean = {float(np.mean(h0_modification)):.5f}",
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(baseline_deposition, origin="lower")
    ax5.set_title("baseline_deposition = target_base - h0_mod")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 1])
    t_det_v = np.arange(len(det_velocities))
    t_b = np.arange(len(v_base_vec))
    ax6.plot(t_b, v_base_vec, label="v_base_i")
    ax6.plot(t_det_v, det_velocities, label="deterministic velocity")
    ax6.set_title("velocity tracking (deterministic)")
    ax6.grid(alpha=0.3)
    ax6.legend(loc="best")
    # ======== CHANGED: tem deterministic giống file 1D ========
    ax6.text(
        0.02,
        0.98,
        det_stamp_text,
        transform=ax6.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax7 = fig.add_subplot(gs[1, 2])
    t_sto_v = np.arange(len(sto_velocities))
    ax7.plot(t_b, v_base_vec, label="v_base_i")
    if len(sto_velocities) > 0:
        ax7.plot(t_sto_v, sto_velocities, label="stochastic velocity")
    ax7.set_title("velocity tracking (stochastic)")
    ax7.grid(alpha=0.3)
    ax7.legend(loc="best")
    # ======== CHANGED: tem stochastic giống file 1D ========
    ax7.text(
        0.02,
        0.98,
        sto_stamp_text,
        transform=ax7.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax8 = fig.add_subplot(gs[1, 3])
    t_det_o = np.arange(len(det_offsets))
    ax8.plot(t_det_o, det_offsets, label="deterministic offset")
    ax8.set_title("offset tracking (deterministic)")
    ax8.grid(alpha=0.3)
    ax8.legend(loc="best")

    ax9 = fig.add_subplot(gs[2, 0])
    t_sto_o = np.arange(len(sto_offsets))
    if len(sto_offsets) > 0:
        ax9.plot(t_sto_o, sto_offsets, label="stochastic offset")
    ax9.set_title("offset tracking (stochastic)")
    ax9.grid(alpha=0.3)
    ax9.legend(loc="best")

    ax10 = fig.add_subplot(gs[2, 1])
    height_diff = height - target
    diff_abs = float(np.max(np.abs(height_diff))) if np.size(height_diff) else 1.0
    if diff_abs < 1e-12:
        diff_abs = 1.0
    im10 = ax10.imshow(height_diff, origin="lower", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm")
    ax10.set_title("final_height - target_modified")
    fig.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)

    ax11 = fig.add_subplot(gs[2, 2])
    # ======== CHANGED: xem trực tiếp h0 ban đầu sau GP với scale phân kỳ tâm 0 ========
    im11 = ax11.imshow(data["heights"][0], origin="lower", cmap="coolwarm", norm=h0_norm)
    ax11.set_title("initial height h0 after GP")
    fig.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

    ax12 = fig.add_subplot(gs[2, 3])
    im12 = ax12.imshow(target_modulation, origin="lower", cmap="coolwarm")
    ax12.set_title("target_modification GP")
    fig.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_name is None:
        save_name = f"rollout_12d_snake_{stamp}.png"
    return _save_fig(fig, save_name)

# =============================================================================
# Untrained / zero-action diagnostics
# =============================================================================

def _resolve_shared_target_bundle(
    cfg: ColdSpray2DConfig,
    shared_target_bundle: Optional[GPTarget2DBundle] = None,
    steps: Optional[List[dict]] = None,
) -> Optional[GPTarget2DBundle]:
    """
    Resolve the one-and-only shared target bundle used in shared-target mode.

    Quy ước đúng cho mode:
    - gp_resample_every_reset=True:
        không dùng shared bundle.
    - gp_resample_every_reset=False:
        mọi env / mọi episode / mọi rollout phải dùng đúng cùng 1 bundle,
        được xác định bởi cfg.shared_bundle_seed nếu caller chưa truyền vào.
    """
    if cfg.gp_resample_every_reset:
        return None
    if shared_target_bundle is not None:
        return shared_target_bundle
    if steps is None:
        steps = build_fixed_steps(cfg)
    return build_shared_gp_target_2d_bundle(
        cfg=cfg,
        steps=steps,
        seed=int(cfg.shared_bundle_seed),
    )


def rollout_untrained_baseline(
    cfg: ColdSpray2DConfig,
    env_seed: int = 0,
):
    steps = build_fixed_steps(cfg)
    shared_target_bundle = _resolve_shared_target_bundle(
        cfg=cfg,
        shared_target_bundle=None,
        steps=steps,
    )

    env = ColdSpray2DEnv(
        cfg,
        shared_target_bundle=shared_target_bundle,
        env_seed=env_seed,
    )
    obs, _ = env.reset()

    heights = [env.height.copy()]
    velocities = [float(env.velocity)]
    offsets = [float(env._current_base_offset())]
    mses: List[float] = []

    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    while True:
        obs, r, terminated, truncated, info = env.step(zero_action)

        heights.append(env.height.copy())
        velocities.append(float(info["velocity"]))
        offsets.append(float(info["offset"]))
        mses.append(float(info["mse"]))

        if terminated or truncated:
            break

    final_height = heights[-1]
    final_mse = float(np.mean((final_height - env.target) ** 2))

    return {
        "heights": np.stack(heights, axis=0),
        "target": env.target.copy(),
        "target_base": env.target_base.copy(),
        "target_modulation": env.target_modulation.copy(),
        "h0_modification": env.h0_modification.copy(),
        "baseline_deposition": env.baseline_deposition.copy(),
        "lsq_reconstructed_target_base": env.lsq_reconstructed_target_base.copy(),
        "g_base": env.g_base.copy(),
        "g_mod": env.g_mod.copy(),
        "g_h0": env.g_h0.copy(),
        "gp_resample_every_reset": bool(cfg.gp_resample_every_reset),
        "sampled_target_base_mean_height": float(env.sampled_target_base_mean_height),
        "sampled_target_mod_relative_amplitude": float(env.sampled_target_mod_relative_amplitude),
        "sampled_h0_mod_relative_amplitude": float(env.sampled_h0_mod_relative_amplitude),
        "abs_gp_scale": float(
            env.sampled_target_mod_relative_amplitude
            * float(np.mean(env.target_base.astype(np.float64)))
        ),
        "v_base_vec": env.v_base_vec.copy(),
        "offset_base_vec": env.offset_base_vec.copy(),
        "velocities": np.array(velocities, dtype=np.float32),
        "offsets": np.array(offsets, dtype=np.float32),
        "mses": np.array(mses, dtype=np.float32),
        "final_mse": final_mse,
        "lsq_cost": float(env.lsq_cost),
        "lsq_success": bool(env.lsq_success),
        "lsq_status": int(env.lsq_status),
    }


def plot_untrained_rollout_summary(data: Dict[str, Any], save_name: Optional[str] = None) -> str:
    final_height = np.asarray(data["heights"][-1], dtype=np.float64)
    target = np.asarray(data["target"], dtype=np.float64)
    target_base = np.asarray(data["target_base"], dtype=np.float64)
    h0_modification = np.asarray(data["h0_modification"], dtype=np.float64)
    baseline_deposition = np.asarray(data["baseline_deposition"], dtype=np.float64)
    target_modification = np.asarray(data["target_modulation"], dtype=np.float64)
    v_base = np.asarray(data["v_base_vec"], dtype=np.float64)
    # ======== FIXED: align untrained logs with step indices ========
    # data["velocities"] has length T+1 because rollout_untrained_baseline logs:
    #   - velocities[0]   = env.velocity right after reset (before any step)
    #   - velocities[k+1] = info["velocity"] after step k
    #
    # Meanwhile v_base has length T and v_base[k] corresponds to step k.
    # Therefore the correctly aligned comparison is:
    #   v_base[k]  <-> velocities[k+1]
    # i.e. we must use [1:], not [:-1].
    velocities = np.asarray(data["velocities"][1:], dtype=np.float64)
    offset_base = np.asarray(data["offset_base_vec"], dtype=np.float64)
    # Same alignment logic for offsets:
    #   offsets[0]   = initial value before any step
    #   offsets[k+1] = executed offset after step k
    # so the step-aligned series is offsets[1:].
    offsets = np.asarray(data["offsets"][1:], dtype=np.float64)

    final_mse = float(data.get("final_mse", np.mean((final_height - target) ** 2)))
    relative_mse = compute_relative_mse_mean_target(final_height=final_height, target=target)

    # ======== FIXED: optional sanity checks for untrained zero-action rollout ========
    # For zero_action = [0, 0], velocity should match v_base step-by-step up to tiny float noise.
    # These values can be printed or displayed to confirm the alignment fix.
    vel_alignment_max_abs = float(np.max(np.abs(v_base - velocities))) if len(v_base) == len(velocities) else np.nan
    off_alignment_max_abs = float(np.max(np.abs(offset_base - offsets))) if len(offset_base) == len(offsets) else np.nan

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    im = axes[0, 0].imshow(target_base)
    plt.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title("target_base (2D GP, before target noise)")

    im = axes[0, 1].imshow(target)
    plt.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title("target_modified = target_base + GP noise")
    mode_txt = "shared/fixed target" if not bool(data.get("gp_resample_every_reset", True)) else "resample/episode"
    txt = (
        f"mode = {mode_txt}\n"
        f"base mean = {float(np.mean(target_base)):.6f}\n"
        f"noise amp = {100.0 * float(data.get('sampled_target_mod_relative_amplitude', 0.0)):.2f}%\n"
        f"h0 amp = {100.0 * float(data.get('sampled_h0_mod_relative_amplitude', 0.0)):.2f}%"
    )
    axes[0, 1].text(
        0.02, 0.98, txt,
        transform=axes[0, 1].transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    im = axes[0, 2].imshow(final_height)
    plt.colorbar(im, ax=axes[0, 2])
    axes[0, 2].set_title(f"final height | final mse={final_mse:.3e}")

    vmax_h0 = float(max(abs(np.min(h0_modification)), abs(np.max(h0_modification)), 1e-12))
    im = axes[0, 3].imshow(
        h0_modification,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_h0, vmax=vmax_h0),
    )
    plt.colorbar(im, ax=axes[0, 3])
    axes[0, 3].set_title("h0_modification (GP on initial height, centered at 0)")
    txt_h0 = (
        f"min = {float(np.min(h0_modification)):.5f}\n"
        f"max = {float(np.max(h0_modification)):.5f}\n"
        f"mean = {float(np.mean(h0_modification)):.5f}"
    )
    axes[0, 3].text(
        0.02, 0.98, txt_h0,
        transform=axes[0, 3].transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    im = axes[1, 0].imshow(baseline_deposition)
    plt.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("baseline_deposition = target_base - h0_mod")

    axes[1, 1].plot(v_base, label="v_base_i")
    axes[1, 1].plot(velocities, label="deterministic velocity")
    axes[1, 1].legend()
    axes[1, 1].set_title("velocity tracking (deterministic)")
    stamp = (
        f"GP amplitude = {100.0 * float(data.get('sampled_target_mod_relative_amplitude', 0.0)):.2f}% of mean target_base\n"
        f"abs GP scale = {float(data.get('abs_gp_scale', 0.0)):.6f}\n"
        f"relative MSE = {relative_mse:.6f}\n"
        f"max |v_base - vel| = {vel_alignment_max_abs:.3e}"
    )
    axes[1, 1].text(
        0.02, 0.98, stamp,
        transform=axes[1, 1].transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    axes[1, 2].plot(v_base, label="v_base_i")
    axes[1, 2].plot(velocities, label="stochastic velocity")
    axes[1, 2].legend()
    axes[1, 2].set_title("velocity tracking (stochastic)")
    axes[1, 2].text(
        0.02, 0.98, "untrain zero-action\nstochastic == deterministic",
        transform=axes[1, 2].transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    axes[1, 3].plot(offsets, label="deterministic offset")
    axes[1, 3].plot(offset_base, linestyle="--", label="offset_base")
    axes[1, 3].legend()
    axes[1, 3].set_title(f"offset tracking (deterministic) | max err={off_alignment_max_abs:.3e}")

    axes[2, 0].plot(offsets, label="stochastic offset")
    axes[2, 0].plot(offset_base, linestyle="--", label="offset_base")
    axes[2, 0].legend()
    axes[2, 0].set_title(f"offset tracking (stochastic) | max err={off_alignment_max_abs:.3e}")

    err = final_height - target
    vmax_err = float(max(abs(np.min(err)), abs(np.max(err)), 1e-12))
    im = axes[2, 1].imshow(
        err,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_err, vmax=vmax_err),
    )
    plt.colorbar(im, ax=axes[2, 1])
    axes[2, 1].set_title("final_height - target_modified")

    vmax_h0b = float(max(abs(np.min(h0_modification)), abs(np.max(h0_modification)), 1e-12))
    im = axes[2, 2].imshow(
        h0_modification,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_h0b, vmax=vmax_h0b),
    )
    plt.colorbar(im, ax=axes[2, 2])
    axes[2, 2].set_title("initial height h0 after GP")

    vmax_tm = float(max(abs(np.min(target_modification)), abs(np.max(target_modification)), 1e-12))
    im = axes[2, 3].imshow(
        target_modification,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_tm, vmax=vmax_tm),
    )
    plt.colorbar(im, ax=axes[2, 3])
    axes[2, 3].set_title("target modification GP")

    plt.tight_layout()

    if save_name is None:
        save_name = f"untrained_rollout_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return _save_fig(fig, save_name)


def make_and_save_untrained_baseline_figure(
    cfg: ColdSpray2DConfig,
    seed: int = 987,
    save_name: str = "untrained_rollout_baseline_check.png",
) -> str:
    data = rollout_untrained_baseline(cfg=cfg, env_seed=seed)
    return plot_untrained_rollout_summary(data, save_name=save_name)

# =============================================================================
# Standalone 2D target-GP amplitude sensitivity analysis
# =============================================================================

# ----------------------------- USER SETTINGS -----------------------------
# Server paths.
CHECKPOINT_DIR = Path("/netscratch/nham/checkpoints")
MODEL_PREFIX = "best_model_12"
OUT_DIR = Path("/netscratch/nham/logs/pics-ppo-12")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matches the requested diagram: 0%, 1%, ..., 10%, with 2048 rollouts each.
AMPLITUDES_PERCENT = np.arange(0.0, 10.5, 0.5, dtype=np.float64)
N_TESTS_PER_AMPLITUDE = 2048
DETERMINISTIC_POLICY = True
GLOBAL_SEED = 20260721
FIG_DPI = 400


def make_analysis_config(amplitude_relative: float) -> ColdSpray2DConfig:
    """Return the exact 2D training configuration with one fixed target-GP amplitude."""
    if not np.isfinite(amplitude_relative) or not 0.0 <= float(amplitude_relative) <= 1.0:
        raise ValueError("amplitude_relative must be finite and lie in [0, 1].")
    return ColdSpray2DConfig(
        grid_n=32,
        spacing_px=2,
        ds=0.05,
        sigma=0.08,
        beta=1.0,
        eta=0.4,
        rho_m=6500.0,
        samples_per_pixel=1.0,
        A_amp=1.0,
        n_offset_lanes=7,
        feed_rate=2.0,
        v_max=1.5,
        v_min=0.01,
        base_v=0.5,
        delta_v_max=1.0,
        target_base_mean=0.05,
        target_base_gp_amplitude=0.01,
        off_max=0.15,
        offset_limit=0.30,
        target_mod_relative_amplitude_min=float(amplitude_relative),
        target_mod_relative_amplitude_max=float(amplitude_relative),
        gp_std=1.0,
        gp_lengthscale_x=0.20,
        gp_lengthscale_y=0.20,
        gp_jitter=1e-8,
        gp_resample_every_reset=True,
        n_envs=1,
        resume=True,
        use_geometry_dependent_kernel=True,
        geometry_kernel_p=1.0,
        slope_smoothing_window=9,
    )


def checkpoint_paths() -> Tuple[Path, Path]:
    model_path = CHECKPOINT_DIR / f"{MODEL_PREFIX}.zip"
    vecnorm_path = CHECKPOINT_DIR / f"{MODEL_PREFIX}_vecnormalize.pkl"
    missing = [str(p) for p in (model_path, vecnorm_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint file(s):\n  " + "\n  ".join(missing)
            + "\nUpload both files to CHECKPOINT_DIR or edit CHECKPOINT_DIR/MODEL_PREFIX."
        )
    return model_path, vecnorm_path


def make_loading_env() -> DummyVecEnv:
    cfg0 = make_analysis_config(0.0)
    return DummyVecEnv([
        lambda: Monitor(ColdSpray2DEnv(cfg0, env_seed=GLOBAL_SEED))
    ])


def checkpoint_policy_keys(model_path: Path) -> List[str]:
    """Read policy.pth without unpickling SB3's Python metadata."""
    try:
        with ZipFile(model_path, "r") as archive:
            with archive.open("policy.pth", "r") as policy_file:
                payload = policy_file.read()
        try:
            state_dict = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before weights_only was introduced
            state_dict = torch.load(BytesIO(payload), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not inspect policy.pth in {model_path}: {exc}") from exc

    if not isinstance(state_dict, dict):
        raise RuntimeError("policy.pth does not contain a PyTorch state_dict.")
    return [str(key) for key in state_dict.keys()]


def checkpoint_uses_expected_extractor(model_path: Path) -> bool:
    """Confirm that the checkpoint contains the custom map-CNN/state-MLP extractor."""
    keys = checkpoint_policy_keys(model_path)
    required_suffixes = (
        "map_cnn.0.weight",
        "map_head.0.weight",
        "state_mlp.0.weight",
    )
    missing = [suffix for suffix in required_suffixes if not any(key.endswith(suffix) for key in keys)]
    if missing:
        preview = "\n  ".join(keys[:30])
        raise RuntimeError(
            "The selected checkpoint was not trained with the expected "
            "ColdSpray2DMultiInputExtractor. Missing parameter suffixes: "
            f"{missing}. First checkpoint keys:\n  {preview}"
        )
    return True


def load_analysis_model() -> Tuple[PPO, VecNormalize]:
    model_path, vecnorm_path = checkpoint_paths()
    venv = VecNormalize.load(str(vecnorm_path), make_loading_env())
    venv.training = False
    venv.norm_reward = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_uses_expected_extractor(model_path)

    # IMPORTANT: the custom extractor class inside an older SB3 checkpoint may
    # fail to deserialize on Python 3.12/cloudpickle ("code() argument 13 ...").
    # Override only the metadata that contains Python class objects; the actual
    # learned tensors are still loaded strictly from policy.pth.
    policy_kwargs = {
        "log_std_init": -4.0,
        "features_extractor_class": ColdSpray2DMultiInputExtractor,
        "features_extractor_kwargs": {
            "cnn_output_dim": 128,
            "state_output_dim": 32,
        },
    }
    custom_objects = {
        "policy_class": MultiInputActorCriticPolicy,
        "policy_kwargs": policy_kwargs,
        "observation_space": venv.observation_space,
        "action_space": venv.action_space,
    }

    try:
        model = PPO.load(
            str(model_path),
            env=venv,
            device=device,
            custom_objects=custom_objects,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint tensors could not be loaded into the architecture from "
            "sb3_cold_spray_generalized_gaussian_ppo_12(3).py. Verify that "
            f"MODEL_PREFIX={MODEL_PREFIX!r} points to the matching model.\n"
            f"Original load error:\n{exc}"
        ) from exc

    # Fail early if a future edit silently changes the analysis observation API.
    if model.observation_space != venv.observation_space:
        raise RuntimeError("Loaded model and VecNormalize observation spaces differ.")
    print(f"Loaded model:        {model_path}")
    print(f"Loaded VecNormalize: {vecnorm_path}")
    print(f"Device:              {device}")
    return model, venv


def batch_dict_observation(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32)[None, ...] for key, value in obs.items()}


def evaluate_one_rollout(
    model: PPO,
    venv: VecNormalize,
    amplitude_relative: float,
    env_seed: int,
) -> Dict[str, Any]:
    # Fix all randomness belonging to this rollout. The environment still samples
    # independent 2D g_base, g_h0 and g_mod fields from env_seed.
    np.random.seed(GLOBAL_SEED + env_seed)
    torch.manual_seed(GLOBAL_SEED + env_seed)

    cfg = make_analysis_config(amplitude_relative)
    env = ColdSpray2DEnv(cfg=cfg, env_seed=env_seed)
    obs, _ = env.reset(seed=env_seed)

    # This experiment varies TARGET GP only; h0 must remain exactly flat.
    if abs(float(env.sampled_h0_mod_relative_amplitude)) > 1e-12:
        raise AssertionError("h0 GP amplitude must be exactly 0 in target-GP analysis.")
    if float(np.max(np.abs(env.h0_modification))) > 1e-10:
        raise AssertionError("h0 must be flat in target-GP analysis.")
    if not np.isclose(
        env.sampled_target_mod_relative_amplitude,
        amplitude_relative,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("The sampled target amplitude is not the requested fixed level.")

    while True:
        obs_norm = venv.normalize_obs(batch_dict_observation(obs))
        action, _ = model.predict(obs_norm, deterministic=DETERMINISTIC_POLICY)
        obs, _, terminated, truncated, _ = env.step(np.asarray(action[0], dtype=np.float32))
        if terminated or truncated:
            break

    relative_mse = compute_relative_mse_mean_target(env.height, env.target)
    if not np.isfinite(relative_mse):
        raise FloatingPointError(
            f"Non-finite relative MSE at amplitude={amplitude_relative}, seed={env_seed}."
        )
    return {
        "relative_mse": float(relative_mse),
        "max_overshoot": float(np.max(np.maximum(env.height - env.target, 0.0))),
        "seed": int(env_seed),
        "target_GP_amplitude_percent": 100.0 * float(amplitude_relative),
        "h0_GP_amplitude_percent": 0.0,
        "target_mean": float(np.mean(env.target)),
        "target_base_mean": float(np.mean(env.target_base)),
        "target_mod_std": float(np.std(env.target_modulation)),
        "h0_std": float(np.std(env.h0_modification)),
    }


def build_eval_seed(amplitude_percent: float, run_index: int) -> int:
    return int(GLOBAL_SEED + 100000 * int(round(amplitude_percent)) + run_index)


def run_sweep(model: PPO, venv: VecNormalize) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    total = len(AMPLITUDES_PERCENT) * N_TESTS_PER_AMPLITUDE
    completed = 0

    for amp_percent in AMPLITUDES_PERCENT:
        amp_relative = float(amp_percent / 100.0)
        print(f"Evaluating target GP amplitude {amp_percent:.2f}% "
              f"({N_TESTS_PER_AMPLITUDE} independent 2D rollouts)...")
        for run_index in range(N_TESTS_PER_AMPLITUDE):
            seed = build_eval_seed(float(amp_percent), run_index)
            row = evaluate_one_rollout(model, venv, amp_relative, seed)
            row["amplitude_percent"] = float(amp_percent)
            row["run_index"] = int(run_index)
            row["model_prefix"] = MODEL_PREFIX
            rows.append(row)
            completed += 1
            if completed % 256 == 0 or completed == total:
                print(f"  progress: {completed}/{total}")

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby("amplitude_percent", as_index=False).agg(
        mean_relative_mse=("relative_mse", "mean"),
        std_relative_mse=("relative_mse", "std"),
        min_relative_mse=("relative_mse", "min"),
        max_relative_mse=("relative_mse", "max"),
        mean_overshoot=("max_overshoot", "mean"),
        mean_target_mod_std=("target_mod_std", "mean"),
        mean_h0_std=("h0_std", "mean"),
    )
    summary_df["std_relative_mse"] = summary_df["std_relative_mse"].fillna(0.0)
    return raw_df, summary_df


def save_results(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[Path, Path]:
    raw_path = OUT_DIR / f"2d_target_gp_amp_raw_{MODEL_PREFIX}.csv"
    summary_path = OUT_DIR / f"2d_target_gp_amp_summary_{MODEL_PREFIX}.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return raw_path, summary_path


def plot_diagram(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[Path, Path]:
    # Thesis-ready typography.  The width grows with the number of tested
    # amplitude levels, so adding more levels does not squeeze all labels into
    # the same horizontal space.
    n_levels = max(int(len(summary_df)), 1)
    fig_width = min(max(14.5, 0.72 * n_levels), 24.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.6))

    ax.scatter(
        raw_df["amplitude_percent"],
        raw_df["relative_mse"],
        s=20,
        alpha=0.32,
        color="#5DADE2",
        edgecolors="none",
        label="individual runs",
        zorder=2,
    )

    x = summary_df["amplitude_percent"].to_numpy(dtype=np.float64)
    mean = summary_df["mean_relative_mse"].to_numpy(dtype=np.float64)
    std = summary_df["std_relative_mse"].to_numpy(dtype=np.float64)
    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="o-",
        color="#1F77B4",
        ecolor="#1F77B4",
        linewidth=3.0,
        elinewidth=2.4,
        markersize=8.0,
        capsize=5,
        capthick=2.2,
        label="mean ± std",
        zorder=4,
    )

    # Alternate mean labels above and below the mean markers.  Four different
    # point offsets prevent neighbouring labels from forming one dense row.
    # A label intended for the lower side is automatically moved upward when
    # the mean is too close to zero, so labels never collide with the x-axis.
    finite_raw = raw_df["relative_mse"].to_numpy(dtype=np.float64)
    finite_raw = finite_raw[np.isfinite(finite_raw)]
    data_top = float(np.max(finite_raw)) if finite_raw.size else 1.0
    error_top = float(np.max(mean + std)) if mean.size else data_top
    visible_top = max(data_top, error_top, 1e-9)
    lower_safety_level = 0.12 * visible_top
    upper_offsets = (16, 34)
    lower_offsets = (-18, -34)

    for idx, (xi, yi, si) in enumerate(zip(x, mean, std)):
        prefer_above = (idx % 2 == 0)

        if prefer_above or yi <= lower_safety_level:
            anchor_y = float(yi)
            offset_y = upper_offsets[(idx // 2) % len(upper_offsets)]
            vertical_alignment = "bottom"
        else:
            anchor_y = float(yi)
            offset_y = lower_offsets[(idx // 2) % len(lower_offsets)]
            vertical_alignment = "top"

        ax.annotate(
            f"{yi:.4f}",
            xy=(xi, anchor_y),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=13,
            fontweight="bold",
            color="#243746",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
            ),
            annotation_clip=False,
            zorder=6,
        )

    ax.set_xlabel("Relative target GP amplitude (%)", fontsize=18, labelpad=12)
    ax.set_ylabel("Relative MSE", fontsize=18, labelpad=12)
    ax.set_title(
        "Individual runs + mean ± std\n"
        f"Relative target GP amplitude vs relative MSE "
        f"({N_TESTS_PER_AMPLITUDE} eval samples each)",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    ax.set_xticks(AMPLITUDES_PERCENT)
    ax.set_xlim(float(AMPLITUDES_PERCENT.min()) - 0.4, float(AMPLITUDES_PERCENT.max()) + 0.4)
    ax.set_ylim(0.0, visible_top * 1.18)
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.4, length=6)
    ax.grid(True, alpha=0.25, linewidth=1.0)
    ax.legend(loc="upper left", fontsize=15, framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    fig.tight_layout(pad=1.2)

    png_path = OUT_DIR / f"2d_target_gp_amp_relative_mse_{MODEL_PREFIX}.png"
    pdf_path = OUT_DIR / f"2d_target_gp_amp_relative_mse_{MODEL_PREFIX}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    print("2D TARGET-GP amplitude analysis; h0 GP amplitude is fixed at 0%.")
    print(f"Levels: {AMPLITUDES_PERCENT.tolist()}")
    print(f"Rollouts per level: {N_TESTS_PER_AMPLITUDE}")
    model, venv = load_analysis_model()
    raw_df, summary_df = run_sweep(model, venv)
    raw_path, summary_path = save_results(raw_df, summary_df)
    print(summary_df.to_string(index=False))
    png_path, pdf_path = plot_diagram(raw_df, summary_df)
    print(f"Saved raw CSV:     {raw_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved PNG:         {png_path}")
    print(f"Saved PDF:         {pdf_path}")
