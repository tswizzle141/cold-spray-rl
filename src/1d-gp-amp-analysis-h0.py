from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from IPython.display import display
except Exception:
    display = print

# ============================================================
# 1. USER SETTINGS
# ============================================================

CHECKPOINT_DIR = Path("/content/drive/MyDrive/thesis/checkpoint")
MODEL_PREFIX = "best_model_3"   # change to "best_model_2" if needed
OUT_DIR = Path("/content/gp_bumps_eval_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMPLITUDES_PERCENT = np.arange(0.0, 10.0 + 0.5, 0.5, dtype=np.float64)
N_TESTS_PER_AMPLITUDE = 2048
DETERMINISTIC_POLICY = True
GLOBAL_SEED = 20260630
FIG_DPI = 400

np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# ============================================================
# 2. CONFIG
# ============================================================

@dataclass
class ColdSpray1DConfig:
    n_bins: int = 256
    x_max: float = 1.0
    v_max: float = 1.5
    v_min: float = 1e-2
    base_v: float = 0.5
    delta_v_max: float = 1.0
    feed_rate: float = 2.0
    sigma: float = 0.02
    use_geometry_dependent_kernel: bool = True
    geometry_kernel_p: float = 1.0
    slope_smoothing_window: int = 9
    n_steps: int = 64
    v_base_gp_amplitude: float = 0.05
    gp_std: float = 1.0
    gp_lengthscale: float = 0.25
    gp_jitter: float = 1e-8
    action_penalty: float = 0.1
    smoothness_penalty: float = 10.0
    overshoot_weight: float = 10.0
    reward_scale: float = 10.0

cfg = ColdSpray1DConfig()

# ============================================================
# 3. UTILITIES
# ============================================================

def kernel_gaussian(x_grid: np.ndarray, x0: float, sigma: float, dx: float) -> np.ndarray:
    z = (x_grid - x0) / sigma
    w = np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * np.pi))
    return (w * dx).astype(np.float32)


def smooth_1d_reflect(y: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return y.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y.astype(np.float64), pad_width=pad, mode="reflect")
    ker = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y_pad, ker, mode="valid").astype(np.float32)


def surface_slope(height: np.ndarray, dx: float, smoothing_window: int) -> np.ndarray:
    h_smooth = smooth_1d_reflect(height, smoothing_window)
    return np.gradient(h_smooth.astype(np.float64), float(dx)).astype(np.float32)


def geometry_efficiency_from_slope(slope: np.ndarray, p: float) -> np.ndarray:
    m = slope.astype(np.float64)
    eta = np.power(1.0 + m * m, -0.5 * float(p))
    return eta.astype(np.float32)


def rbf_covariance(x: np.ndarray, std: float, lengthscale: float, jitter: float) -> np.ndarray:
    x = x.astype(np.float64)
    diff = x[:, None] - x[None, :]
    K = (std ** 2) * np.exp(-0.5 * (diff / max(lengthscale, 1e-12)) ** 2)
    K += jitter * np.eye(len(x), dtype=np.float64)
    return K


def sample_zero_mean_gp(rng: np.random.Generator, x_grid: np.ndarray, std: float, lengthscale: float, jitter: float) -> np.ndarray:
    K = rbf_covariance(x_grid, std, lengthscale, jitter)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(len(x_grid))
    gp = L @ z
    gp = gp - np.mean(gp)
    return gp.astype(np.float32)


def compute_relative_mse_mean_target(final_height: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    err = final_height.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(err ** 2))
    mean_target = float(np.mean(target.astype(np.float64)))
    return mse / max(mean_target * mean_target, eps)


def compute_max_overshoot(final_height: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.maximum(final_height - target, 0.0)))

# ============================================================
# 4. FIXED GP PROFILES
# ============================================================

@dataclass
class GPProfiles:
    x_grid: np.ndarray
    v_base_vec: np.ndarray
    target_base: np.ndarray
    target: np.ndarray
    h0_modification: np.ndarray
    baseline_deposition: np.ndarray


def build_fixed_h0_gp_profiles(cfg: ColdSpray1DConfig, rng: np.random.Generator, h0_relative_amplitude: float) -> GPProfiles:
    x_grid = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
    dx = float(x_grid[1] - x_grid[0])

    g_base = sample_zero_mean_gp(rng, x_grid, cfg.gp_std, cfg.gp_lengthscale, cfg.gp_jitter)
    v_base_raw = float(cfg.base_v) + float(cfg.v_base_gp_amplitude) * g_base.astype(np.float64)
    v_base_vec = np.clip(v_base_raw, cfg.v_min, cfg.v_max).astype(np.float32)

    baseline_deposition = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)
    target_base = baseline_deposition.copy()
    target = np.maximum(target_base, 0.0).astype(np.float32)

    g_h0 = sample_zero_mean_gp(rng, x_grid, cfg.gp_std, cfg.gp_lengthscale, cfg.gp_jitter)
    h0_modification = (
        float(h0_relative_amplitude)
        * float(np.mean(baseline_deposition.astype(np.float64)))
        * g_h0.astype(np.float64)
    ).astype(np.float32)
    h0_modification = (h0_modification - np.mean(h0_modification)).astype(np.float32)

    return GPProfiles(
        x_grid=x_grid.copy(),
        v_base_vec=v_base_vec.copy(),
        target_base=target_base.copy(),
        target=target.copy(),
        h0_modification=h0_modification.copy(),
        baseline_deposition=baseline_deposition.copy(),
    )

# ============================================================
# 5. ENVIRONMENT
# ============================================================

class ColdSpray1DTestEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: ColdSpray1DConfig, h0_relative_amplitude: float = 0.03, env_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.h0_relative_amplitude = float(h0_relative_amplitude)
        self._rng = np.random.default_rng(env_seed)

        self.x_grid = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.ds = float(cfg.x_max / cfg.n_steps)

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

    def _set_profiles(self) -> None:
        profiles = build_fixed_h0_gp_profiles(self.cfg, self._rng, self.h0_relative_amplitude)
        self.v_base_vec = profiles.v_base_vec.copy()
        self.target_base = profiles.target_base.copy()
        self.target = profiles.target.copy()
        self.h0_modification = profiles.h0_modification.copy()
        self.baseline_deposition = profiles.baseline_deposition.copy()

    def _x_to_idx(self, x: float) -> int:
        return int(np.clip(np.searchsorted(self.x_grid, x, side="left"), 0, self.cfg.n_bins - 1))

    def _base_velocity_at(self, x: float) -> float:
        return float(self.v_base_vec[self._x_to_idx(x)])

    def _kernel_at(self, x0: float, height_for_slope: Optional[np.ndarray] = None) -> np.ndarray:
        k_nominal = kernel_gaussian(self.x_grid, x0, self.cfg.sigma, self.dx)
        if (not self.cfg.use_geometry_dependent_kernel) or height_for_slope is None:
            return k_nominal
        slope = surface_slope(height_for_slope, self.dx, self.cfg.slope_smoothing_window)
        eta = geometry_efficiency_from_slope(slope, self.cfg.geometry_kernel_p)
        return (eta * k_nominal).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        delta = (self.height - self.target).astype(np.float32)
        x_norm = np.float32(self.nozzle_x / self.cfg.x_max)
        remaining = np.float32(1.0 - x_norm)
        local_base_v = np.float32(self._base_velocity_at(self.nozzle_x))
        return np.concatenate(
            [self.height, delta, np.array([x_norm, remaining, self.velocity, local_base_v], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._set_profiles()
        self.height = self.h0_modification.copy()
        self.nozzle_x = 0.0
        self.velocity = self._base_velocity_at(0.0)
        self.step_count = 0
        self._prev_velocity = self.velocity
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
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

        h_old_for_kernel = self.height.copy()
        k_acc = np.zeros_like(self.height)
        for j in range(n_q):
            s_j = (j + 0.5) * L / n_q
            x_j = x_prev + s_j
            k_acc += self._kernel_at(x_j, height_for_slope=h_old_for_kernel)
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

        terminated = False
        truncated = self.step_count >= cfg.n_steps
        info = {"velocity": float(self.velocity), "dv": float(dv)}
        return self._get_obs(), float(reward), terminated, truncated, info

# ============================================================
# 6. LOAD MODEL
# ============================================================

def check_required_files(model_prefix: str) -> Tuple[Path, Path]:
    model_path = CHECKPOINT_DIR / f"{model_prefix}.zip"
    vecnorm_path = CHECKPOINT_DIR / f"{model_prefix}_vecnormalize.pkl"
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"CHECKPOINT_DIR does not exist: {CHECKPOINT_DIR}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize file: {vecnorm_path}")
    return model_path, vecnorm_path


def make_base_vec_env_for_loading() -> DummyVecEnv:
    def _factory():
        return Monitor(ColdSpray1DTestEnv(cfg=cfg, h0_relative_amplitude=0.03, env_seed=123))
    return DummyVecEnv([_factory])


def load_model_bundle(model_prefix: str) -> Tuple[PPO, VecNormalize]:
    model_path, vecnorm_path = check_required_files(model_prefix)
    base_venv = make_base_vec_env_for_loading()
    venv = VecNormalize.load(str(vecnorm_path), base_venv)
    venv.training = False
    venv.norm_reward = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), env=venv, device=device)

    expected_obs_dim = 2 * cfg.n_bins + 4
    if venv.observation_space.shape != (expected_obs_dim,):
        raise ValueError(f"Observation shape mismatch: got {venv.observation_space.shape}, expected {(expected_obs_dim,)}")

    print("=" * 80)
    print(f"Loaded model:        {model_path}")
    print(f"Loaded VecNormalize: {vecnorm_path}")
    print(f"Device:              {device}")
    print("=" * 80)
    return model, venv

model, venv = load_model_bundle(MODEL_PREFIX)

# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_one_rollout(model: PPO, venv: VecNormalize, cfg: ColdSpray1DConfig, h0_relative_amplitude: float, env_seed: int, deterministic: bool) -> Dict[str, Any]:
    if deterministic:
        torch.manual_seed(GLOBAL_SEED + int(env_seed))
        np.random.seed(GLOBAL_SEED + int(env_seed))
    else:
        torch.manual_seed(GLOBAL_SEED + int(env_seed) + 100000)
        np.random.seed(GLOBAL_SEED + int(env_seed) + 100000)

    env = ColdSpray1DTestEnv(cfg=cfg, h0_relative_amplitude=h0_relative_amplitude, env_seed=env_seed)
    obs, _ = env.reset()
    for _ in range(cfg.n_steps):
        obs_norm = venv.normalize_obs(obs[None, :].astype(np.float32))
        action, _ = model.predict(obs_norm, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            break

    final_height = env.height.copy()
    target = env.target.copy()
    relative_mse = compute_relative_mse_mean_target(final_height, target)
    overshoot = compute_max_overshoot(final_height, target)
    return {
        "relative_mse": float(relative_mse),
        "max_overshoot": float(overshoot),
        "seed": int(env_seed),
        "h0_relative_amplitude": float(h0_relative_amplitude),
    }


def build_eval_seed(amplitude_percent: float, run_index: int) -> int:
    amp_code = int(round(amplitude_percent * 10.0))
    return int(GLOBAL_SEED + 1000 * amp_code + run_index)

all_rows: List[Dict[str, Any]] = []
for amp_percent in AMPLITUDES_PERCENT:
    amp_rel = float(amp_percent / 100.0)
    print(f"Evaluating amplitude {amp_percent:.1f}% ({N_TESTS_PER_AMPLITUDE} runs)...")
    for run_idx in range(N_TESTS_PER_AMPLITUDE):
        env_seed = build_eval_seed(float(amp_percent), run_idx)
        out = evaluate_one_rollout(model, venv, cfg, amp_rel, env_seed, DETERMINISTIC_POLICY)
        out["amplitude_percent"] = float(amp_percent)
        out["run_index"] = int(run_idx)
        out["model_prefix"] = MODEL_PREFIX
        all_rows.append(out)

raw_df = pd.DataFrame(all_rows)
summary_df = raw_df.groupby("amplitude_percent", as_index=False).agg(
    mean_relative_mse=("relative_mse", "mean"),
    std_relative_mse=("relative_mse", "std"),
    min_relative_mse=("relative_mse", "min"),
    max_relative_mse=("relative_mse", "max"),
    mean_overshoot=("max_overshoot", "mean"),
)
summary_df["std_relative_mse"] = summary_df["std_relative_mse"].fillna(0.0)

raw_csv_path = OUT_DIR / f"gp_bumps_raw_{MODEL_PREFIX}.csv"
summary_csv_path = OUT_DIR / f"gp_bumps_summary_{MODEL_PREFIX}.csv"
raw_df.to_csv(raw_csv_path, index=False)
summary_df.to_csv(summary_csv_path, index=False)

display(summary_df)
print(f"Saved raw CSV:     {raw_csv_path}")
print(f"Saved summary CSV: {summary_csv_path}")

# ============================================================
# 8. PLOTTING
# ============================================================

def plot_gp_bumps_figure(raw_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, model_prefix: str):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.scatter(
        raw_df["amplitude_percent"].to_numpy(),
        raw_df["relative_mse"].to_numpy(),
        s=18,
        alpha=0.35,
        label="individual runs",
        edgecolors="none",
    )

    x = summary_df["amplitude_percent"].to_numpy()
    y = summary_df["mean_relative_mse"].to_numpy()
    yerr = summary_df["std_relative_mse"].to_numpy()
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=3,
        label="mean ± std",
        zorder=4,
    )

    y_floor = max(float(np.min(raw_df["relative_mse"])), 0.0)
    label_y = y_floor + 0.002
    for xi, yi in zip(x, y):
        ax.text(xi, max(label_y, yi + 0.0005), f"{yi:.4f}", fontsize=7, ha="center", va="bottom", alpha=0.95)

    ax.set_xlabel("GP amplitude (% of mean h0)")
    ax.set_ylabel("relative MSE")
    ax.set_title(
        f"Individual runs + mean ± std\n"
        f"GP amplitude (% of mean h0) vs relative MSE ({N_TESTS_PER_AMPLITUDE} eval samples each)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_xticks(np.arange(0.0, 10.0 + 1.0, 1.0))

    png_path = out_dir / f"gp_bumps_relative_mse_{model_prefix}.png"
    pdf_path = out_dir / f"gp_bumps_relative_mse_{model_prefix}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    return png_path, pdf_path


def plot_slide_style_figure(raw_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, model_prefix: str):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(100, 100)

    header_ax = fig.add_subplot(gs[0:10, :])
    header_ax.set_facecolor("#3133A8")
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(False)
    header_ax.text(0.01, 0.55, "GP-BASE CONSTRUCTION FOR TRAINING", color="white", fontsize=28, va="center", ha="left", transform=header_ax.transAxes)

    ax = fig.add_subplot(gs[18:82, 8:96])
    ax.scatter(raw_df["amplitude_percent"].to_numpy(), raw_df["relative_mse"].to_numpy(), s=18, alpha=0.35, label="individual runs", edgecolors="none")

    x = summary_df["amplitude_percent"].to_numpy()
    y = summary_df["mean_relative_mse"].to_numpy()
    yerr = summary_df["std_relative_mse"].to_numpy()
    ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.8, markersize=5, capsize=3, label="mean ± std", zorder=4)

    y_floor = max(float(np.min(raw_df["relative_mse"])), 0.0)
    label_y = y_floor + 0.002
    for xi, yi in zip(x, y):
        ax.text(xi, max(label_y, yi + 0.0005), f"{yi:.4f}", fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("GP amplitude (% of mean h0)")
    ax.set_ylabel("relative MSE")
    ax.set_title(
        f"Individual runs + mean ± std\n"
        f"GP amplitude (% of mean h0) vs relative MSE ({N_TESTS_PER_AMPLITUDE} eval samples each)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks(np.arange(0.0, 10.0 + 1.0, 1.0))

    caption_ax = fig.add_subplot(gs[86:98, :])
    caption_ax.set_xticks([])
    caption_ax.set_yticks([])
    for spine in caption_ax.spines.values():
        spine.set_visible(False)
    caption_ax.text(0.40, 0.5, "Figure:", color="#3A3AC8", fontsize=26, ha="right", va="center")
    caption_ax.text(0.41, 0.5, " Testing image for GP bumps", color="black", fontsize=26, ha="left", va="center")

    png_path = out_dir / f"gp_bumps_relative_mse_slide_style_{model_prefix}.png"
    pdf_path = out_dir / f"gp_bumps_relative_mse_slide_style_{model_prefix}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    return png_path, pdf_path

png_path, pdf_path = plot_gp_bumps_figure(raw_df, summary_df, OUT_DIR, MODEL_PREFIX)
slide_png_path, slide_pdf_path = plot_slide_style_figure(raw_df, summary_df, OUT_DIR, MODEL_PREFIX)

print("\nDONE.")
print(f"Main PNG:        {png_path}")
print(f"Main PDF:        {pdf_path}")
print(f"Slide-style PNG: {slide_png_path}")
print(f"Slide-style PDF: {slide_pdf_path}")
print(f"Raw CSV:         {raw_csv_path}")
print(f"Summary CSV:     {summary_csv_path}")