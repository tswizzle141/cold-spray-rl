from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from math import sqrt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


SAVE_DIR = Path("pics-ppo-9-test")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, filename: str, dpi: int = 160) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)

@dataclass
class ColdSpray1DConfig:
    n_bins: int = 256
    x_max: float = 1.0

    v_max: float = 1.5
    v_min: float = 1e-2
    base_v: float = 1.0
    delta_v_max: float = 0.5

    feed_rate: float = 10.0
    sigma: float = 0.02

    n_steps: int = 1000
    n_envs: int = 1  

    # target-base / v_base_i shape config
    min_num_bumps: int = 1
    max_num_bumps: int = 3
    bump_height_min: float = 0.25
    bump_height_max: float = 1.0
    bump_width_min: float = 0.05
    bump_width_max: float = 0.22
    plateau_width_min: float = 0.10
    plateau_width_max: float = 0.35
    transition_width_min: float = 0.015
    transition_width_max: float = 0.06
    second_bump_scale_min: float = 0.35
    second_bump_scale_max: float = 0.85
    target_min_height_ratio: float = 0.35
    target_max_height_ratio: float = 3.20

    use_random_shape_bumps: bool = True
    allowed_bump_shapes: Tuple[str, ...] = (
        "triangle",
        "trapezoid",
        "rectangle",
        "semicircle",
        "gaussian",
        "flat",
    )

    # GP noise config
    gp_noise_std: float = 5e-4
    gp_lengthscale: float = 0.08
    gp_jitter: float = 1e-8
    gp_resample_every_reset: bool = True

    action_penalty: float = 1e-3
    smoothness_penalty: float = 1e-3
    overshoot_weight: float = 10.0
    reward_scale: float = 100.0

    # model / vecnormalize path for test-only inference
    model_path: str = str(CHECKPOINT_DIR / "best_model_9.zip")
    vecnormalize_path: str = str(CHECKPOINT_DIR / "best_vecnormalize_9.pkl")


def _kernel_gaussian(x_grid: np.ndarray, x0: float, sigma: float, dx: float) -> np.ndarray:
    z = (x_grid - x0) / sigma
    w = np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * np.pi))
    w = (w * dx).astype(np.float32)
    return w


def _sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-6)
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def _triangle_bump(x: np.ndarray, center: float, half_width: float, amp: float) -> np.ndarray:
    half_width = max(float(half_width), 1e-6)
    y = 1.0 - np.abs(x - center) / half_width
    y = np.clip(y, 0.0, 1.0)
    return (amp * y).astype(np.float64)


def _rectangle_bump(x: np.ndarray, center: float, half_width: float, amp: float) -> np.ndarray:
    half_width = max(float(half_width), 1e-6)
    mask = (np.abs(x - center) <= half_width).astype(np.float64)
    return (amp * mask).astype(np.float64)

def _flat_line_bump(
    x: np.ndarray,
    center: float,
    half_width: float,
    amp: float,) -> np.ndarray:

    half_width = max(float(half_width), 1e-6)

    left = center - half_width
    right = center + half_width

    y = np.zeros_like(x, dtype=np.float64)

    mask = (x >= left) & (x <= right)

    y[mask] = amp

    return y

def _trapezoid_bump(
    x: np.ndarray,
    center: float,
    top_half_width: float,
    bottom_half_width: float,
    amp: float,
) -> np.ndarray:
    top_half_width = max(float(top_half_width), 1e-6)
    bottom_half_width = max(float(bottom_half_width), top_half_width + 1e-6)
    dist = np.abs(x - center)
    y = np.zeros_like(x, dtype=np.float64)
    y[dist <= top_half_width] = 1.0
    side_mask = (dist > top_half_width) & (dist <= bottom_half_width)
    y[side_mask] = 1.0 - (dist[side_mask] - top_half_width) / (bottom_half_width - top_half_width)
    y = np.clip(y, 0.0, 1.0)
    return (amp * y).astype(np.float64)


def _semicircle_bump(x: np.ndarray, center: float, radius: float, amp: float) -> np.ndarray:
    radius = max(float(radius), 1e-6)
    z = 1.0 - ((x - center) / radius) ** 2
    z = np.clip(z, 0.0, None)
    y = np.sqrt(z)
    return (amp * y).astype(np.float64)


def _sample_target_base_and_vbase(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray]:
    baseline_height = (cfg.feed_rate / cfg.base_v) * dx
    target_base = np.full_like(x_grid, baseline_height, dtype=np.float64)

    n_bumps = int(rng.integers(cfg.min_num_bumps, cfg.max_num_bumps))
    component_types = list(cfg.allowed_bump_shapes) if cfg.use_random_shape_bumps else ["gaussian"]

    for bump_idx in range(n_bumps):
        shape = str(rng.choice(component_types))
        amp = float(rng.uniform(cfg.bump_height_min, cfg.bump_height_max)) * baseline_height
        if bump_idx == 1:
            amp *= float(rng.uniform(cfg.second_bump_scale_min, cfg.second_bump_scale_max))

        center = float(rng.uniform(0.18, 0.82))
        width = float(rng.uniform(cfg.bump_width_min, cfg.bump_width_max))

        if shape == "gaussian":
            comp = amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)
        elif shape == "triangle":
            comp = _triangle_bump(x_grid, center=center, half_width=width, amp=amp)
        elif shape == "rectangle":
            comp = _rectangle_bump(x_grid, center=center, half_width=width, amp=amp)
        elif shape == "trapezoid":
            top_half_width = float(rng.uniform(0.35 * width, 0.75 * width))
            bottom_half_width = float(rng.uniform(max(top_half_width + 1e-6, 0.90 * width), 1.60 * width))
            comp = _trapezoid_bump(
                x_grid,
                center=center,
                top_half_width=top_half_width,
                bottom_half_width=bottom_half_width,
                amp=amp,
            )
        elif shape == "semicircle":
            comp = _semicircle_bump(x_grid, center=center, radius=width, amp=amp)
        elif shape == "flat":
            comp = _flat_line_bump(
                x_grid,
                center=center,
                half_width=width,
                amp=amp,
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

        target_base += comp

    min_height = cfg.target_min_height_ratio * baseline_height
    max_height = cfg.target_max_height_ratio * baseline_height
    target_base = np.clip(target_base, min_height, max_height)

    v_base_vec = (cfg.feed_rate * dx) / np.maximum(target_base, 1e-12)
    v_base_vec = np.clip(v_base_vec, cfg.v_min, cfg.v_max)
    target_base = (cfg.feed_rate / v_base_vec) * dx
    return v_base_vec.astype(np.float32), target_base.astype(np.float32)

def _rbf_covariance(x: np.ndarray, std: float, lengthscale: float, jitter: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    diff = x[:, None] - x[None, :]
    K = (std ** 2) * np.exp(-0.5 * (diff / lengthscale) ** 2)
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
        self.target_base = ((cfg.feed_rate / self.v_base_vec) * self.dx).astype(np.float32)
        self.base_profile = self.target_base.copy()
        self.target = self.target_base.copy()
        self.gp_noise = np.zeros_like(self.x_grid, dtype=np.float32)

        obs_dim = 2 * cfg.n_bins + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

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

        self.v_base_vec, self.target_base = _sample_target_base_and_vbase(
            rng=self.np_random,
            x_grid=self.x_grid,
            cfg=self.cfg,
            dx=self.dx,
        )
        self.base_profile = self.target_base.copy()

        if self.cfg.gp_resample_every_reset or self.step_count == 0:
            self.gp_noise = _sample_zero_mean_gp(
                rng=self.np_random,
                x_grid=self.x_grid,
                std=self.cfg.gp_noise_std,
                lengthscale=self.cfg.gp_lengthscale,
                jitter=self.cfg.gp_jitter,
            )

        self.target = np.maximum(self.target_base + self.gp_noise, 0.0).astype(np.float32)

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
        truncated = self.step_count >= cfg.n_steps
        term_reason: Optional[str] = None

        info: Dict[str, Any] = {
            "nozzle_x": float(self.nozzle_x),
            "velocity": float(self.velocity),
            "local_base_velocity": float(local_base_v),
            "dv": float(dv),
            "ds_eff": float(ds_eff),
            "dt_step": float(dt_step),
            "step_mass": float(m_step),
            "gp_noise_mean": float(np.mean(self.gp_noise)),
            "gp_noise_std": float(np.std(self.gp_noise)),
            "gp_lengthscale": float(self.cfg.gp_lengthscale),
            "mse": float(mse),
            "max_overshoot": float(ov_max),
            "termination_reason": term_reason,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def make_test_venv(cfg: ColdSpray1DConfig) -> VecNormalize:
    def make_env():
        return Monitor(ColdSpray1DEnv(cfg))

    base_venv = DummyVecEnv([make_env])

    vec_path = Path(cfg.vecnormalize_path)
    if not vec_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vec_path}")

    venv = VecNormalize.load(str(vec_path), base_venv)

    venv.training = False
    venv.norm_reward = False
    return venv

def load_trained_model(cfg: ColdSpray1DConfig, venv: VecNormalize) -> PPO:
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[load_trained_model] Loading model from: {model_path}")
    model = PPO.load(
        str(model_path),
        env=venv,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


def rollout_model(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    deterministic: bool,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,):

    env = ColdSpray1DEnv(cfg)
    obs, _ = env.reset(seed=seed)

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

    final_mse = float(mses[-1]) if len(mses) > 0 else float("nan")
    final_ov_max = float(ov_maxs[-1]) if len(ov_maxs) > 0 else float("nan")
    total_reward = float(np.sum(rewards)) if len(rewards) > 0 else 0.0

    return {
        "x_grid": env.x_grid.copy(),
        "target": env.target.copy(),
        "target_base": env.target_base.copy(),
        "v_base_vec": env.v_base_vec.copy(),
        "base_profile": env.base_profile.copy(),
        "heights": np.stack(heights, axis=0),
        "xs": np.array(xs, dtype=np.float32),
        "vs": np.array(vs, dtype=np.float32),
        "mses": np.array(mses, dtype=np.float32),
        "ov_maxs": np.array(ov_maxs, dtype=np.float32),
        "losses": np.array(losses, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "reasons": reasons,
        "seed": seed,
        "final_mse": final_mse,
        "final_ov_max": final_ov_max,
        "total_reward": total_reward,
    }


def plot_rollouts_side_by_side(
    data_left: Dict[str, Any],
    data_right: Dict[str, Any],
    title_left: str,
    title_right: str,
    save_name: Optional[str] = None,
):
    x_grid = data_left["x_grid"]
    target = data_left["target"]
    target_base = data_left.get("target_base", data_left.get("base_profile", None))

    h_left = data_left["heights"][-1]
    h_right = data_right["heights"][-1]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for ax, h, title in [(axes[0], h_left, title_left), (axes[1], h_right, title_right)]:
        ax.plot(x_grid, h, linewidth=2, label="height")
        ax.plot(x_grid, target, linewidth=2, alpha=0.6, label="target")
        if target_base is not None:
            ax.plot(x_grid, target_base, linewidth=2, alpha=0.6, label="target_base")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("height")
        ax.legend(loc="upper right")

    r = data_right.get("rewards", None)
    t = None
    if r is not None and len(r) > 0:
        t = np.arange(len(r), dtype=np.float32)
        axes[2].plot(t, r, linewidth=1.5)
        axes[2].set_title("Stochastic reward")
    else:
        axes[2].set_title("Stochastic reward (none)")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("reward")

    v = data_right.get("vs", None)
    if v is not None and len(v) > 0:
        t_v = np.arange(len(v), dtype=np.float32)
        axes[3].plot(t_v, v, linewidth=1.5)
        axes[3].set_title("Stochastic velocity")
    else:
        axes[3].set_title("Stochastic velocity (none)")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("v")

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_name is None:
        save_name = f"rollouts_test_{stamp}.png"
    out_path = _save_fig(fig, save_name)
    print(f"[plot_rollouts_side_by_side] Saved figure to: {out_path}")
    return out_path


def run_many_tests(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    n_tests: int = 50,
    base_seed: int = 12345,
):
    det_runs: List[Dict[str, Any]] = []
    stoch_runs: List[Dict[str, Any]] = []

    for i in range(n_tests):
        seed_i = base_seed + i

        det_data = rollout_model(model, venv, cfg, deterministic=True, seed=seed_i)
        stoch_data = rollout_model(model, venv, cfg, deterministic=False, seed=seed_i)

        det_runs.append(det_data)
        stoch_runs.append(stoch_data)

        print(
            f"[test {i + 1:02d}/{n_tests}] "
            f"det_mse={det_data['final_mse']:.8f} | "
            f"stoch_mse={stoch_data['final_mse']:.8f}"
        )

    return det_runs, stoch_runs


def summarize_runs(runs: List[Dict[str, Any]], tag: str) -> Dict[str, float]:
    final_mses = np.array([r["final_mse"] for r in runs], dtype=np.float64)
    final_ov = np.array([r["final_ov_max"] for r in runs], dtype=np.float64)
    total_rewards = np.array([r["total_reward"] for r in runs], dtype=np.float64)

    summary = {
        "n_tests": float(len(runs)),
        "mse_mean": float(np.mean(final_mses)),
        "mse_std": float(np.std(final_mses)),
        "mse_min": float(np.min(final_mses)),
        "mse_max": float(np.max(final_mses)),
        "ov_mean": float(np.mean(final_ov)),
        "ov_std": float(np.std(final_ov)),
        "reward_mean": float(np.mean(total_rewards)),
        "reward_std": float(np.std(total_rewards)),
    }

    print(f"===== {tag} summary over {len(runs)} tests =====")
    for k, v in summary.items():
        if k == "n_tests":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.8f}")
    return summary


def plot_mse_hist_comparison(
    det_runs: List[Dict[str, Any]],
    stoch_runs: List[Dict[str, Any]],
    save_name: str = "mse_hist_50_tests.png",
):
    det_mse = np.array([r["final_mse"] for r in det_runs], dtype=np.float64)
    stoch_mse = np.array([r["final_mse"] for r in stoch_runs], dtype=np.float64)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(det_mse, bins=15, alpha=0.6, label="deterministic")
    plt.hist(stoch_mse, bins=15, alpha=0.6, label="stochastic")
    plt.xlabel("final mse")
    plt.ylabel("count")
    plt.title("Final MSE over 50 tests")
    plt.legend()
    plt.tight_layout()
    out_path = _save_fig(fig, save_name)
    print(f"[plot_mse_hist_comparison] Saved figure to: {out_path}")
    return out_path


def save_all_test_plots(
    det_runs: List[Dict[str, Any]],
    stoch_runs: List[Dict[str, Any]],
    save_dir_name: str = "all_50_tests",):

    out_dir = SAVE_DIR / save_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    n_tests = min(len(det_runs), len(stoch_runs))

    for i in range(n_tests):
        save_name = str(Path(save_dir_name) / f"test_{i + 1:02d}_seed_{det_runs[i]['seed']}.png")
        out_path = plot_rollouts_side_by_side(
            det_runs[i],
            stoch_runs[i],
            f"Deterministic (test #{i + 1})",
            f"Stochastic (test #{i + 1})",
            save_name=save_name,
        )
        saved_paths.append(out_path)

    print(f"[save_all_test_plots] Saved {len(saved_paths)} plot files in: {out_dir}")
    return saved_paths

if __name__ == "__main__":
    cfg = ColdSpray1DConfig(
        n_bins=256,
        x_max=1.0,
        v_max=2.0,
        v_min=1e-2,
        base_v=0.5,
        delta_v_max=1.0,
        feed_rate=2.0,
        sigma=0.02,
        n_steps=64,
        n_envs=1,
        gp_noise_std=2e-3,
        gp_lengthscale=1.0,
        gp_jitter=1e-8,
        action_penalty=0.1,
        smoothness_penalty=10.0,
        overshoot_weight=10.0,
        reward_scale=10.0,
        model_path=str("/netscratch/nham/checkpoints/best_model_9.zip"),
        vecnormalize_path=str("/netscratch/nham/checkpoints/best_vecnormalize_9.pkl"),
        use_random_shape_bumps=True,
        allowed_bump_shapes=("triangle", "trapezoid", "rectangle", "semicircle", "flat"),
    )

    N_TESTS = 100
    BASE_SEED = 12345

    venv = make_test_venv(cfg)
    model = load_trained_model(cfg, venv)

    det_runs, stoch_runs = run_many_tests(
        model=model,
        venv=venv,
        cfg=cfg,
        n_tests=N_TESTS,
        base_seed=BASE_SEED,
    )

    det_summary = summarize_runs(det_runs, tag="Deterministic")
    stoch_summary = summarize_runs(stoch_runs, tag="Stochastic")

    all_plot_paths = save_all_test_plots(
        det_runs,
        stoch_runs,
        save_dir_name="all_50_tests_random_shapes" if cfg.use_random_shape_bumps else "all_50_tests_gaussian",
    )
    print(f"Saved per-test images: {len(all_plot_paths)}")