from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from math import sqrt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

SAVE_DIR = Path("/netscratch/nham/logs/pics-ppo-9")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("/netscratch/nham/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, filename: str, dpi: int = 160) -> str:
    """
    Save a matplotlib figure to SAVE_DIR, close it to free memory, return full path as str.
    """
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
    v_min: float = 1e-2  # clamp to avoid infinite dwell / stopping degeneracy
    base_v: float = 1.0
    delta_v_max: float = 0.5  # action scales to +/- around local base_v_i

    feed_rate: float = 10.0
    sigma: float = 0.02

    n_steps: int = 1000
    n_envs: int = 4

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

    # ===== GP noise config =====
    gp_noise_std: float = 5e-4          
    gp_lengthscale: float = 0.08        
    gp_jitter: float = 1e-8             
    gp_resample_every_reset: bool = True

    action_penalty: float = 1e-3
    smoothness_penalty: float = 1e-3
    overshoot_weight: float = 10.0
    reward_scale: float = 100.0

    # PPO hyperparameters
    ppo_n_steps: int = 8192
    ppo_batch_size: int = 64
    ppo_gamma: float = 1.0
    ppo_gae_lambda: float = 0.95
    ppo_learning_rate: float = 3e-4
    ppo_clip_range: float = 0.2
    ppo_target_kl: float = 0.02
    ppo_n_epochs: int = 4
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_log_std_init: float = -3.0
    total_timesteps: int = 1000000

     # ===== CHANGED: resume-training config =====
    resume: bool = True
    resume_model_path: str = str(CHECKPOINT_DIR / "best_model_9.zip")
    resume_vecnormalize_path: str = str(CHECKPOINT_DIR / "best_vecnormalize_9.pkl")
    save_freq: int = 50000

def _kernel_gaussian(x_grid: np.ndarray, x0: float, sigma: float, dx: float) -> np.ndarray:
    z = (x_grid - x0) / sigma
    w = np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * np.pi))
    w = (w * dx).astype(np.float32)
    return w

def _sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-6)
    return 1.0 / (1.0 + np.exp(-(x - center) / width))

def _sample_target_base_and_vbase(
    rng: np.random.Generator,
    x_grid: np.ndarray,
    cfg: ColdSpray1DConfig,
    dx: float,) -> Tuple[np.ndarray, np.ndarray]:

    baseline_height = (cfg.feed_rate / cfg.base_v) * dx
    target_base = np.full_like(x_grid, baseline_height, dtype=np.float64)

    n_bumps = int(rng.integers(cfg.min_num_bumps, cfg.max_num_bumps + 1))
    #component_types = ["gaussian", "plateau", "sigmoid_up", "sigmoid_down"]
    component_types = ["gaussian"]

    for bump_idx in range(n_bumps):
        shape = rng.choice(component_types)

        amp = float(rng.uniform(cfg.bump_height_min, cfg.bump_height_max)) * baseline_height
        if bump_idx == 1:
            amp *= float(rng.uniform(cfg.second_bump_scale_min, cfg.second_bump_scale_max))

        if shape == "gaussian":
            center = float(rng.uniform(0.18, 0.82))
            width = float(rng.uniform(cfg.bump_width_min, cfg.bump_width_max))
            comp = amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)

        elif shape == "plateau":
            left = float(rng.uniform(0.08, 0.62))
            plateau_width = float(rng.uniform(cfg.plateau_width_min, cfg.plateau_width_max))
            right = min(0.95, left + plateau_width)
            trans = float(rng.uniform(cfg.transition_width_min, cfg.transition_width_max))
            comp = amp * (_sigmoid(x_grid, left, trans) - _sigmoid(x_grid, right, trans))

        elif shape == "sigmoid_up":
            center = float(rng.uniform(0.18, 0.82))
            trans = float(rng.uniform(cfg.transition_width_min, cfg.transition_width_max))
            comp = amp * _sigmoid(x_grid, center, trans)

        elif shape == "sigmoid_down":
            center = float(rng.uniform(0.18, 0.82))
            trans = float(rng.uniform(cfg.transition_width_min, cfg.transition_width_max))
            comp = amp * (1.0 - _sigmoid(x_grid, center, trans))

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
    """
    Covariance matrix of GP with kernel RBF:
        k(x_i, x_j) = std^2 * exp(-(x_i - x_j)^2 / (2 * lengthscale^2))
    """
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
    """
    Sample a vector GP mean=0 on x_grid
    """
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

        # ===== CHANGED: base_profile constant -> vector v_base_vec + target_base =====
        self.v_base_vec = np.full_like(self.x_grid, fill_value=cfg.base_v, dtype=np.float32)
        self.target_base = ((cfg.feed_rate / self.v_base_vec) * self.dx).astype(np.float32)
        self.base_profile = self.target_base.copy()  # giữ tên cũ để plot/debug không vỡ code cũ
        self.target = self.target_base.copy()
        self.gp_noise = np.zeros_like(self.x_grid, dtype=np.float32)

        # Observation: [h, h - t, x_norm, remaining_norm, v, local_base_v]
        obs_dim = 2 * cfg.n_bins + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action is scalar in [-1, 1] (mapped to dv and then v around local base velocity).
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

    # ===== CHANGED: local base velocity lookup =====
    def _x_to_idx(self, x: float) -> int:
        return int(np.clip(np.searchsorted(self.x_grid, x, side="left"), 0, self.cfg.n_bins - 1))

    def _base_velocity_at(self, x: float) -> float:
        return float(self.v_base_vec[self._x_to_idx(x)])

    def _kernel_at(self, x0: float) -> np.ndarray:
        """Convenience wrapper for the Gaussian kernel at position x0."""
        return _kernel_gaussian(self.x_grid, x0, self.cfg.sigma, self.dx)

    def _get_obs(self) -> np.ndarray:
        """Assemble observation vector.

        obs = [h, h - t, x_norm, remaining_norm, v, local_base_v]
        """
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
        """Advance the environment by one fixed-distance step.

        1) Map action -> dv -> v.
        2) Advance nozzle by ds_eff along the path.
        3) Compute dt_step = ds_eff / v and m_step = feed_rate * dt_step.
        4) Deposit mass by quadrature along the traversed segment.
        5) Compute progress reward (loss_old - loss_new) plus penalties.
        """

        cfg = self.cfg
        self.step_count += 1

        # ===== CHANGED: action now perturbs local base_v_i instead of a single constant base_v =====
        local_base_v = self._base_velocity_at(self.nozzle_x)
        dv = float(np.clip(action[0], -1.0, 1.0)) * cfg.delta_v_max
        v = float(np.clip(local_base_v + dv, cfg.v_min, cfg.v_max))
        self.velocity = v

        # Fixed arc-length step (clamp near boundary).
        x_prev = self.nozzle_x
        ds_eff = float(min(self.ds, cfg.x_max - self.nozzle_x))
        self.nozzle_x = float(min(cfg.x_max, self.nozzle_x + ds_eff))

        # Deposition physics.
        dt_step = ds_eff / v if ds_eff > 0.0 else 0.0
        m_step = cfg.feed_rate * dt_step

        # Quadrature of kernel along segment [x_prev, x_prev + ds_eff].
        # k_seg ≈ (1/n_q) * Σ_j K(x_grid; x_prev + s_j).
        L = ds_eff
        n_q = max(2, int(np.ceil(L / self.dx))) if L > 0.0 else 2
        k_acc = np.zeros_like(self.height)
        for j in range(n_q):
            s_j = (j + 0.5) * L / n_q
            x_j = x_prev + s_j
            k_acc += self._kernel_at(x_j)
        k_seg = k_acc / float(n_q)
        delta_h = (m_step * k_seg).astype(np.float32)

        # Apply deposition.
        h_old = self.height.copy()
        h_new = h_old + delta_h
        self.height = h_new

        # Loss components: deficit and overshoot.
        def_old = np.maximum(self.target - h_old, 0.0)
        def_new = np.maximum(self.target - h_new, 0.0)
        ov_old = np.maximum(h_old - self.target, 0.0)
        ov_new = np.maximum(h_new - self.target, 0.0)

        # Progress reward (potential-difference).
        loss_old = float(np.sum(def_old) + cfg.overshoot_weight * np.sum(ov_old))
        loss_new = float(np.sum(def_new) + cfg.overshoot_weight * np.sum(ov_new))
        reward = (loss_old - loss_new) * cfg.reward_scale

        # Control effort and smoothness penalties.
        reward -= cfg.action_penalty * (dv / cfg.delta_v_max) ** 2
        dv_smooth = (v - self._prev_velocity) / (cfg.v_max + 1e-12)
        reward -= cfg.smoothness_penalty * (dv_smooth * dv_smooth)
        self._prev_velocity = v

        # Global metrics for logging only.
        err = self.height - self.target
        mse = float(np.mean(err * err))
        ov_max = float(np.max(np.maximum(err, 0.0)))

        terminated = False
        truncated = False
        term_reason: Optional[str] = None

        # Fixed horizon termination via truncation.
        if self.step_count >= cfg.n_steps:
            truncated = True

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

def make_venv(
    cfg: ColdSpray1DConfig,
    vecnormalize_path: Optional[str] = None,
) -> VecNormalize:
    def make_env():
        return Monitor(ColdSpray1DEnv(cfg))

    base_venv = DummyVecEnv([make_env for _ in range(cfg.n_envs)])

    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        venv = VecNormalize.load(vecnormalize_path, base_venv)
        venv.training = True
        venv.norm_reward = True
        print(f"[make_venv] Loaded VecNormalize from: {vecnormalize_path}")
        return venv

    venv = VecNormalize(base_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv

def make_ppo(cfg: ColdSpray1DConfig, venv: VecNormalize, seed: int = 0) -> PPO:
    """Construct PPO with hyperparameters from config.

    Notes on key PPO parameters:
    - n_steps, batch_size: rollout length and minibatch size.
    - gamma, gae_lambda: discount and GAE smoothing (gamma=1 preserves telescoping reward).
    - clip_range, target_kl: PPO trust-region controls.
    - n_epochs: number of optimization epochs per update.
    - log_std_init: initial exploration scale for continuous actions.
    """

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

def load_or_create_model(cfg: ColdSpray1DConfig, venv: VecNormalize, seed: int = 0) -> PPO:
    model_path = Path(cfg.resume_model_path)

    if cfg.resume and model_path.exists():
        print(f"[load_or_create_model] Resuming from: {model_path}")
        return PPO.load(
            model_path,
            env=venv,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print("[load_or_create_model] No checkpoint found -> train from scratch")
    return make_ppo(cfg, venv, seed=seed)

class VisualizeEveryNEpisodesCallback(BaseCallback):
    def __init__(
        self,
        every_episodes: int,
        venv: VecNormalize,
        cfg: ColdSpray1DConfig,
        start_at: int = 0,
        max_plots: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.every_episodes = int(every_episodes)
        self.venv = venv
        self.cfg = cfg
        self.start_at = int(start_at)
        self.max_plots = None if max_plots is None else int(max_plots)

        self.episode_count = 0
        self._next_trigger = self._compute_first_trigger()
        self._plots_done = 0

    def _compute_first_trigger(self) -> int:
        if self.every_episodes <= 0:
            raise ValueError("every_episodes must be > 0")
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

                if self.episode_count >= self._next_trigger:
                    if self.max_plots is not None and self._plots_done >= self.max_plots:
                        self._next_trigger += self.every_episodes
                        continue

                    det = rollout_model(self.model, self.venv, self.cfg, deterministic=True)
                    stoch = rollout_model(self.model, self.venv, self.cfg, deterministic=False)

                    plot_rollouts_side_by_side(
                        det,
                        stoch,
                        f"After {self.episode_count} episodes (deterministic)",
                        f"After {self.episode_count} episodes (stochastic)",
                        save_name=f"after_ep_{self.episode_count:08d}.png",
                    )

                    self._plots_done += 1
                    self._next_trigger += self.every_episodes
        return True

class SaveBestMSECallback(BaseCallback):
    def __init__(
        self,
        venv: VecNormalize,
        cfg: ColdSpray1DConfig,
        check_freq: int = 10000,
        save_prefix: str = "best_model_mse",
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
        self.best_step = 0

    def _evaluate_once(self) -> float:
        data = rollout_model(
            self.model,
            self.venv,
            self.cfg,
            deterministic=self.deterministic,
        )

        mses = data["mses"]
        if len(mses) == 0:
            return np.inf
        return float(mses[-1])

    def _save_best(self, mse: float):
        model_path = CHECKPOINT_DIR / f"{self.save_prefix}"
        vecnorm_path = CHECKPOINT_DIR / f"{self.save_prefix}_vecnormalize.pkl"

        self.model.save(model_path)
        self.venv.save(vecnorm_path)

        if self.verbose > 0:
            print(
                f"[SaveBestMSECallback] New best MSE = {mse:.8f} "
                f"at timestep = {self.num_timesteps}. Overwritten {model_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        mse = self._evaluate_once()

        if self.verbose > 0:
            print(
                f"[SaveBestMSECallback] Eval at timestep {self.num_timesteps}, "
                f"MSE = {mse:.8f}, best = {self.best_mse:.8f}"
            )

        if mse < self.best_mse:
            self.best_mse = mse
            self.best_step = self.num_timesteps
            self._save_best(mse)

        return True

def train_ppo(model: PPO, total_timesteps: int, callback=None) -> PPO:
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )
    return model

def rollout_model(
    model: PPO,
    venv: VecNormalize,
    cfg: ColdSpray1DConfig,
    deterministic: bool,
    max_steps: Optional[int] = None,):

    env = ColdSpray1DEnv(cfg)
    obs, _ = env.reset()

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
    }

def animate_rollout(data: Dict[str, Any], interval_ms: int = 25):
    x_grid = data["x_grid"]
    target = data["target"]
    target_base = data.get("target_base", data.get("base_profile", None))
    heights = data["heights"]
    xs = data["xs"]
    vs = data["vs"]

    mses = data["mses"]
    ov_maxs = data["ov_maxs"]
    losses = data["losses"]
    rewards = data["rewards"]
    reasons = data["reasons"]

    fig, ax = plt.subplots(figsize=(10, 4))
    (line_h,) = ax.plot(x_grid, heights[0], linewidth=2, label="height")
    (line_t,) = ax.plot(x_grid, target, linewidth=2, alpha=0.6, label="target")
    if target_base is not None:
        ax.plot(x_grid, target_base, linewidth=2, alpha=0.6, label="target_base")
    nozzle_line = ax.axvline(xs[0], linestyle="--", linewidth=2, label="nozzle")

    ax.set_xlim(0.0, float(x_grid[-1]))
    ax.set_ylim(0.0, float(max(np.max(target), np.max(heights)) + 0.1))
    ax.set_xlabel("x")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")

    title = ax.set_title("")

    def update(frame: int):
        h = heights[frame]
        line_h.set_ydata(h)
        nozzle_line.set_xdata([xs[frame], xs[frame]])

        ymax = float(max(np.max(h), np.max(target))) + 0.1
        if ymax > ax.get_ylim()[1] * 0.95:
            ax.set_ylim(0.0, ymax)

        if frame == 0:
            title.set_text(f"step=0000  x={xs[0]:.3f}  v={vs[0]:.3f}")
        else:
            i = frame - 1
            reason = reasons[frame]
            title.set_text(
                f"step={frame:04d}  x={xs[frame]:.3f}  v={vs[frame]:.3f}  "
                f"mse={mses[i]:.6f}  ov_max={ov_maxs[i]:.3f}  loss={losses[i]:.6f}  r={rewards[i]:.3f}"
                + (f"  [{reason}]" if reason else "")
            )

        return line_h, nozzle_line, title

    anim = FuncAnimation(
        fig,
        update,
        frames=heights.shape[0],
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()

    last_frame = heights.shape[0] - 1
    update(last_frame)

    _save_fig(fig, f"rollout_snapshot.png")
    return

def plot_rollouts_side_by_side(
    data_left: Dict[str, Any],
    data_right: Dict[str, Any],
    title_left: str,
    title_right: str,
    save_name: Optional[str] = None,):
    
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
        if t is None or len(t) != len(v):
            t = np.arange(len(v), dtype=np.float32)
        axes[3].plot(t, v, linewidth=1.5)
        axes[3].set_title("Stochastic velocity")
    else:
        axes[3].set_title("Stochastic velocity (none)")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("v")

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_name is None:
        save_name = f"rollouts_{stamp}.png"
    _save_fig(fig, save_name)
    return

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
        n_envs=8,
        gp_noise_std=2e-3,
        gp_lengthscale=1.0,
        gp_jitter=1e-8,
        action_penalty=0.1,
        smoothness_penalty=10.0,
        overshoot_weight=10.0,
        reward_scale=10.0,
        ppo_n_steps=256,
        ppo_batch_size=64,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_learning_rate=1e-4,
        ppo_clip_range=0.1,
        ppo_target_kl=0.05,
        ppo_n_epochs=4,
        ppo_ent_coef=0.0,
        ppo_vf_coef=0.5,
        ppo_max_grad_norm=0.5,
        ppo_log_std_init=-5.0,
        total_timesteps=500000000,
    )

    venv = make_venv(
        cfg,
        vecnormalize_path=cfg.resume_vecnormalize_path if cfg.resume else None,
    )
    model = load_or_create_model(cfg, venv, seed=0)

    untrained_det = rollout_model(model, venv, cfg, deterministic=True)
    untrained_stoch = rollout_model(model, venv, cfg, deterministic=False)
    plot_rollouts_side_by_side(
        untrained_det,
        untrained_stoch,
        "Untrained (deterministic)",
        "Untrained (stochastic)",
        save_name="untrained.png",
    )

    viz_cb = VisualizeEveryNEpisodesCallback(
        every_episodes=50000,
        venv=venv,
        cfg=cfg,
        start_at=0,
        max_plots=None,
    )

    best_mse_cb = SaveBestMSECallback(
        venv=venv,
        cfg=cfg,
        check_freq=50000,
        save_prefix="best_model_9",
        deterministic=True,
        verbose=1,
    )

    model = train_ppo(
        model,
        total_timesteps=cfg.total_timesteps,
        callback=[viz_cb, best_mse_cb],
    )

    model.save(CHECKPOINT_DIR / "last_model_9")
    venv.save(CHECKPOINT_DIR / "last_vecnormalize_9.pkl")

    trained_det = rollout_model(model, venv, cfg, deterministic=True)
    trained_stoch = rollout_model(model, venv, cfg, deterministic=False)
    plot_rollouts_side_by_side(
        trained_det,
        trained_stoch,
        "Trained (deterministic)",
        "Trained (stochastic)",
        save_name="trained.png",
    )
