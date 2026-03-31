#!pip install "stable-baselines3[extra]" torch gymnasium

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

SAVE_DIR = Path("pics-ppo-3")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
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
    v_min: float = 1e-2
    base_v: float = 1.0
    delta_v_max: float = 0.5

    feed_rate: float = 2.0
    sigma: float = 0.02

    n_steps: int = 1000
    n_envs: int = 4

    # ===== GP-based target construction =====
    # ONE fixed GP target_base and ONE fixed GP target modification
    # for the WHOLE training process.
    v_base_gp_amplitude: float = 0.25
    target_mod_gp_amplitude: float = 2e-3

    gp_std: float = 1.0
    gp_lengthscale: float = 0.25
    gp_jitter: float = 1e-8

    # Keep this False because user does NOT want per-episode resampling.
    gp_resample_every_reset: bool = False

    action_penalty: float = 0.1
    smoothness_penalty: float = 10.0
    overshoot_weight: float = 10.0
    reward_scale: float = 10.0

    # PPO hyperparameters
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

    # ===== resume-training config =====
    resume: bool = True
    resume_model_path: str = str(CHECKPOINT_DIR / "best_model_3.zip")
    resume_vecnormalize_path: str = str(CHECKPOINT_DIR / "best_model_3_vecnormalize.pkl")
    save_freq: int = 50000


@dataclass
class FixedGPProfiles:
    """
    One shared GP construction used by ALL envs and ALL rollouts.

    g_base -> constructs v_base(x)
    g_mod  -> constructs target_modification(x)
    """
    x_grid: np.ndarray
    v_base_vec: np.ndarray
    target_base: np.ndarray
    target: np.ndarray
    target_modulation: np.ndarray
    g_base: np.ndarray
    g_mod: np.ndarray


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
    gp = gp - np.mean(gp)
    return gp.astype(np.float32)


def build_fixed_gp_profiles(
    cfg: ColdSpray1DConfig,
    seed: int = 0,
) -> FixedGPProfiles:
    """
    Sample ONCE:
      - one GP for target_base construction
      - one GP noise / target_modification with mean 0

    Then reuse this same pair for the whole training process.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(0.0, cfg.x_max, cfg.n_bins, dtype=np.float32)
    dx = float(x_grid[1] - x_grid[0])

    # GP for base velocity
    g_base = _sample_zero_mean_gp(
        rng=rng,
        x_grid=x_grid,
        std=cfg.gp_std,
        lengthscale=cfg.gp_lengthscale,
        jitter=cfg.gp_jitter,
    )

    v_base_raw = float(cfg.base_v) + float(cfg.v_base_gp_amplitude) * g_base.astype(np.float64)
    v_base_vec = np.clip(v_base_raw, cfg.v_min, cfg.v_max).astype(np.float32)

    # target_base derived from v_base_vec
    target_base = (cfg.feed_rate * dx / np.maximum(v_base_vec, 1e-12)).astype(np.float32)

    # GP noise / target modification with mean 0
    g_mod = _sample_zero_mean_gp(
        rng=rng,
        x_grid=x_grid,
        std=cfg.gp_std,
        lengthscale=cfg.gp_lengthscale,
        jitter=cfg.gp_jitter,
    )

    target_modulation = (float(cfg.target_mod_gp_amplitude) * g_mod).astype(np.float32)
    target_modulation = (target_modulation - np.mean(target_modulation)).astype(np.float32)

    target = np.maximum(target_base + target_modulation, 0.0).astype(np.float32)

    return FixedGPProfiles(
        x_grid=x_grid.copy(),
        v_base_vec=v_base_vec.copy(),
        target_base=target_base.copy(),
        target=target.copy(),
        target_modulation=target_modulation.copy(),
        g_base=g_base.copy(),
        g_mod=g_mod.copy(),
    )


class ColdSpray1DEnv(gym.Env):
    """
    1D cold-spray environment with fixed arc-length stepping.

    IMPORTANT:
    This environment uses ONE shared fixed GP profile bundle
    (target_base + zero-mean target_modification) across the whole training.
    No per-episode resampling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: ColdSpray1DConfig,
        render_mode: Optional[str] = None,
        fixed_profiles: Optional[FixedGPProfiles] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        if fixed_profiles is None:
            raise ValueError(
                "ColdSpray1DEnv now requires fixed_profiles so that all episodes "
                "share the same GP target_base and GP target_modification."
            )

        self.fixed_profiles = fixed_profiles

        self.x_grid = self.fixed_profiles.x_grid.copy()
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.ds = float(cfg.x_max / cfg.n_steps)

        # Shared fixed profiles
        self.v_base_vec = self.fixed_profiles.v_base_vec.copy()
        self.target_base = self.fixed_profiles.target_base.copy()
        self.base_profile = self.target_base.copy()
        self.target = self.fixed_profiles.target.copy()
        self.target_modulation = self.fixed_profiles.target_modulation.copy()
        self.g_base = self.fixed_profiles.g_base.copy()
        self.g_mod = self.fixed_profiles.g_mod.copy()

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

        # No GP resampling here.
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

        #mse form
        err_old = h_old - self.target
        err_new = h_new - self.target

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

        info: Dict[str, Any] = {
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


def make_venv(
    cfg: ColdSpray1DConfig,
    fixed_profiles: FixedGPProfiles,
    vecnormalize_path: Optional[str] = None,
) -> VecNormalize:
    def make_env():
        return Monitor(ColdSpray1DEnv(cfg, fixed_profiles=fixed_profiles))

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
        fixed_profiles: FixedGPProfiles,
        start_at: int = 0,
        max_plots: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.every_episodes = int(every_episodes)
        self.venv = venv
        self.cfg = cfg
        self.fixed_profiles = fixed_profiles
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

                    det = rollout_model(
                        self.model, self.venv, self.cfg, self.fixed_profiles, deterministic=True
                    )
                    stoch = rollout_model(
                        self.model, self.venv, self.cfg, self.fixed_profiles, deterministic=False
                    )

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
        fixed_profiles: FixedGPProfiles,
        check_freq: int = 10000,
        save_prefix: str = "best_model_3",
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.venv = venv
        self.cfg = cfg
        self.fixed_profiles = fixed_profiles
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
            self.fixed_profiles,
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
                f"at timestep = {self.num_timesteps}. Overwritten {model_path}"
            )

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
    fixed_profiles: FixedGPProfiles,
    deterministic: bool,
    max_steps: Optional[int] = None,
):
    """
    Use the SAME fixed GP profiles as training.
    No new GP sample is created here.
    """
    env = ColdSpray1DEnv(cfg, fixed_profiles=fixed_profiles)
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

    _save_fig(fig, "rollout_snapshot.png")
    return anim


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
        # vs has length T+1 while rewards has length T
        tv = np.arange(len(v), dtype=np.float32)
        axes[3].plot(tv, v, linewidth=1.5)
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
        sigma=0.05,
        n_steps=64,
        n_envs=8,
        v_base_gp_amplitude=0.05,
        target_mod_gp_amplitude=5e-3,
        gp_std=1.0,
        gp_lengthscale=0.25,
        gp_jitter=1e-8,
        gp_resample_every_reset=False,
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

    # Build ONE shared GP bundle for the whole train/eval process.
    fixed_profiles = build_fixed_gp_profiles(cfg, seed=0)

    print("[GP setup] Shared fixed GP profiles created once for whole training.")
    print(f"[GP setup] mean(g_base) = {np.mean(fixed_profiles.g_base):.6e}")
    print(f"[GP setup] mean(g_mod) = {np.mean(fixed_profiles.g_mod):.6e}")
    print(f"[GP setup] mean(target_modulation) = {np.mean(fixed_profiles.target_modulation):.6e}")

    venv = make_venv(
        cfg,
        fixed_profiles=fixed_profiles,
        vecnormalize_path=cfg.resume_vecnormalize_path if cfg.resume else None,
    )
    model = load_or_create_model(cfg, venv, seed=0)

    untrained_det = rollout_model(model, venv, cfg, fixed_profiles, deterministic=True)
    untrained_stoch = rollout_model(model, venv, cfg, fixed_profiles, deterministic=False)
    plot_rollouts_side_by_side(
        untrained_det,
        untrained_stoch,
        "Untrained (deterministic)",
        "Untrained (stochastic)",
        save_name="untrained.png",
    )

    viz_cb = VisualizeEveryNEpisodesCallback(
        every_episodes=500,
        venv=venv,
        cfg=cfg,
        fixed_profiles=fixed_profiles,
        start_at=0,
        max_plots=None,
    )

    best_mse_cb = SaveBestMSECallback(
        venv=venv,
        cfg=cfg,
        fixed_profiles=fixed_profiles,
        check_freq=50000,
        save_prefix="best_model_3",
        deterministic=True,
        verbose=1,
    )

    model = train_ppo(
        model,
        total_timesteps=cfg.total_timesteps,
        callback=[viz_cb, best_mse_cb],
    )

    model.save(CHECKPOINT_DIR / "last_model_3")
    venv.save(CHECKPOINT_DIR / "last_vecnormalize_3.pkl")

    trained_det = rollout_model(model, venv, cfg, fixed_profiles, deterministic=True)
    trained_stoch = rollout_model(model, venv, cfg, fixed_profiles, deterministic=False)
    plot_rollouts_side_by_side(
        trained_det,
        trained_stoch,
        "Trained (deterministic)",
        "Trained (stochastic)",
        save_name="trained.png",
    )