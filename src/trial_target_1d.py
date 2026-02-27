import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# 1) Gaussian Process sampler (zero-mean RBF GP)
# -------------------------------------------------
def sample_gp_1d(x, length_scale=0.15, sigma=1.0, jitter=1e-8, rng=None):
    """
    Draw a single sample from a 1D zero-mean Gaussian Process
    with an RBF (squared-exponential) kernel.

    n(x) ~ N(0, K), where
    K_ij = sigma^2 * exp(-(x_i - x_j)^2 / (2 * length_scale^2))

    Parameters
    ----------
    x : array-like
        1D input grid.
    length_scale : float
        Kernel length scale controlling smoothness.
    sigma : float
        Kernel amplitude.
    jitter : float
        Small diagonal term added for numerical stability.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    n : ndarray
        One GP sample evaluated on x.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    dx = x[:, None] - x[None, :]

    # RBF kernel matrix
    K = (sigma**2) * np.exp(-(dx**2) / (2.0 * length_scale**2))

    # Add jitter for numerical stability (Cholesky)
    K = K + jitter * np.eye(len(x))

    # Sample via Cholesky factorization
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(len(x))
    n = L @ z

    return n


# -------------------------------------------------
# 2) Add local "top" (bump) to target
# -------------------------------------------------
def add_top(y, x, center, width, height, shape="rect"):
    """
    Add a localized bump ("top") to the target function.

    Parameters
    ----------
    y : ndarray
        Current target values.
    x : ndarray
        Input grid.
    center : float
        Center position of the bump.
    width : float
        Width of the bump.
    height : float
        Maximum height of the bump.
    shape : str
        Shape type: "rect", "tri", "arch", "cos", "saw".
    """
    left = center - width / 2
    right = center + width / 2

    mask = (x >= left) & (x <= right)
    if not np.any(mask):
        return y

    # Normalized local coordinate in [0,1]
    xi = (x[mask] - left) / width

    if shape == "rect":
        bump = height * np.ones_like(xi)

    elif shape == "tri":
        bump = height * (1.0 - np.abs(2.0 * xi - 1.0))

    elif shape == "arch":
        t = 2.0 * xi - 1.0
        bump = height * np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))

    elif shape == "cos":
        bump = height * 0.5 * (1.0 - np.cos(2.0 * np.pi * xi))

    elif shape == "saw":
        bump = height * xi

    else:
        raise ValueError(f"Unknown shape: {shape}")

    y_new = y.copy()
    y_new[mask] += bump

    return y_new


# -------------------------------------------------
# 3) Target generator
# -------------------------------------------------
def generate_target(x, baseline=0.0, n_tops=3, rng=None,
                    difficulty="medium", y_max=0.3):
    """
    Generate a structured target function consisting of
    multiple local bumps ("tops").

    The resulting target is constrained to:
        baseline <= y(x) <= baseline + y_max
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize target with constant baseline
    y = np.full_like(x, baseline)

    # Add n_tops random bumps
    for _ in range(n_tops):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.05, 0.2)
        height = rng.uniform(0.05, y_max * 0.8)

        shape_options = ["rect", "tri", "arch", "cos", "saw"]
        shape = rng.choice(shape_options)

        y = add_top(y, x, center, width, height, shape=shape)

    # Ensure target stays within valid range
    y = np.clip(y, baseline, baseline + y_max)

    return y


# -------------------------------------------------
# 4) Generate examples and apply GP perturbation
# -------------------------------------------------
rng = np.random.default_rng(42)
N = 400
x = np.linspace(0, 1, N)
baseline = 0.0

examples = [
    dict(n_tops=1, difficulty="easy",   gp_ls=0.22, noise_std=0.010),
    dict(n_tops=2, difficulty="easy",   gp_ls=0.18, noise_std=0.012),
    dict(n_tops=3, difficulty="medium", gp_ls=0.14, noise_std=0.015),
    dict(n_tops=4, difficulty="medium", gp_ls=0.12, noise_std=0.018),
    dict(n_tops=5, difficulty="hard",   gp_ls=0.10, noise_std=0.022),
    dict(n_tops=7, difficulty="hard",   gp_ls=0.08, noise_std=0.028),
]

metrics = []  # store per-example stats for a final summary (optional)
for i, cfg in enumerate(examples, start=1):

    y_min, y_max = 0.0, 0.3

    # Generate structured target
    target = generate_target(
        x,
        baseline=baseline,
        n_tops=cfg["n_tops"],
        rng=rng,
        difficulty=cfg["difficulty"],
        y_max=y_max
    )

    # Sample zero-mean GP noise
    noise = sample_gp_1d(
        x,
        length_scale=cfg["gp_ls"],
        sigma=1.0,
        rng=rng
    )

    # Enforce zero empirical mean
    noise = noise - noise.mean()

    # Normalize to unit variance and scale to desired std
    noise = noise / (noise.std() + 1e-12)
    noise = noise * cfg["noise_std"]

    # Add noise to target
    perturbed = target + noise

    # Clip final result to valid range [0, 0.3]
    perturbed = np.clip(perturbed, y_min, y_max)

    # Plot results
    plt.figure(figsize=(10, 3))
    plt.plot(x, target, label="Target (in [0, 0.3])")
    plt.plot(x, noise, label="GP noise (zero-mean)")
    plt.plot(x, perturbed, label="Perturbed (clipped to [0, 0.3])")

    plt.axhline(baseline, linestyle="--", linewidth=1, label="Baseline")
    plt.ylim(-0.10, 0.40)

    plt.title(
        f"Example {i}: tops={cfg['n_tops']}, "
        f"GP(ls={cfg['gp_ls']}), noise_std={cfg['noise_std']}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------
    # Post-plot stats (for RL velocity config)
    # ---------------------------------------------
    v_min = float(np.min(perturbed))
    v_max = float(np.max(perturbed))

    # Max per-step velocity change on the discretized grid (useful as a dv constraint)
    dv_max = float(np.max(np.abs(np.diff(perturbed))))

    # Optional: max slope dv/dx (change rate w.r.t. x). Often interpreted as acceleration-like constraint.
    dv_dx_max = float(np.max(np.abs(np.gradient(perturbed, x))))

    print(f"[Example {i}] v_min={v_min:.6f}, v_max={v_max:.6f}, dv_max={dv_max:.6f}, dv_dx_max={dv_dx_max:.6f}")

    # Optional: store for a final table-like printout
    metrics.append({
        "example": i,
        "n_tops": cfg["n_tops"],
        "difficulty": cfg["difficulty"],
        "gp_ls": cfg["gp_ls"],
        "noise_std": cfg["noise_std"],
        "v_min": v_min,
        "v_max": v_max,
        "dv_max": dv_max,
        "dv_dx_max": dv_dx_max,
    })