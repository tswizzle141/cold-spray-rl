from __future__ import annotations
from dataclasses import dataclass
from math import gamma
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

@dataclass
class LSQ2DConfig:
    # ================================================================
    # 1) Grid / domain definition
    # ================================================================
    # IMPORTANT MEMORY CHANGE:
    # Old default was 32. Lowering this to 24 reduces pixels from 1024 to 576.
    # Since every basis column has one value per pixel, this directly reduces
    # both disk size and compute.
    grid_n: int = 64

    # IMPORTANT MEMORY CHANGE:
    # Old default was 1, meaning the snake visited EVERY horizontal row.
    # Using 2 means it only visits every second row -> much fewer rows -> much
    # shorter total path -> many fewer optimization variables.
    spacing_px: int = 1

    # ================================================================
    # 2) RANDOM TARGET GENERATION
    # ================================================================
    n_bumps_min: int = 1
    n_bumps_max: int = 5

    bump_sigma_x_min: float = 0.05
    bump_sigma_x_max: float = 0.15
    bump_sigma_y_min: float = 0.05
    bump_sigma_y_max: float = 0.15

    bump_amp_min: float = 0.08
    bump_amp_max: float = 0.20
    target_peak_max: float = 0.50

    use_rotated_bumps: bool = True
    bump_theta_min: float = -0.8
    bump_theta_max: float = 0.8

    # ================================================================
    # 3) Deposition / process model
    # ================================================================
    feed_rate: float = 2.0
    sigma: float = 0.02
    beta: float = 1.0
    eta: float = 0.4
    rho_m: float = 6500.0
    samples_per_pixel: float = 1.0
    A_amp: float = 1.0

    # ================================================================
    # 4) Motion / action constraints
    # ================================================================
    v_max: float = 1.0
    v_min_fraction: float = 0.1

    off_max: float = 0.05
    offset_limit: float = 0.10

    # IMPORTANT MEMORY CHANGE:
    # Old default was 0.02. Smaller ds -> many more steps.
    # Using 0.04 cuts the number of time steps roughly in half.
    ds: float = 0.05

    # IMPORTANT MEMORY CHANGE:
    # Old default was 7. Fewer lateral lanes -> fewer variables.
    n_offset_lanes: int = 7

    # ================================================================
    # 5) Regularization / LSQ weighting
    # ================================================================
    lambda_mass: float = 2e-4
    lambda_step_smooth: float = 8e-3
    lambda_lane_smooth: float = 4e-3

    positive_target_weight: float = 2.5
    seed: Optional[int] = 42

    # ================================================================
    # 6) Memory / chunk controls  (NEW)
    # ================================================================
    # Each chunk processes this many variables at once when multiplying the
    # operator. Smaller chunk -> lower peak RAM, but slower runtime.
    chunk_cols: int = 128

    # Basis columns are written to a disk-backed memmap using float16.
    # This drastically reduces RAM. Values are converted to float32/64 only
    # while computing a chunk.
    basis_dtype: str = "float32"

    # Optional directory for memmap scratch files. If None, a temporary folder
    # is created automatically and deleted when the solve ends.
    scratch_dir: Optional[str] = None


# =============================================================================
# Deposition model
# =============================================================================

class DepositionModel2D:
    """
    Converts one path segment into a 2D deposited-height basis map.
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

class SnakePath2D:
    """
    Build one horizontal snake pathway over the full domain [0,1] x [0,1].
    """

    def __init__(self, cfg: LSQ2DConfig, grid_shape: Tuple[int, int]) -> None:
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

def make_random_bell_target_map(cfg: LSQ2DConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.grid_n
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    Y, X = np.meshgrid(grid, grid, indexing="ij")

    H = np.zeros((n, n), dtype=np.float64)
    n_bumps = int(rng.integers(cfg.n_bumps_min, cfg.n_bumps_max + 1))

    for _ in range(n_bumps):
        x0 = float(rng.uniform(0.0, 1.0))
        y0 = float(rng.uniform(0.0, 1.0))
        sigma_x = float(rng.uniform(cfg.bump_sigma_x_min, cfg.bump_sigma_x_max))
        sigma_y = float(rng.uniform(cfg.bump_sigma_y_min, cfg.bump_sigma_y_max))
        amp = float(rng.uniform(cfg.bump_amp_min, cfg.bump_amp_max))
        theta = float(rng.uniform(cfg.bump_theta_min, cfg.bump_theta_max)) if cfg.use_rotated_bumps else 0.0

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        dx = X - x0
        dy = Y - y0

        xr = c * dx + s * dy
        yr = -s * dx + c * dy

        bump = amp * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
        H += bump

    h_max = float(np.max(H))
    if h_max > cfg.target_peak_max and h_max > 1e-12:
        H *= cfg.target_peak_max / h_max

    return H.astype(np.float64)


def flatten_map(M: np.ndarray) -> np.ndarray:
    return np.asarray(M, dtype=np.float64).reshape(-1)


def make_pixel_weights(cfg: LSQ2DConfig, target_map: np.ndarray) -> np.ndarray:
    W = np.ones_like(target_map, dtype=np.float64)
    W[target_map > 1e-12] = cfg.positive_target_weight
    return W.reshape(-1)

class BasisBankMemmap:
    """
    Stores all basis columns in a disk-backed memmap instead of a dense RAM matrix.

    Memory benefit:
    ---------------
    The old code stored:
        A shape = (n_pixels, n_vars) in float64 RAM

    This class stores:
        basis_mm shape = (n_vars, n_pixels) on disk in float16

    During computation we only load small chunks into RAM.
    """

    def __init__(self, cfg: LSQ2DConfig, deposition: DepositionModel2D, meta: List[dict]) -> None:
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
            basis = self.deposition.basis_for_segment(m["x0"], m["y0"], m["x1"], m["y1"])
            self.mm[j, :] = basis.reshape(-1).astype(self.dtype, copy=False)
        self.mm.flush()

    def get_chunk(self, j0: int, j1: int, out_dtype=np.float32) -> np.ndarray:
        return np.asarray(self.mm[j0:j1, :], dtype=out_dtype)

    def close(self) -> None:
        # Ensure memmap is released and temp dir is cleaned up.
        try:
            if hasattr(self, "mm") and self.mm is not None:
                self.mm.flush()
                del self.mm
        finally:
            if self._tmpdir_ctx is not None:
                self._tmpdir_ctx.cleanup()

class LeastSquaresSlicer2D:
    """
    Main slicer class.

    MEMORY-IMPORTANT DIFFERENCE:
    ----------------------------
    This class no longer constructs a full dense A matrix in RAM.
    """

    def __init__(self, cfg: LSQ2DConfig) -> None:
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
        self.path_builder = SnakePath2D(cfg, (cfg.grid_n, cfg.grid_n))
        self.pts, self.cumlen = self.path_builder.build_single_snake_polyline()

        self.offset_lanes = np.linspace(
            -cfg.off_max, cfg.off_max, cfg.n_offset_lanes, dtype=np.float64
        )

    def build_fixed_steps(self) -> List[dict]:
        total_len = float(self.cumlen[-1])
        s_values = np.arange(0.0, total_len, self.cfg.ds, dtype=np.float64)
        if len(s_values) == 0 or s_values[-1] < total_len:
            s_values = np.append(s_values, total_len).astype(np.float64)

        steps: List[dict] = []
        for i in range(len(s_values) - 1):
            s0 = float(s_values[i])
            s1 = float(s_values[i + 1])

            x0, y0, tx0, ty0 = SnakePath2D.pose_at_s(self.pts, self.cumlen, s0)
            x1, y1, tx1, ty1 = SnakePath2D.pose_at_s(self.pts, self.cumlen, s1)

            tx = 0.5 * (tx0 + tx1)
            ty = 0.5 * (ty0 + ty1)
            L = float(np.hypot(tx, ty) + 1e-12)
            tx, ty = tx / L, ty / L

            nx, ny = -ty, tx

            steps.append(
                {
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

    def build_meta(self) -> Tuple[List[dict], List[dict]]:
        steps = self.build_fixed_steps()
        meta: List[dict] = []

        for step_idx, st in enumerate(steps):
            for lane_idx, lane_off in enumerate(self.offset_lanes):
                x0 = float(np.clip(st["x0"] + lane_off * st["nx"], 0.0, 1.0))
                y0 = float(np.clip(st["y0"] + lane_off * st["ny"], 0.0, 1.0))
                x1 = float(np.clip(st["x1"] + lane_off * st["nx"], 0.0, 1.0))
                y1 = float(np.clip(st["y1"] + lane_off * st["ny"], 0.0, 1.0))

                meta.append(
                    {
                        "step_idx": step_idx,
                        "lane_idx": lane_idx,
                        "lane_off": float(lane_off),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "length": float(st["length"]),
                    }
                )
        return meta, steps

    def _regularization_diag_counts(self, n_steps: int, lane_count: int) -> np.ndarray:
        """
        Count the diagonal contribution from the implicit regularization.
        This is used only to scale columns well.

        Why needed:
        -----------
        In the old dense code, column norms came from the explicit A_aug matrix.
        Since we no longer build A_aug, we reconstruct the same diagonal
        contribution analytically.
        """
        n_vars = n_steps * lane_count
        reg_diag = np.zeros(n_vars, dtype=np.float64)

        if self.cfg.lambda_mass > 0.0:
            reg_diag += self.cfg.lambda_mass

        if self.cfg.lambda_step_smooth > 0.0:
            lam = self.cfg.lambda_step_smooth
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
            lam = self.cfg.lambda_lane_smooth
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
        """
        Compute sum_i (sqrt(w_i) * A_ij)^2 for each column j, in chunks.
        """
        n_vars = basis_bank.n_vars
        out = np.zeros(n_vars, dtype=np.float64)

        for j0 in range(0, n_vars, chunk_cols):
            j1 = min(j0 + chunk_cols, n_vars)
            chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)  # shape (chunk, pix)
            weighted = chunk * pix_w[None, :]
            out[j0:j1] = np.sum(weighted.astype(np.float64) ** 2, axis=1)
        return out

    def build_operator_and_rhs(
        self,
        basis_bank: BasisBankMemmap,
        meta: List[dict],
        target_map: np.ndarray,
    ) -> Tuple[LinearOperator, np.ndarray, np.ndarray]:
        """
        Build the augmented least-squares operator WITHOUT building the matrix.

        Blocks represented implicitly:
            [ W^{1/2} A          ]
            [ sqrt(lambda_m) I   ]
            [ sqrt(lambda_s) D_t ]
            [ sqrt(lambda_l) D_l ]
        """
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
            """
            Compute y = A_aug_scaled @ z, but without ever creating A_aug_scaled.
            """
            w = z / col_scale
            parts: List[np.ndarray] = []

            # ------------------------------------------------------------
            # 1) data block = sqrt(W) * A * w
            # ------------------------------------------------------------
            y_data = np.zeros(n_pix, dtype=np.float64)
            for j0 in range(0, n_vars, chunk_cols):
                j1 = min(j0 + chunk_cols, n_vars)
                chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)  # (chunk, pix)
                y_data += chunk.T @ w[j0:j1]
            y_data *= pix_w
            parts.append(y_data)

            # ------------------------------------------------------------
            # 2) mass regularization block = sqrt(lambda_mass) * w
            # ------------------------------------------------------------
            if n_mass > 0:
                parts.append(sqrt_lm * w)

            # ------------------------------------------------------------
            # 3) step smooth block
            # ------------------------------------------------------------
            if n_step > 0:
                w2 = w.reshape(n_steps, lane_count)
                step_diff = (w2[:-1, :] - w2[1:, :]).reshape(-1)
                parts.append(sqrt_ls * step_diff)

            # ------------------------------------------------------------
            # 4) lane smooth block
            # ------------------------------------------------------------
            if n_lane > 0:
                w2 = w.reshape(n_steps, lane_count)
                lane_diff = (w2[:, :-1] - w2[:, 1:]).reshape(-1)
                parts.append(sqrt_ll * lane_diff)

            return np.concatenate(parts).astype(np.float64)

        def rmatvec(y: np.ndarray) -> np.ndarray:
            """
            Compute z = A_aug_scaled.T @ y, chunked and memory-safe.
            """
            out = np.zeros(n_vars, dtype=np.float64)
            pos = 0

            # ------------------------------------------------------------
            # 1) transpose of data block
            # ------------------------------------------------------------
            y_data = y[pos:pos + n_pix]
            pos += n_pix
            y_data_weighted = y_data * pix_w

            for j0 in range(0, n_vars, chunk_cols):
                j1 = min(j0 + chunk_cols, n_vars)
                chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)  # (chunk, pix)
                out[j0:j1] += chunk @ y_data_weighted

            # ------------------------------------------------------------
            # 2) transpose of mass regularization
            # ------------------------------------------------------------
            if n_mass > 0:
                y_mass = y[pos:pos + n_mass]
                pos += n_mass
                out += sqrt_lm * y_mass

            # ------------------------------------------------------------
            # 3) transpose of step smooth
            # ------------------------------------------------------------
            if n_step > 0:
                y_step = y[pos:pos + n_step].reshape(n_steps - 1, lane_count)
                pos += n_step
                acc = np.zeros((n_steps, lane_count), dtype=np.float64)
                acc[:-1, :] += y_step
                acc[1:, :] -= y_step
                out += sqrt_ls * acc.reshape(-1)

            # ------------------------------------------------------------
            # 4) transpose of lane smooth
            # ------------------------------------------------------------
            if n_lane > 0:
                y_lane = y[pos:pos + n_lane].reshape(n_steps, lane_count - 1)
                pos += n_lane
                acc = np.zeros((n_steps, lane_count), dtype=np.float64)
                acc[:, :-1] += y_lane
                acc[:, 1:] -= y_lane
                out += sqrt_ll * acc.reshape(-1)

            # Because solver variable is z and w = z / col_scale
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
        """
        Rebuild predicted height in chunks:
            h = A @ w

        Again, no dense A matrix is ever built.
        """
        n_pix = basis_bank.n_pix
        n_vars = basis_bank.n_vars
        chunk_cols = max(1, int(self.cfg.chunk_cols))

        h = np.zeros(n_pix, dtype=np.float64)
        for j0 in range(0, n_vars, chunk_cols):
            j1 = min(j0 + chunk_cols, n_vars)
            chunk = basis_bank.get_chunk(j0, j1, out_dtype=np.float32)
            h += chunk.T @ w[j0:j1]

        return h.reshape((self.cfg.grid_n, self.cfg.grid_n)).astype(np.float64)

    def solve_masses(self, target_map: np.ndarray) -> Dict[str, object]:
        meta, steps = self.build_meta()

        # Build disk-backed basis bank.
        basis_bank = BasisBankMemmap(self.cfg, self.deposition, meta)
        try:
            basis_bank.build()

            Aop, b_aug, col_scale = self.build_operator_and_rhs(basis_bank, meta, target_map)

            sol = lsq_linear(
                Aop,
                b_aug,
                bounds=(0.0, np.inf),
                lsmr_tol="auto",
                verbose=1,
                method="trf",
                max_iter=60,
            )

            z = sol.x.astype(np.float64)
            w = z / col_scale

            height = self.reconstruct_height_from_weights(basis_bank, w)

        finally:
            basis_bank.close()

        return {
            "w": w,
            "meta": meta,
            "steps": steps,
            "height": height,
            "target_map": target_map,
            "lsq_cost": float(sol.cost),
            "lsq_status": int(sol.status),
            "lsq_success": bool(sol.success),
        }

    def masses_to_actions(self, solve_result: Dict[str, object]) -> Dict[str, object]:
        w = np.asarray(solve_result["w"], dtype=np.float64)
        steps = solve_result["steps"]

        lane_count = self.cfg.n_offset_lanes
        n_steps = len(steps)

        masses = w.reshape(n_steps, lane_count)
        lane_offsets = self.offset_lanes[None, :]

        m_total = np.sum(masses, axis=1)
        offset_cm = np.sum(masses * lane_offsets, axis=1) / np.maximum(m_total, 1e-12)
        offset_cm = np.clip(offset_cm, -self.cfg.offset_limit, self.cfg.offset_limit)

        v_min = self.cfg.v_min_fraction * self.cfg.v_max
        velocity = self.cfg.feed_rate * self.cfg.ds / np.maximum(m_total, 1e-12)
        velocity = np.clip(velocity, v_min, self.cfg.v_max)

        dv_action = np.clip(2.0 * (velocity / self.cfg.v_max) - 1.0, -1.0, 1.0)
        abweichung_action = np.clip(offset_cm / max(self.cfg.off_max, 1e-12), -1.0, 1.0)
        actions = np.stack([dv_action, abweichung_action], axis=1).astype(np.float64)

        center_xy = np.zeros((n_steps, 2), dtype=np.float64)
        realized_xy = np.zeros((n_steps, 2), dtype=np.float64)

        for t, st in enumerate(steps):
            xc = 0.5 * (st["x0"] + st["x1"])
            yc = 0.5 * (st["y0"] + st["y1"])
            center_xy[t] = [xc, yc]

            realized_xy[t] = [
                np.clip(xc + offset_cm[t] * st["nx"], 0.0, 1.0),
                np.clip(yc + offset_cm[t] * st["ny"], 0.0, 1.0),
            ]

        return {
            **solve_result,
            "actions": actions,
            "velocity": velocity,
            "offset_cm": offset_cm,
            "mass_per_step": m_total,
            "mass_per_lane": masses,
            "center_xy": center_xy,
            "realized_xy": realized_xy,
        }

    def run(self, target_map: Optional[np.ndarray] = None) -> Dict[str, object]:
        if target_map is None:
            target_map = make_random_bell_target_map(self.cfg)
        else:
            target_map = target_map.astype(np.float64, copy=True)
        return self.masses_to_actions(self.solve_masses(target_map))

def compute_metrics(height: np.ndarray, target_map: np.ndarray) -> Dict[str, float]:
    err = height - target_map
    overshoot = np.maximum(err, 0.0)
    deficit = np.maximum(-err, 0.0)
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "l1_deficit": float(np.sum(deficit)),
        "l1_overshoot": float(np.sum(overshoot)),
        "max_height": float(np.max(height)),
        "target_max": float(np.max(target_map)),
    }


def plot_result(result: Dict[str, object], cfg: LSQ2DConfig, save_dir: Optional[Path] = None) -> None:
    target_map = np.asarray(result["target_map"], dtype=np.float64)
    height = np.asarray(result["height"], dtype=np.float64)
    center_xy = np.asarray(result["center_xy"], dtype=np.float64)
    realized_xy = np.asarray(result["realized_xy"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float64)
    velocity = np.asarray(result["velocity"], dtype=np.float64)
    offset_cm = np.asarray(result["offset_cm"], dtype=np.float64)
    mass_per_step = np.asarray(result["mass_per_step"], dtype=np.float64)
    metrics = compute_metrics(height, target_map)

    fig1 = plt.figure(figsize=(15, 10))

    ax1 = fig1.add_subplot(2, 2, 1)
    im1 = ax1.imshow(target_map, origin="lower")
    ax1.set_title("2D random bell target map")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig1.add_subplot(2, 2, 2)
    im2 = ax2.imshow(height, origin="lower")
    ax2.set_title(
        f"2D predicted height (single-pass LSQ)\nMSE={metrics['mse']:.3e}, MAE={metrics['mae']:.3e}"
    )
    fig1.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.plot(center_xy[:, 0], center_xy[:, 1], "--", linewidth=1.0, label="single snake centerline")
    ax3.plot(realized_xy[:, 0], realized_xy[:, 1], linewidth=2.0, label="solved path with abweichung")
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_aspect("equal")
    ax3.grid(alpha=0.3)
    ax3.set_title("Full-domain single-pass snake + solved abweichung")
    ax3.legend(loc="best")

    ax4 = fig1.add_subplot(2, 2, 4)
    t = np.arange(len(actions))
    ax4.plot(t, actions[:, 0], label="dv action")
    ax4.plot(t, actions[:, 1], label="abweichung action")
    ax4.plot(t, velocity, label="velocity")
    ax4.plot(t, offset_cm, label="offset (m)")
    ax4.plot(t, mass_per_step, label="mass / step")
    ax4.set_title("Recovered controls from single-pass least squares")
    ax4.set_xlabel("step")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="best")
    plt.tight_layout()

    n = target_map.shape[0]
    g = (np.arange(n, dtype=np.float64) + 0.5) / n
    X, Y = np.meshgrid(g, g, indexing="xy")

    fig2 = plt.figure(figsize=(15, 6))
    ax5 = fig2.add_subplot(1, 2, 1, projection="3d")
    ax5.plot_surface(X, Y, target_map, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax5.set_title("3D random bell target surface")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("height")

    ax6 = fig2.add_subplot(1, 2, 2, projection="3d")
    ax6.plot_surface(X, Y, height, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax6.set_title("3D single-pass LSQ predicted surface")
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_zlabel("height")
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_dir / "lsq_random_bells_2d_overview.png", dpi=180, bbox_inches="tight")
        fig2.savefig(save_dir / "lsq_random_bells_3d_surfaces.png", dpi=180, bbox_inches="tight")

    plt.show()


def main() -> None:
    """
    Demo entry point.
    """
    cfg = LSQ2DConfig()
    target_map = make_random_bell_target_map(cfg)
    slicer = LeastSquaresSlicer2D(cfg)
    result = slicer.run(target_map=target_map)
    metrics = compute_metrics(np.asarray(result["height"]), np.asarray(result["target_map"]))

    print("===== Single-pass least-squares slicer summary =====")
    print(f"steps                : {np.asarray(result['actions']).shape[0]}")
    print(f"action shape         : {np.asarray(result['actions']).shape}  (columns = [dv, abweichung])")
    print(f"lsq success          : {bool(result['lsq_success'])}")
    print(f"lsq status           : {int(result['lsq_status'])}")
    print(f"lsq cost             : {float(result['lsq_cost']):.6e}")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")

    out_dir = Path("lsq_random_bells")
    plot_result(result, cfg, save_dir=out_dir)
    print(f"Saved 2D figure to   : {out_dir / 'lsq_random_bells_2d_overview.png'}")
    print(f"Saved 3D figure to   : {out_dir / 'lsq_random_bells_3d_surfaces.png'}")


if __name__ == "__main__":
    main()