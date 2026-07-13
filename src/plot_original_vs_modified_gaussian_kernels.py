#!/usr/bin/env python3
"""Reproduce the original six-panel kernel comparison with print-size text."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 21,
    "axes.titlesize": 24,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 17,
    "figure.titlesize": 30,
    "lines.linewidth": 3.2,
    "axes.linewidth": 1.4,
    "savefig.dpi": 300,
})


def gaussian(x, center=0.0, sigma=0.70):
    """Gaussian normalized to unit peak, as in the comparison figure."""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def local_surface(x):
    """Fixed example profile used in the original explanatory figure."""
    left = 0.22 * np.exp(-0.5 * ((x + 1.55) / 0.42) ** 2)
    right = 0.145 * np.exp(-0.5 * ((x - 1.40) / 0.63) ** 2)
    valley = 0.10 * np.exp(-0.5 * (x / 0.20) ** 2)
    return left + right - valley


def decorate(ax):
    ax.grid(True, alpha=0.30)
    ax.set_xlim(-4.4, 4.4)
    ax.set_xlabel(r"$x$")
    ax.tick_params(length=5, width=1.0)


def main():
    x = np.linspace(-4.0, 4.0, 2001)
    dx = x[1] - x[0]
    h = local_surface(x)
    slope = np.abs(np.gradient(h, dx))
    gain, p = 7.0, 2.0
    scaled_slope = gain * slope
    eta = (1.0 + scaled_slope**2) ** (-p / 2.0)
    kernel = gaussian(x)
    modified_kernel = eta * kernel

    deposited_mass = 0.20
    nominal_final = h + deposited_mass * kernel
    modified_final = h + deposited_mass * modified_kernel
    gap = nominal_final - modified_final

    # Extra height deliberately reserves space for large text and outside legends.
    fig, axs = plt.subplots(2, 3, figsize=(24, 16.5))
    fig.suptitle("Original and Geometry-Dependent Modified Gaussian Kernels", y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.90,
                        wspace=0.22, hspace=0.52)

    ax = axs[0, 0]
    ax.plot(x, h, color="tab:blue")
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1.8)
    ax.annotate(r"nozzle center $x_0$", xy=(0, h[np.argmin(np.abs(x))]),
                xytext=(0.30, 0.117), fontsize=20,
                arrowprops=dict(arrowstyle="->", color="black", lw=2.2))
    ax.set_title(r"(a) Example local surface profile $h(x)$")
    ax.set_ylabel(r"$h(x)$")
    decorate(ax)

    ax = axs[0, 1]
    ax.plot(x, slope, color="tab:blue", label=r"$|h'(x)|$")
    ax.plot(x, scaled_slope, color="tab:orange", linestyle="--",
            label="scaled slope, gain=7")
    ax.set_title("(b) Local slope magnitude")
    ax.set_ylabel("slope magnitude")
    ax.legend(loc="upper right", framealpha=0.96)
    decorate(ax)

    ax = axs[0, 2]
    ax.plot(x, eta, color="tab:blue")
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1.8)
    ax.set_title(r"(c) Geometry factor $\eta(x)=(1+(g|h'(x)|)^2)^{-p/2}$")
    ax.set_ylabel(r"$\eta(x)$")
    ax.set_ylim(0, 1.05)
    decorate(ax)

    ax = axs[1, 0]
    ax.plot(x, kernel, color="tab:blue", label=r"original Gaussian kernel $K(x)$")
    ax.fill_between(x, 0, kernel, color="tab:blue", alpha=0.22)
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1.8)
    ax.set_title(r"(d) Original kernel $K(x)$")
    ax.set_ylabel("kernel value")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", framealpha=0.96)
    decorate(ax)

    ax = axs[1, 1]
    ax.plot(x, kernel, color="tab:blue", linestyle="--", label=r"nominal $K(x)$")
    ax.plot(x, modified_kernel, color="tab:orange",
            label=r"modified $\widetilde{K}(x)=\eta(x)K(x)$")
    ax.fill_between(x, modified_kernel, kernel, color="tab:blue", alpha=0.18,
                    label="locally removed part")
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1.8)
    ax.set_title("(e) Geometry-dependent modified kernel")
    ax.set_ylabel("kernel value")
    ax.set_ylim(0, 1.08)
    # Outside the data rectangle: large legend cannot hide either kernel.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
              framealpha=0.98, borderaxespad=0.0, columnspacing=1.1,
              handlelength=2.4, fontsize=16)
    decorate(ax)

    ax = axs[1, 2]
    ax.plot(x, h, color="tab:blue", alpha=0.58, label=r"pre-deposition $h(x)$")
    ax.plot(x, nominal_final, color="tab:orange", linewidth=3.0,
            label=r"nominal result $h(x)+mK(x)$")
    ax.plot(x, modified_final, color="tab:green", linestyle="--", linewidth=3.0,
            label=r"modified result $h(x)+m\widetilde{K}(x)$")
    ax.fill_between(x, modified_final, nominal_final, color="tab:blue", alpha=0.17,
                    label="visible final-geometry gap")
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1.8)
    gap_index = int(np.argmax(gap))
    gap_x = x[gap_index]
    ax.axvline(gap_x, color="tab:blue", linestyle=":", linewidth=1.8)
    ax.annotate("largest gap between final profiles",
                xy=(gap_x, modified_final[gap_index]), xytext=(0.62, 0.108),
                fontsize=17, arrowprops=dict(arrowstyle="->", lw=2.0))
    ax.set_title("(f) Resulting geometry after one deposition step")
    ax.set_ylabel("height")
    ax.set_ylim(-0.13, 0.265)
    # Outside the plot, beneath the panel; no curve or inset is covered.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
              framealpha=0.98, borderaxespad=0.0, fontsize=15.5,
              columnspacing=1.0, handlelength=2.5)
    decorate(ax)

    inset = inset_axes(ax, width="43%", height="36%", loc="lower right", borderpad=1.2)
    mask = (x >= -1.25) & (x <= 1.25)
    inset.plot(x[mask], gap[mask], color="tab:blue", linewidth=2.3)
    inset.fill_between(x[mask], 0, gap[mask], color="tab:blue", alpha=0.22)
    inset.axvline(0, color="tab:blue", linestyle="--", linewidth=1.4)
    inset.axvline(gap_x, color="tab:blue", linestyle=":", linewidth=1.4)
    inset.set_title("final-profile gap", fontsize=17)
    inset.set_xlabel(r"$x$", fontsize=16)
    inset.set_ylabel("gap", fontsize=16)
    inset.tick_params(labelsize=14, length=4)
    inset.grid(True, alpha=0.28)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "original_vs_modified_gaussian_kernels.png", bbox_inches="tight")
    fig.savefig(out / "original_vs_modified_gaussian_kernels.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
