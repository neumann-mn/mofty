# This file is part of MOFTy.
#
# MOFTy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free
# Software Foundation.
#
# MOFTy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# MOFTy. If not, see http://www.gnu.org/licenses/
#
# Copyright(C) 2026 Maximilian Neumann

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

COL_MAP = "seismic"
FIG_SIZE = (6 / 2.54, 4.5 / 2.54)
FONT_SIZE = 5
TICKS_WIDTH = 0.5
TICKS_LENGTH = 1.5
BORDER_WIDTH = 0.5
LABEL_PAD = 3
TITLE_PAD = 3
GRID_FIG_SCALE = 0.5
LINE_WIDTH = 1
PLOT_DPI = 1200

def _apply_axes_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)
    ax.tick_params(width=TICKS_WIDTH, length=TICKS_LENGTH)


def cfm_plotting(
    val,
    kk,
    *,
    coords,
    covariate_keys,
    plot_dot_size=1,
    z_slices=10,
    flip_image=False,
    padding=2,
    output_dir,
    prefix="Factor",
    stat_name=None,
    master,
    iteration=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_hist():
        if stat_name == "std":
            return

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        try:
            _apply_axes_style(ax)

            hist_vals = np.asarray(val)
            hist_vals = hist_vals[np.isfinite(hist_vals)]

            if hist_vals.size > 0:
                ax.hist(
                    hist_vals.flatten(),
                    bins=100,
                    color="#3E84E0",
                    edgecolor="#3E84E0",
                )

            ax.set_title(kk, fontsize=FONT_SIZE, pad=TITLE_PAD)
            ax.set_xlabel("Value", fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.set_ylabel("Frequency", fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.tick_params(
                axis="x",
                labelsize=FONT_SIZE,
                width=TICKS_WIDTH,
                length=TICKS_LENGTH,
            )
            ax.set_yticks([])
            fig.tight_layout()

            if master:
                s_kk = f"{kk}"
                s_name = f"_{stat_name}" if stat_name else ""
                s_iter = f"_iteration_{iteration}" if iteration is not None else ""

                if iteration is not None and iteration > 0:
                    s_iter_min = f"_iteration_{iteration - 1}"
                    out_file_latest_prev = f"latest_{s_kk}{s_name}_hist{s_iter_min}.png"
                    (output_dir / out_file_latest_prev).unlink(missing_ok=True)

                out_file = f"{s_kk}{s_name}_hist{s_iter}.png"
                fig.savefig(
                    output_dir / out_file,
                    bbox_inches="tight",
                    dpi=PLOT_DPI,
                    transparent=True,
                )

                if iteration is not None and "runs_combined" not in output_dir.parent.name:
                    out_file_latest = f"latest_{s_kk}{s_name}_hist{s_iter}.png"
                    fig.savefig(
                        output_dir / out_file_latest,
                        bbox_inches="tight",
                        dpi=PLOT_DPI,
                        transparent=True,
                    )

        finally:
            plt.close(fig)

    n_dims = coords.shape[1]
    if n_dims not in [2, 3]:
        print(
            f"Warning: cfm_plotting only supports 2D or 3D coordinates. "
            f"Got {n_dims}D. Skipping plot for {kk}."
        )
        return

    n_factors_to_plot = val.shape[0]

    # ---------- 3D: slice along z into a grid of 2D subplots ----------
    if n_dims == 3:
        if kk == "Z_cfm_op":
            z_slices = z_slices * padding

        z_vals_rounded = coords[:, 2]
        z_levels = np.unique(z_vals_rounded)
        indices = np.linspace(0, len(z_levels) - 1, z_slices, dtype=int)
        z_levels = z_levels[indices]
        n_slices = len(z_levels)

        for f_idx in range(n_factors_to_plot):
            if kk == "Z_cfm_op":
                factor_values = val[f_idx, ...].flatten()
            else:
                factor_values = val[f_idx, :]

            if stat_name == "std":
                cmap = "cividis"
                mi = np.nanmin(factor_values)
                ma = None
            else:
                amax = np.nanmax(np.abs(factor_values))
                if np.nanmin(factor_values) < 0:
                    cmap = COL_MAP
                    mi, ma = -amax, amax
                else:
                    cmap = "cividis"
                    mi = np.nanmin(factor_values)
                    ma = None

            ncols = min(5, n_slices)
            nrows = int(np.ceil(n_slices / ncols))

            fig = plt.figure(
                figsize=(ncols * 2 * GRID_FIG_SCALE, nrows * 2 * GRID_FIG_SCALE)
            )

            try:
                gs = GridSpec(
                    nrows + 1,
                    ncols + 1,
                    height_ratios=[1] * nrows + [0.1],
                    width_ratios=[1] * ncols + [0.1],
                    wspace=0.2,
                    hspace=0.2,
                )

                sc = None

                for s_idx, z_level in enumerate(z_levels):
                    row = s_idx // ncols
                    col = s_idx % ncols
                    ax = fig.add_subplot(gs[row, col])
                    _apply_axes_style(ax)

                    mask = z_vals_rounded == z_level
                    if not np.any(mask):
                        ax.axis("off")
                        continue

                    x = coords[mask, 0]
                    y = coords[mask, 1]
                    c = factor_values[mask]
                    norm = None

                    if kk != "Z_cfm_op":
                        idx = np.argsort(c)
                        x_sorted = x[idx]
                        y_sorted = y[idx]
                        c_sorted = c[idx]

                        sc = ax.scatter(
                            x_sorted,
                            y_sorted,
                            c=c_sorted,
                            cmap=cmap,
                            norm=norm,
                            s=plot_dot_size,
                            vmin=mi,
                            vmax=ma,
                            marker="s",
                            edgecolors="none",
                            linewidths=0,
                        )

                        if flip_image:
                            ax.invert_yaxis()

                    else:
                        if c.size == 0:
                            ax.axis("off")
                            continue

                        unique_x_slice = np.sort(np.unique(x))
                        unique_y_slice = np.sort(np.unique(y))

                        if len(unique_x_slice) * len(unique_y_slice) != c.size:
                            sc = ax.scatter(
                                x=x,
                                y=y,
                                c=c,
                                cmap=cmap,
                                vmin=mi,
                                vmax=ma,
                                s=plot_dot_size,
                                rasterized=False,
                            )

                            if flip_image:
                                ax.invert_yaxis()

                        else:
                            order = np.lexsort((x, y))
                            grid_c = c[order].reshape(
                                len(unique_y_slice), len(unique_x_slice)
                            )
                            origin = "upper" if flip_image else "lower"

                            sc = ax.imshow(
                                grid_c,
                                cmap=cmap,
                                vmin=mi,
                                vmax=ma,
                                origin=origin,
                                norm=norm,
                                extent=(
                                    unique_x_slice.min(),
                                    unique_x_slice.max(),
                                    unique_y_slice.min(),
                                    unique_y_slice.max(),
                                ),
                                interpolation="none",
                                aspect="auto",
                            )

                    ax.set_aspect("equal", adjustable="box")
                    ax.set_title(
                        f"{covariate_keys[2]} = {z_level:.2f}",
                        fontsize=FONT_SIZE,
                        pad=TITLE_PAD,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if col == 0:
                        ax.set_ylabel(
                            covariate_keys[1],
                            fontsize=FONT_SIZE,
                            labelpad=LABEL_PAD,
                        )

                    if row == nrows - 1:
                        ax.set_xlabel(
                            covariate_keys[0],
                            fontsize=FONT_SIZE,
                            labelpad=LABEL_PAD,
                        )

                if sc is not None:
                    cbar_proportion = 0.3
                    spacer = (1 - cbar_proportion) / 2

                    cbar_gs = gs[:, -1].subgridspec(
                        3,
                        1,
                        height_ratios=[spacer, cbar_proportion, spacer],
                    )
                    cbar_ax = fig.add_subplot(cbar_gs[1])
                    _apply_axes_style(cbar_ax)

                    cb = fig.colorbar(sc, cax=cbar_ax, extend="neither", pad=0.05)
                    cb.ax.tick_params(
                        labelsize=FONT_SIZE,
                        length=TICKS_LENGTH,
                        width=TICKS_WIDTH,
                    )
                    cb.outline.set_linewidth(BORDER_WIDTH)

                if master:
                    s_name = f"_{stat_name}" if stat_name else ""
                    s_kk = f"{kk}"
                    s_iter = f"_iteration_{iteration}" if iteration is not None else ""

                    if iteration is not None and iteration > 0:
                        s_iter_min = f"_iteration_{iteration - 1}"
                        out_file_latest_prev = (
                            f"latest_{s_kk}{s_name}_factor{f_idx+1}{s_iter_min}.png"
                        )
                        (output_dir / out_file_latest_prev).unlink(missing_ok=True)

                    out_file = f"{s_kk}{s_name}_factor{f_idx+1}{s_iter}.png"
                    fig.savefig(
                        output_dir / out_file,
                        bbox_inches="tight",
                        dpi=PLOT_DPI,
                        transparent=True,
                    )

                    if iteration is not None and "runs_combined" not in output_dir.parent.name:
                        out_file_latest = (
                            f"latest_{s_kk}{s_name}_factor{f_idx+1}{s_iter}.png"
                        )
                        fig.savefig(
                            output_dir / out_file_latest,
                            bbox_inches="tight",
                            dpi=PLOT_DPI,
                            transparent=True,
                        )

            finally:
                plt.close(fig)

        plot_hist()
        return

    # ---------- 2D: preserve original multi-factor layout ----------
    ncols = min(5, n_factors_to_plot)
    nrows = int(np.ceil(n_factors_to_plot / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4 * GRID_FIG_SCALE, nrows * 4 * GRID_FIG_SCALE),
        squeeze=False,
    )

    try:
        axes = axes.flatten()
        sc = None

        for i in range(n_factors_to_plot):
            ax = axes[i]
            _apply_axes_style(ax)
            factor_values = val[i, :]

            if stat_name == "std":
                mi = np.nanmin(factor_values)
                ma = None
                cmap = "cividis"
            else:
                amax = np.nanmax(np.abs(factor_values))
                if np.nanmin(factor_values) < 0:
                    cmap = COL_MAP
                    mi, ma = -amax, amax
                else:
                    cmap = "cividis"
                    mi = np.nanmin(factor_values)
                    ma = None

            if kk != "Z_cfm_op":
                norm = None
                x = coords[:, 0]
                y = coords[:, 1]
                c = factor_values

                nan_mask = np.isnan(c)
                valid_mask = ~nan_mask

                if np.any(nan_mask):
                    ax.scatter(
                        x[nan_mask],
                        y[nan_mask],
                        color="lightgray",
                        s=plot_dot_size,
                        marker="s",
                        edgecolors="none",
                        linewidths=0,
                        rasterized=False,
                    )

                if np.any(valid_mask):
                    c_valid = c[valid_mask]
                    order = np.argsort(c_valid)
                    x_sorted = x[valid_mask][order]
                    y_sorted = y[valid_mask][order]
                    c_sorted = c_valid[order]

                    sc = ax.scatter(
                        x_sorted,
                        y_sorted,
                        c=c_sorted,
                        cmap=cmap,
                        vmin=mi,
                        vmax=ma,
                        s=plot_dot_size,
                        marker="s",
                        edgecolors="none",
                        linewidths=0,
                        rasterized=False,
                    )

                if flip_image:
                    ax.invert_yaxis()

            else:
                x = coords[:, 0]
                y = coords[:, 1]
                xu = np.sort(np.unique(x))
                yu = np.sort(np.unique(y))
                fv = np.asarray(factor_values)
                grid_c = fv.T
                origin = "upper" if flip_image else "lower"
                norm = None

                sc = ax.imshow(
                    grid_c,
                    cmap=cmap,
                    vmin=mi,
                    vmax=ma,
                    norm=norm,
                    extent=(xu.min(), xu.max(), yu.min(), yu.max()),
                    origin=origin,
                    interpolation="none",
                    aspect="auto",
                )

            ax.set_xlabel(covariate_keys[0], fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.set_ylabel(covariate_keys[1], fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

            if prefix:
                ax.set_title(f"{prefix} {i+1}", fontsize=FONT_SIZE, pad=TITLE_PAD)
            else:
                ax.set_title("", fontsize=FONT_SIZE, pad=TITLE_PAD)

            if sc is not None:
                cb = fig.colorbar(sc, ax=ax, shrink=0.6, extend="neither", pad=0.05)
                cb.outline.set_linewidth(BORDER_WIDTH)
                cb.ax.tick_params(
                    labelsize=FONT_SIZE,
                    length=TICKS_LENGTH,
                    width=TICKS_WIDTH,
                )
                _apply_axes_style(cb.ax)

        for i in range(n_factors_to_plot, len(axes)):
            axes[i].axis("off")

        fig.suptitle("")
        fig.subplots_adjust(hspace=0.2)

        if master:
            s_name = f"_{stat_name}" if stat_name else ""
            s_kk = f"{kk}"
            s_iter = f"_iteration_{iteration}" if iteration is not None else ""

            if iteration is not None and iteration > 0:
                s_iter_min = f"_iteration_{iteration - 1}"
                out_file_latest_prev = f"latest_{s_kk}{s_name}{s_iter_min}.png"
                (output_dir / out_file_latest_prev).unlink(missing_ok=True)

            out_file = f"{s_kk}{s_name}{s_iter}.png"
            fig.savefig(
                output_dir / out_file,
                bbox_inches="tight",
                dpi=PLOT_DPI,
                transparent=True,
            )

            if iteration is not None and "runs_combined" not in output_dir.parent.name:
                out_file_latest = f"latest_{s_kk}{s_name}{s_iter}.png"
                fig.savefig(
                    output_dir / out_file_latest,
                    bbox_inches="tight",
                    dpi=PLOT_DPI,
                    transparent=True,
                )

    finally:
        plt.close(fig)

    plot_hist()


def standard_plotting(
    val,
    kk,
    *,
    xlabel="",
    ylabel="",
    stat_name=None,
    iteration=None,
    percentile=99,
    output_dir,
    master,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(val)

    # -------------------------
    # Main heatmap-like plot
    # -------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    try:
        _apply_axes_style(ax)

        # Make colormap selection robust to NaNs and sign of data
        finite_mask = np.isfinite(arr)
        has_finite = finite_mask.any()

        min_val = np.nanmin(arr) if has_finite else 0.0
        max_val = np.nanmax(arr) if has_finite else 0.0

        if "Z" not in kk and "W" not in kk and has_finite:
            lim = np.percentile(np.abs(arr[finite_mask]), percentile)
        else:
            lim = max(abs(min_val), abs(max_val))

        # Force non-diverging scaling for std-like plots and Z_cfm_op_scaling
        if kk.startswith("std") or "Z_cfm_op_scaling" in kk or stat_name == "std":
            ma = None
            mi = 0
            cmap = "cividis"
        else:
            # Diverging if data span both signs; otherwise start at zero / min
            if (min_val < 0) and (max_val > 0) and (lim is not None):
                mi, ma = -lim, lim
                cmap = COL_MAP
            else:
                ma = lim
                mi = np.nanmin(arr) if has_finite else 0.0
                cmap = "cividis"

        norm = None

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("lightgray")

        plot_arr = (
            val
            if getattr(val, "ndim", np.ndim(val)) == 2
            else np.asarray(val)[None, :]
        )
        plot_arr = np.ma.masked_invalid(plot_arr)

        im = ax.imshow(
            plot_arr,
            vmin=mi,
            vmax=ma,
            cmap=cmap_obj,
            norm=norm,
            origin="lower",
            aspect="auto",
            interpolation="none",
        )

        ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=LABEL_PAD)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=LABEL_PAD)
        ax.set_xticks([])
        ax.set_yticks([])

        if has_finite:
            unique_vals = np.unique(arr[finite_mask])
            is_binary_like = np.all(np.isin(unique_vals, [0, 1]))
        else:
            is_binary_like = False

        if (
            "Z" not in kk
            and "W" not in kk
            and ma is not None
            and mi >= 0
            and percentile < 100
            and not is_binary_like
        ):
            extend = "max"
        elif (
            "Z" not in kk
            and "W" not in kk
            and mi is not None
            and mi < 0
            and ma is not None
            and ma > 0
            and percentile < 100
        ):
            extend = "both"
        else:
            extend = "neither"

        cb = fig.colorbar(im, ax=ax, extend=extend, pad=0.05)
        cb.outline.set_linewidth(BORDER_WIDTH)
        cb.ax.tick_params(
            labelsize=FONT_SIZE,
            length=TICKS_LENGTH,
            width=TICKS_WIDTH,
        )
        _apply_axes_style(cb.ax)

        ax.set_title(kk, fontsize=FONT_SIZE, pad=TITLE_PAD)
        fig.tight_layout()

        if master:
            s_kk = f"{kk}"
            s_name = f"_{stat_name}" if stat_name else ""
            s_iter = f"_iteration_{iteration}" if iteration is not None else ""

            if iteration is not None and iteration > 0:
                s_iter_min = f"_iteration_{iteration - 1}"
                out_file_latest_prev = f"latest_{s_kk}{s_name}{s_iter_min}.png"
                (output_dir / out_file_latest_prev).unlink(missing_ok=True)

            out_file = f"{s_kk}{s_name}{s_iter}.png"

            fig.savefig(
                output_dir / out_file,
                bbox_inches="tight",
                dpi=PLOT_DPI,
                transparent=True,
            )

            if iteration is not None and "runs_combined" not in output_dir.parent.name:
                out_file_latest = f"latest_{s_kk}{s_name}{s_iter}.png"
                fig.savefig(
                    output_dir / out_file_latest,
                    bbox_inches="tight",
                    dpi=PLOT_DPI,
                    transparent=True,
                )

    finally:
        plt.close(fig)

    # -------------------------
    # Histogram plot
    # -------------------------
    if stat_name != "std" and kk != "Z_cfm_op_pspec" and kk != "Z_cfm_op_scaling":
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        try:
            _apply_axes_style(ax)

            hist_vals = arr[np.isfinite(arr)]

            if hist_vals.size > 0:
                ax.hist(
                    hist_vals.flatten(),
                    bins=100,
                    color="#3E84E0",
                    edgecolor="#3E84E0",
                )

            ax.set_title(kk, fontsize=FONT_SIZE, pad=TITLE_PAD)
            ax.set_xlabel("Value", fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.set_ylabel("Frequency", fontsize=FONT_SIZE, labelpad=LABEL_PAD)

            ax.tick_params(
                axis="x",
                labelsize=FONT_SIZE,
                width=TICKS_WIDTH,
                length=TICKS_LENGTH,
            )

            # Hide y-ticks for cleaner look
            ax.set_yticks([])

            fig.tight_layout()

            if master:
                s_kk = f"{kk}"
                s_name = f"_{stat_name}" if stat_name else ""
                s_iter = f"_iteration_{iteration}" if iteration is not None else ""

                if iteration is not None and iteration > 0:
                    s_iter_min = f"_iteration_{iteration - 1}"
                    out_file_latest_prev = f"latest_{s_kk}{s_name}_hist{s_iter_min}.png"
                    (output_dir / out_file_latest_prev).unlink(missing_ok=True)

                out_file = f"{s_kk}{s_name}_hist{s_iter}.png"

                fig.savefig(
                    output_dir / out_file,
                    bbox_inches="tight",
                    dpi=PLOT_DPI,
                    transparent=True,
                )

                if iteration is not None and "runs_combined" not in output_dir.parent.name:
                    out_file_latest = f"latest_{s_kk}{s_name}_hist{s_iter}.png"
                    fig.savefig(
                        output_dir / out_file_latest,
                        bbox_inches="tight",
                        dpi=PLOT_DPI,
                        transparent=True,
                    )

        finally:
            plt.close(fig)


def pspec_plotting(
    op_kk,
    samples,
    kk,
    *,
    n_factors=1,
    iteration=None,
    output_dir,
    master,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppspec_samples = samples.iterator()

    pspec_vals = []
    for sample in ppspec_samples:
        if op_kk is not None:
            sample = op_kk.force(sample)
        sample_val = sample.asnumpy()
        pspec_vals.append(sample_val)

    all_samples = np.array(pspec_vals)  # (n_samples, n_factors, k)
    plot_mean = all_samples.mean(axis=0)  # (n_factors, k)

    ncols = min(4, n_factors)
    nrows = int(np.ceil(n_factors / ncols))
    subplot_width = 4 * GRID_FIG_SCALE
    subplot_height = 3 * GRID_FIG_SCALE
    hspace = 0.4
    fig_height = subplot_height * (nrows + (nrows - 1) * hspace)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * subplot_width, fig_height),
        squeeze=False,
    )

    try:
        axes = axes.flatten()

        if plot_mean.ndim == 1:
            # Single factor case: reshape to (1, k)
            plot_mean = plot_mean[np.newaxis, :]
            all_samples = (
                all_samples[:, np.newaxis, :]
                if all_samples.ndim == 2
                else all_samples
            )

        for i in range(n_factors):
            ax = axes[i]
            _apply_axes_style(ax)

            if plot_mean.shape[1] == 0:
                continue

            x = np.arange(1, plot_mean.shape[1] + 1)

            # Posterior samples
            for s_idx in range(all_samples.shape[0]):
                sample_factor = all_samples[s_idx, i, :].squeeze()
                ax.plot(
                    x,
                    sample_factor,
                    color="lightsteelblue",
                    alpha=0.7,
                    linewidth=LINE_WIDTH,
                    label="Post. samples" if s_idx == 0 else None,
                )

            # Posterior mean
            mean_factor = plot_mean[i, :].squeeze()
            ax.plot(
                x,
                mean_factor,
                color="blue",
                linewidth=LINE_WIDTH,
                label="Post. mean",
            )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.minorticks_off()

            if n_factors > 1:
                ax.text(
                    0.02,
                    0.02,
                    f"Factor {i + 1}",
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE,
                    ha="left",
                    va="bottom",
                )

            ax.grid(False)

            ax.set_xlabel("|k|", fontsize=FONT_SIZE, labelpad=LABEL_PAD - 1)
            ax.set_ylabel("p(|k|)", fontsize=FONT_SIZE, labelpad=LABEL_PAD)
            ax.tick_params(
                labelsize=FONT_SIZE,
                length=TICKS_LENGTH,
                width=TICKS_WIDTH,
            )

            leg = ax.legend(
                loc="upper right",
                fontsize=FONT_SIZE - 1,
                frameon=False,
                handlelength=1,
            )

            for line in leg.get_lines():
                line.set_linewidth(LINE_WIDTH)

        for i in range(n_factors, len(axes)):
            axes[i].axis("off")

        fig.suptitle("")
        fig.subplots_adjust(hspace=hspace, wspace=0.4)

        if master:
            s_kk = f"{kk}"
            s_iter = f"_iteration_{iteration}" if iteration is not None else ""

            if iteration is not None and iteration > 0:
                s_iter_min = f"_iteration_{iteration - 1}"
                out_file_latest_prev = f"latest_{s_kk}{s_iter_min}.png"
                (output_dir / out_file_latest_prev).unlink(missing_ok=True)

            out_file = f"{s_kk}{s_iter}.png"

            fig.savefig(
                output_dir / out_file,
                bbox_inches="tight",
                dpi=PLOT_DPI,
                transparent=True,
            )

            if iteration is not None and "runs_combined" not in output_dir.parent.name:
                out_file_latest = f"latest_{s_kk}{s_iter}.png"
                fig.savefig(
                    output_dir / out_file_latest,
                    bbox_inches="tight",
                    dpi=PLOT_DPI,
                    transparent=True,
                )

    finally:
        plt.close(fig)
