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

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mofax
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import squidpy as sq
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


def cm(value):
    return value / 2.54


@dataclass(frozen=True)
class MoftyModelSpec:
    factor_count: int
    variant: str
    label: str
    run_dir: Path
    filename: str


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_factor", type=int, default=10, help="Maximum displayed factor index.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("../input/input_gbm"),
        help="Directory containing processed_mdata.h5mu.",
    )
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=Path("../paper/supp_fig_comparison_mofty_factors"),
        help="Output directory for plots and CSV summaries.",
    )
    return parser


def get_model_specs():
    return [
        MoftyModelSpec(
            factor_count=4,
            variant="combined",
            label="4 comb.",
            run_dir=Path("../output/output_gbm_cfm_4/runs_combined_factors_4"),
            filename="mofty_model_Z.hdf5",
        ),
        MoftyModelSpec(
            factor_count=4,
            variant="cfm",
            label="4 CFM",
            run_dir=Path("../output/output_gbm_cfm_4/runs_combined_factors_4"),
            filename="mofty_model_Z_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=4,
            variant="non-cfm",
            label="4 non-CFM",
            run_dir=Path("../output/output_gbm_cfm_4/runs_combined_factors_4"),
            filename="mofty_model_Z_non_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=6,
            variant="combined",
            label="6 comb.",
            run_dir=Path("../output/output_gbm_cfm_6/runs_combined_factors_6"),
            filename="mofty_model_Z.hdf5",
        ),
        MoftyModelSpec(
            factor_count=6,
            variant="cfm",
            label="6 CFM",
            run_dir=Path("../output/output_gbm_cfm_6/runs_combined_factors_6"),
            filename="mofty_model_Z_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=6,
            variant="non-cfm",
            label="6 non-CFM",
            run_dir=Path("../output/output_gbm_cfm_6/runs_combined_factors_6"),
            filename="mofty_model_Z_non_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=8,
            variant="combined",
            label="8 comb.",
            run_dir=Path("../output/output_gbm_cfm_8/runs_combined_factors_8"),
            filename="mofty_model_Z.hdf5",
        ),
        MoftyModelSpec(
            factor_count=8,
            variant="cfm",
            label="8 CFM",
            run_dir=Path("../output/output_gbm_cfm_8/runs_combined_factors_8"),
            filename="mofty_model_Z_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=8,
            variant="non-cfm",
            label="8 non-CFM",
            run_dir=Path("../output/output_gbm_cfm_8/runs_combined_factors_8"),
            filename="mofty_model_Z_non_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=10,
            variant="combined",
            label="10 comb.",
            run_dir=Path("../output/output_gbm_cfm_10/runs_combined_factors_10"),
            filename="mofty_model_Z.hdf5",
        ),
        MoftyModelSpec(
            factor_count=10,
            variant="cfm",
            label="10 CFM",
            run_dir=Path("../output/output_gbm_cfm_10/runs_combined_factors_10"),
            filename="mofty_model_Z_cfm.hdf5",
        ),
        MoftyModelSpec(
            factor_count=10,
            variant="non-cfm",
            label="10 non-CFM",
            run_dir=Path("../output/output_gbm_cfm_10/runs_combined_factors_10"),
            filename="mofty_model_Z_non_cfm.hdf5",
        ),
    ]


def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        }
    )


def prepare_rowstd_spatial_neighbors(adata_obj):
    sq.gr.spatial_neighbors(adata_obj)

    spatial_conn = adata_obj.obsp["spatial_connectivities"]
    row_sums = np.asarray(spatial_conn.sum(axis=1)).ravel()

    row_inv = np.zeros_like(row_sums, dtype=float)
    np.divide(1.0, row_sums, out=row_inv, where=row_sums > 0)

    row_std_conn = sp.diags(row_inv) @ spatial_conn
    adata_obj.obsp["spatial_connectivities_std"] = row_std_conn

    adata_obj.uns["neighbors"] = {
        "connectivities_key": "spatial_connectivities_std",
        "distances_key": "spatial_distances",
        "params": {"n_neighbors": 6, "coord_type": "generic", "method": "umap"},
        "connectivities": row_std_conn,
        "distances": adata_obj.obsp["spatial_distances"],
    }
    return adata_obj


def sort_factor_columns(columns):
    def factor_key(col_name):
        match = re.search(r"(\d+)$", str(col_name))
        if match:
            return int(match.group(1))
        return 10**9

    return sorted(columns, key=factor_key)


def get_first_n_factor_df(model, n_factors):
    factors_df = model.get_factors(df=True).copy()
    factor_cols = [
        col for col in sort_factor_columns(factors_df.columns) if re.match(r"^Factor\d+$", str(col))
    ]

    if len(factor_cols) < n_factors:
        raise ValueError(f"Model has only {len(factor_cols)} factors, but {n_factors} are required.")

    out = factors_df[factor_cols[:n_factors]].copy()
    out.columns = [f"F{i + 1}" for i in range(n_factors)]
    return out


def align_scores_to_obs(obs_names, score_df, model_label):
    aligned = score_df.reindex(obs_names)
    if aligned.isna().any().any():
        missing_rows = int(aligned.isna().any(axis=1).sum())
        print(f"Warning: {model_label} has {missing_rows} samples with missing factors after alignment; filling with 0.")
        aligned = aligned.fillna(0.0)
    return aligned


def compute_abs_corr_matrix(reference_df, target_df):
    n_ref = reference_df.shape[1]
    n_target = target_df.shape[1]
    corr_abs = np.zeros((n_ref, n_target), dtype=float)

    for i in range(n_ref):
        ref_values = pd.to_numeric(reference_df.iloc[:, i], errors="coerce")
        for j in range(n_target):
            target_values = pd.to_numeric(target_df.iloc[:, j], errors="coerce")
            corr_val = ref_values.corr(target_values)
            corr_abs[i, j] = abs(float(corr_val)) if np.isfinite(corr_val) else 0.0

    return corr_abs


def safe_pearson_corr_and_p(x_vals, y_vals):
    x_num = pd.to_numeric(pd.Series(x_vals), errors="coerce").to_numpy(dtype=float)
    y_num = pd.to_numeric(pd.Series(y_vals), errors="coerce").to_numpy(dtype=float)

    finite_mask = np.isfinite(x_num) & np.isfinite(y_num)
    if int(np.sum(finite_mask)) < 3:
        return np.nan, np.nan

    try:
        corr_val, p_val = pearsonr(x_num[finite_mask], y_num[finite_mask])
    except Exception:
        return np.nan, np.nan

    if not np.isfinite(corr_val) or not np.isfinite(p_val):
        return np.nan, np.nan

    return float(corr_val), float(p_val)


def format_float_for_txt(value):
    if not np.isfinite(value):
        return "N/A"
    return f"{float(value):.1e}"


def write_p_values_summary_txt(out_file, spatial_row_stats_by_variant, weight_p_by_view):
    lines = []
    lines.append("MOFTy significance summary")
    lines.append("")
    lines.append("Spatial grid row-level significance")
    lines.append("Each value is the maximum pairwise p-value across model pairs for that row.")

    for variant in ["combined", "cfm", "non-cfm"]:
        lines.append("")
        lines.append(f"Variant: {variant}")
        lines.append("row\tmax_pairwise_p")
        row_stats = spatial_row_stats_by_variant.get(variant, {})
        for row_pos in sorted(row_stats):
            _, max_p_val = row_stats[row_pos]
            lines.append(f"F{row_pos}\t{format_float_for_txt(max_p_val)}")

    lines.append("")
    lines.append("Best-match weight correlation significance")
    lines.append("Matrices contain pairwise p-values for best-match factor pairs.")

    for view_name in ["gene expression", "protein expression"]:
        lines.append("")
        lines.append(f"View: {view_name}")
        p_matrix = weight_p_by_view.get(view_name)
        if p_matrix is None or p_matrix.empty:
            lines.append("N/A")
            continue

        matrix_to_write = p_matrix.applymap(format_float_for_txt)
        lines.append(matrix_to_write.to_csv(sep="\t", index=True))

    out_file.write_text("\n".join(lines), encoding="utf-8")


def best_abs_corr_order(reference_df, target_df):
    corr_abs = compute_abs_corr_matrix(reference_df, target_df)
    row_idx, col_idx = linear_sum_assignment(-corr_abs)

    order = [None] * corr_abs.shape[0]
    for r, c in zip(row_idx, col_idx):
        order[r] = int(c) + 1

    assigned = {v for v in order if v is not None}
    remaining = [i for i in range(1, target_df.shape[1] + 1) if i not in assigned]

    for i, value in enumerate(order):
        if value is None:
            order[i] = remaining.pop(0) if remaining else (i + 1)

    return order


def match_to_reference_rows(reference_df, target_df):
    corr_abs = compute_abs_corr_matrix(reference_df, target_df)
    row_idx, col_idx = linear_sum_assignment(-corr_abs)

    # Returns mapping from reference row index (1-based) to target factor index (1-based).
    return {int(r) + 1: int(c) + 1 for r, c in zip(row_idx, col_idx)}


def build_global_row_factor_maps(aligned_scores, model_specs):
    factor_counts = sorted({spec.factor_count for spec in model_specs})
    max_factor = max(factor_counts)

    combined_specs = [spec for spec in model_specs if spec.variant == "combined"]
    combined_ref_spec = max(combined_specs, key=lambda s: s.factor_count)
    reference_df = aligned_scores[combined_ref_spec.label].iloc[:, :max_factor]

    row_factor_ids = list(range(1, max_factor + 1))
    row_factor_maps = {combined_ref_spec.label: list(range(1, max_factor + 1))}

    # 1) Map combined models (4/6/8) into global rows defined by 10-combined.
    for spec in combined_specs:
        if spec.label == combined_ref_spec.label:
            continue

        target_df = aligned_scores[spec.label].iloc[:, :spec.factor_count]
        row_to_factor = match_to_reference_rows(reference_df, target_df)

        factor_map = [None] * max_factor
        for row_idx, target_factor in row_to_factor.items():
            factor_map[row_idx - 1] = target_factor
        row_factor_maps[spec.label] = factor_map

    # 2) For each factor-count group, map CFM/non-CFM to that group's combined,
    # then project onto the same global rows.
    for factor_count in factor_counts:
        group_specs = [spec for spec in model_specs if spec.factor_count == factor_count]
        combined_spec = next(spec for spec in group_specs if spec.variant == "combined")
        combined_df = aligned_scores[combined_spec.label].iloc[:, :factor_count]
        combined_row_map = row_factor_maps[combined_spec.label]

        for variant in ["cfm", "non-cfm"]:
            target_spec = next(spec for spec in group_specs if spec.variant == variant)
            target_df = aligned_scores[target_spec.label].iloc[:, :factor_count]
            combined_to_target = best_abs_corr_order(combined_df, target_df)

            target_row_map = [None] * max_factor
            for row_pos, combined_factor_idx in enumerate(combined_row_map, start=1):
                if combined_factor_idx is None:
                    continue
                target_row_map[row_pos - 1] = combined_to_target[combined_factor_idx - 1]

            row_factor_maps[target_spec.label] = target_row_map

    return row_factor_maps, row_factor_ids, combined_ref_spec.label


def extract_variance_by_view(model, model_label, n_factors):
    ve = model.get_variance_explained()
    if not isinstance(ve, pd.DataFrame):
        ve = pd.DataFrame(ve)
    ve = ve.reset_index()

    col_lut = {str(col).lower(): col for col in ve.columns}

    def pick_col(candidates):
        for candidate in candidates:
            for lower_col, orig_col in col_lut.items():
                if lower_col == candidate or candidate in lower_col:
                    return orig_col
        return None

    view_col = pick_col(["view"])
    factor_col = pick_col(["factor"])
    value_col = pick_col(["r2", "variance_ratio", "variance_explained", "variance explained", "value"])

    if view_col is None or factor_col is None or value_col is None:
        raise ValueError(
            f"Could not parse variance explained columns for {model_label}. Found columns: {list(ve.columns)}"
        )

    out = ve[[view_col, factor_col, value_col]].copy()
    out.columns = ["view", "factor", "variance_ratio"]
    out["view"] = out["view"].astype(str)
    out["factor_idx"] = out["factor"].astype(str).str.extract(r"(\d+)").astype(float)
    out["variance_ratio"] = pd.to_numeric(out["variance_ratio"], errors="coerce")

    out = out.dropna(subset=["factor_idx", "variance_ratio"])
    out["factor_idx"] = out["factor_idx"].astype(int)
    out = out[(out["factor_idx"] >= 1) & (out["factor_idx"] <= n_factors)]

    out = (
        out.groupby(["view", "factor_idx"], as_index=False)["variance_ratio"]
        .mean()
        .sort_values(["view", "factor_idx"])
    )

    if (out["variance_ratio"] > 1.0).any():
        out["variance_ratio"] = out["variance_ratio"] / 100.0

    out["model"] = model_label
    return out[["model", "view", "factor_idx", "variance_ratio"]]


def extract_total_variance_by_view(model):
    ve_total = model.calculate_variance_explained()
    if not isinstance(ve_total, pd.DataFrame):
        ve_total = pd.DataFrame(ve_total)

    col_lut = {str(col).strip().lower(): col for col in ve_total.columns}
    if "view" not in col_lut or "r2" not in col_lut:
        raise ValueError(
            "calculate_variance_explained() must provide 'View' and 'R2' columns. "
            f"Found columns: {list(ve_total.columns)}"
        )

    view_col = col_lut["view"]
    r2_col = col_lut["r2"]

    tmp = ve_total[[view_col, r2_col]].copy()
    tmp.columns = ["view", "r2"]
    tmp["view"] = tmp["view"].astype(str).str.strip().str.lower().str.replace("-", "_", regex=False)
    tmp["view"] = tmp["view"].str.replace(" ", "_", regex=False)
    tmp["r2"] = pd.to_numeric(tmp["r2"], errors="coerce")

    view_map = {
        "gene_exp": "gene_exp",
        "gene_expression": "gene_exp",
        "protein": "protein",
        "protein_expression": "protein",
    }
    tmp["view_norm"] = tmp["view"].map(view_map)
    tmp = tmp.dropna(subset=["view_norm", "r2"])

    totals = {"gene_exp": np.nan, "protein": np.nan}
    grouped = tmp.groupby("view_norm", as_index=True)["r2"].mean()
    for view_name in totals:
        if view_name in grouped.index:
            totals[view_name] = float(grouped.loc[view_name])

    return totals


def clean_spatial_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def add_top_padding(ax, frac=0.12):
    y0, y1 = ax.get_ylim()
    span = abs(y1 - y0)
    if span == 0:
        return

    if y0 < y1:
        ax.set_ylim(y0, y1 + frac * span)
    else:
        ax.set_ylim(y0, y1 - frac * span)


def compute_group_factor_vmax(aligned_scores, row_factor_maps, model_specs, max_factor):
    vmax_lookup = {}
    factor_counts = sorted({spec.factor_count for spec in model_specs})

    for factor_count in factor_counts:
        group_specs = [spec for spec in model_specs if spec.factor_count == factor_count]
        for row_idx in range(1, max_factor + 1):

            values_concat = []
            for spec in group_specs:
                source_factor_idx = row_factor_maps[spec.label][row_idx - 1]
                if source_factor_idx is None:
                    continue
                factor_col = f"F{source_factor_idx}"
                values_concat.append(aligned_scores[spec.label][factor_col].to_numpy(dtype=float))

            if not values_concat:
                continue

            combined = np.concatenate(values_concat)
            vmax = np.nanmax(np.abs(combined))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1e-6

            vmax_lookup[(factor_count, row_idx)] = vmax

    return vmax_lookup


def compute_row_min_abs_corr_for_variant(aligned_scores, row_factor_maps, variant_specs, max_factor):
    row_stats = {}
    model_labels = [spec.label for spec in variant_specs]

    for row_pos in range(1, max_factor + 1):
        row_signals = {}
        for model_label in model_labels:
            source_factor_idx = row_factor_maps[model_label][row_pos - 1]
            if source_factor_idx is None:
                continue
            factor_col = f"F{source_factor_idx}"
            row_signals[model_label] = pd.to_numeric(aligned_scores[model_label][factor_col], errors="coerce")

        if len(row_signals) < 2:
            row_stats[row_pos] = (np.nan, np.nan)
            continue

        labels = list(row_signals.keys())
        abs_corr_values = []
        p_values = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                corr_val, p_val = safe_pearson_corr_and_p(
                    row_signals[labels[i]].to_numpy(dtype=float),
                    row_signals[labels[j]].to_numpy(dtype=float),
                )
                if np.isfinite(corr_val):
                    abs_corr_values.append(abs(corr_val))
                if np.isfinite(p_val):
                    p_values.append(p_val)

        min_abs_corr = float(np.min(abs_corr_values)) if abs_corr_values else np.nan
        max_p_val = float(np.max(p_values)) if p_values else np.nan
        row_stats[row_pos] = (min_abs_corr, max_p_val)

    return row_stats


def draw_group_separators(fig, axes, separator_after_cols):
    y_top = axes[0, 0].get_position().y1
    y_bottom = axes[-1, 0].get_position().y0

    for boundary_idx in separator_after_cols:
        left_ax = axes[0, boundary_idx - 1]
        right_ax = axes[0, boundary_idx]
        x_mid = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2.0
        fig.add_artist(
            plt.Line2D(
                [x_mid, x_mid],
                [y_bottom, y_top],
                transform=fig.transFigure,
                color="black",
                linewidth=0.5,
            )
        )


def plot_spatial_grid(
    adata_plot,
    adata_plot_mi,
    aligned_scores,
    row_factor_maps,
    row_factor_ids,
    model_specs,
    variant,
    max_factor,
    out_file,
):
    variant_specs = sorted(
        [spec for spec in model_specs if spec.variant == variant],
        key=lambda s: s.factor_count,
        reverse=True,
    )
    if not variant_specs:
        raise ValueError(f"No model specs found for variant: {variant}")

    n_rows = max_factor
    n_cols = len(variant_specs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cm(5.9), cm(1.9 * n_rows)))
    fig.subplots_adjust(wspace=0.0, hspace=0)

    row_min_abs_corr = compute_row_min_abs_corr_for_variant(
        aligned_scores=aligned_scores,
        row_factor_maps=row_factor_maps,
        variant_specs=variant_specs,
        max_factor=max_factor,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    vmax_lookup = compute_group_factor_vmax(aligned_scores, row_factor_maps, model_specs, max_factor)
    tmp_col = "__tmp_spatial_factor__"

    for row_idx in range(n_rows):
        row_pos = row_idx + 1
        # row_min_corr, _ = row_min_abs_corr.get(row_pos, (np.nan, np.nan))
        # if np.isfinite(row_min_corr):
        #     row_min_corr_text = f"Minimum |r|: {row_min_corr:.2f}"
        # else:
        #     row_min_corr_text = "Minimum |r|: N/A"

        for col_idx, spec in enumerate(variant_specs):
            ax = axes[row_idx, col_idx]

            if row_idx == 0:
                ax.text(
                    0.5,
                    1.02,
                    spec.label,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="center",
                    fontsize=5,
                )

            source_factor_idx = row_factor_maps[spec.label][row_pos - 1]
            if source_factor_idx is None:
                y_factor_label = "N/A"
                zero_values = np.zeros(adata_plot.n_obs, dtype=float)
                # For missing factors, use a row-consistent scale so zero maps are still informative.
                row_vmax_candidates = [
                    vmax for (fc, row_key), vmax in vmax_lookup.items() if row_key == row_pos
                ]
                vmax = max(row_vmax_candidates) if row_vmax_candidates else 1.0
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1e-6

                adata_plot.obs[tmp_col] = zero_values
                sc.pl.spatial(
                    adata_plot,
                    img_key=None,
                    color=tmp_col,
                    cmap="seismic",
                    legend_loc=None,
                    colorbar_loc=None,
                    show=False,
                    vmin=-vmax,
                    vmax=vmax,
                    title=None,
                    size=1.5,
                    ax=ax,
                )
                ax.set_title("")
                add_top_padding(ax, frac=0.10)

                ax.text(
                    0.02,
                    0.98,
                    f"{y_factor_label}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5,
                    bbox={"facecolor": "none", "edgecolor": "none", "pad": 1.2},
                )

                clean_spatial_axis(ax)
                # if col_idx == (n_cols - 1):
                # ax.yaxis.set_label_position("right")
                # ax.set_ylabel(row_min_corr_text, fontsize=5, rotation=90, labelpad=12)
                # ax.yaxis.set_label_coords(1.08, 0.5)
                # else:
                ax.set_ylabel("")
                continue

            y_factor_label = f"F{source_factor_idx}"
            factor_col = f"F{source_factor_idx}"
            values = aligned_scores[spec.label][factor_col].to_numpy(dtype=float)

            vmax = vmax_lookup[(spec.factor_count, row_pos)]
            vmin = -vmax
            # mi = sc.metrics.morans_i(adata_plot_mi, vals=values)

            adata_plot.obs[tmp_col] = values
            sc.pl.spatial(
                adata_plot,
                img_key=None,
                color=tmp_col,
                cmap="seismic",
                legend_loc=None,
                colorbar_loc=None,
                show=False,
                vmin=vmin,
                vmax=vmax,
                title=None,
                size=1.5,
                ax=ax,
            )
            ax.set_title("")
            add_top_padding(ax, frac=0.10)

            ax.text(
                0.02,
                0.98,
                f"{y_factor_label}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                bbox={"facecolor": "none", "edgecolor": "none", "pad": 1.2},
            )

            clean_spatial_axis(ax)

            # if col_idx == (n_cols - 1):
            #     ax.yaxis.set_label_position("right")
            #     ax.set_ylabel(row_min_corr_text, fontsize=5, rotation=90, labelpad=12)
            #     ax.yaxis.set_label_coords(1.08, 0.5)
            # else:
            ax.set_ylabel("")

    if tmp_col in adata_plot.obs.columns:
        adata_plot.obs = adata_plot.obs.drop(columns=[tmp_col])

    draw_group_separators(fig, axes, separator_after_cols=[])
    fig.savefig(out_file, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return row_min_abs_corr


def build_correlation_dataframe(aligned_scores, row_factor_maps, row_factor_ids, model_specs):
    ref_index = next(iter(aligned_scores.values())).index
    corr_df = pd.DataFrame(index=ref_index)
    block_sizes = []

    variant_rank = {"combined": 0, "cfm": 1, "non-cfm": 2}
    ordered_specs = sorted(
        model_specs,
        key=lambda s: (variant_rank.get(s.variant, 99), -s.factor_count),
    )

    for spec in ordered_specs:
        row_map = row_factor_maps[spec.label]
        present_rows = [row_idx + 1 for row_idx, source in enumerate(row_map) if source is not None]
        block_sizes.append(len(present_rows))

        for row_pos in present_rows:
            source_factor_idx = row_map[row_pos - 1]
            factor_col = f"F{source_factor_idx}"
            corr_df[f"{spec.label} F{row_factor_ids[row_pos - 1]}"] = aligned_scores[spec.label][factor_col]

    return corr_df, block_sizes


def get_first_n_weight_df(model, model_label, view_name, n_factors):
    candidate_views = {
        "gene_exp": ["gene_exp", "gene expression", "gene_expression"],
        "protein": ["protein", "protein expression", "protein_expression"],
    }.get(view_name, [view_name])

    last_error = None
    weights_df = None
    for candidate in candidate_views:
        try:
            weights_df = model.get_weights(df=True, views=candidate)
            break
        except Exception as exc:
            last_error = exc

    if weights_df is None:
        raise ValueError(
            f"Could not load weights for view '{view_name}' in {model_label}. "
            f"Tried views: {candidate_views}. Last error: {last_error}"
        )

    if not isinstance(weights_df, pd.DataFrame):
        weights_df = pd.DataFrame(weights_df)

    factor_cols = [
        col for col in sort_factor_columns(weights_df.columns) if re.match(r"^Factor\d+$", str(col))
    ]

    if len(factor_cols) < n_factors:
        weights_df_t = weights_df.T.copy()
        factor_cols_t = [
            col for col in sort_factor_columns(weights_df_t.columns) if re.match(r"^Factor\d+$", str(col))
        ]
        if len(factor_cols_t) >= n_factors:
            weights_df = weights_df_t
            factor_cols = factor_cols_t

    if len(factor_cols) < n_factors:
        raise ValueError(
            f"Weights for {model_label} ({view_name}) expose only {len(factor_cols)} factor columns, "
            f"but {n_factors} are required."
        )

    out = weights_df[factor_cols[:n_factors]].copy()
    out.columns = [f"F{i + 1}" for i in range(n_factors)]

    return out.apply(pd.to_numeric, errors="coerce")


def build_weight_signed_correlation_dataframe(models, row_factor_maps, model_specs, max_factor, view_name):
    variant_rank = {"combined": 0, "cfm": 1, "non-cfm": 2}
    ordered_specs = sorted(
        model_specs,
        key=lambda s: (variant_rank.get(s.variant, 99), -s.factor_count),
    )

    series_map = {}
    block_sizes = []

    for spec in ordered_specs:
        weights_df = get_first_n_weight_df(
            model=models[spec.label],
            model_label=spec.label,
            view_name=view_name,
            n_factors=spec.factor_count,
        )

        row_map = row_factor_maps[spec.label]
        present_count = 0
        for row_pos in range(1, max_factor + 1):
            source_factor_idx = row_map[row_pos - 1]
            if source_factor_idx is None:
                continue

            factor_col = f"F{source_factor_idx}"
            if factor_col not in weights_df.columns:
                continue

            series_map[f"{spec.label} F{row_pos}"] = pd.to_numeric(weights_df[factor_col], errors="coerce")
            present_count += 1

        block_sizes.append(present_count)

    corr_df = pd.DataFrame(series_map)
    return corr_df, block_sizes


def compute_best_match_weight_signed_corr_matrix(
    models,
    row_factor_maps,
    model_specs,
    max_factor,
    view_name,
    row_positions=None,
):
    # Weights are equivalent across combined/CFM/non-CFM, so only keep one model per factor count.
    # Compute all pairwise comparisons across factor counts (10/8/6/4) on the same global rows.
    combined_specs = sorted(
        [spec for spec in model_specs if spec.variant == "combined"],
        key=lambda s: s.factor_count,
    )
    if len(combined_specs) < 2:
        raise ValueError("At least two combined models are required to compute pairwise correlations.")

    pair_entries = []
    for i in range(len(combined_specs)):
        for j in range(i + 1, len(combined_specs)):
            pair_entries.append((combined_specs[i], combined_specs[j]))

    weights_by_model = {
        spec.label: get_first_n_weight_df(
            model=models[spec.label],
            model_label=spec.label,
            view_name=view_name,
            n_factors=spec.factor_count,
        )
        for spec in combined_specs
    }

    pair_labels = [f"{spec_a.factor_count} vs. {spec_b.factor_count}" for spec_a, spec_b in pair_entries]
    if row_positions is None:
        row_positions = list(range(1, max_factor + 1))
    row_labels = [f"F{row_pos}" for row_pos in row_positions]
    corr_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)
    p_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)

    for row_pos in row_positions:
        row_label = f"F{row_pos}"
        for spec_a, spec_b in pair_entries:
            factor_a = row_factor_maps[spec_a.label][row_pos - 1]
            factor_b = row_factor_maps[spec_b.label][row_pos - 1]
            pair_label = f"{spec_a.factor_count} vs. {spec_b.factor_count}"

            if factor_a is None or factor_b is None:
                continue

            col_a = f"F{factor_a}"
            col_b = f"F{factor_b}"
            if col_a not in weights_by_model[spec_a.label].columns:
                continue
            if col_b not in weights_by_model[spec_b.label].columns:
                continue

            series_a = pd.to_numeric(weights_by_model[spec_a.label][col_a], errors="coerce")
            series_b = pd.to_numeric(weights_by_model[spec_b.label][col_b], errors="coerce")
            corr_val, p_val = safe_pearson_corr_and_p(
                series_a.to_numpy(dtype=float),
                series_b.to_numpy(dtype=float),
            )
            if np.isfinite(corr_val):
                corr_matrix.loc[row_label, pair_label] = float(corr_val)
            if np.isfinite(p_val):
                p_matrix.loc[row_label, pair_label] = float(p_val)

    return corr_matrix, p_matrix


def compute_best_match_factor_signed_corr_matrix(
    aligned_scores,
    row_factor_maps,
    model_specs,
    max_factor,
    variant="combined",
    row_positions=None,
):
    # Best-match factor-score comparison across all pairwise factor counts within a variant.
    variant_specs = sorted(
        [spec for spec in model_specs if spec.variant == variant],
        key=lambda s: s.factor_count,
    )
    if len(variant_specs) < 2:
        raise ValueError(f"At least two '{variant}' models are required to compute pairwise correlations.")

    pair_entries = []
    for i in range(len(variant_specs)):
        for j in range(i + 1, len(variant_specs)):
            pair_entries.append((variant_specs[i], variant_specs[j]))

    pair_labels = [f"{spec_a.factor_count} vs. {spec_b.factor_count}" for spec_a, spec_b in pair_entries]
    if row_positions is None:
        row_positions = list(range(1, max_factor + 1))
    row_labels = [f"F{row_pos}" for row_pos in row_positions]
    corr_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)
    p_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)

    for row_pos in row_positions:
        row_label = f"F{row_pos}"
        for spec_a, spec_b in pair_entries:
            factor_a = row_factor_maps[spec_a.label][row_pos - 1]
            factor_b = row_factor_maps[spec_b.label][row_pos - 1]
            pair_label = f"{spec_a.factor_count} vs. {spec_b.factor_count}"

            if factor_a is None or factor_b is None:
                continue

            col_a = f"F{factor_a}"
            col_b = f"F{factor_b}"
            if col_a not in aligned_scores[spec_a.label].columns:
                continue
            if col_b not in aligned_scores[spec_b.label].columns:
                continue

            series_a = pd.to_numeric(aligned_scores[spec_a.label][col_a], errors="coerce")
            series_b = pd.to_numeric(aligned_scores[spec_b.label][col_b], errors="coerce")
            corr_val, p_val = safe_pearson_corr_and_p(
                series_a.to_numpy(dtype=float),
                series_b.to_numpy(dtype=float),
            )
            if np.isfinite(corr_val):
                corr_matrix.loc[row_label, pair_label] = float(corr_val)
            if np.isfinite(p_val):
                p_matrix.loc[row_label, pair_label] = float(p_val)

    return corr_matrix, p_matrix


def plot_signed_correlation_heatmap(corr_df, block_sizes, out_file):
    corr_signed = corr_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(cm(34), cm(26)))
    sns.heatmap(
        corr_signed,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.05,
        linecolor="white",
        cbar=True,
        ax=ax,
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Signed Pearson r"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    ax.tick_params(axis="both", which="both", width=0.5, length=1.5, labelsize=5)

    boundaries = np.cumsum(block_sizes)[:-1]
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.5)
        ax.axvline(boundary, color="black", linewidth=0.5)

    total_n = corr_signed.shape[0]
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            total_n,
            total_n,
            fill=False,
            edgecolor="black",
            linewidth=0.5,
        )
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_best_match_signed_correlation_heatmap(corr_matrix, out_file, anchor_label):
    fig, ax = plt.subplots(figsize=(cm(10), cm(8)))
    heat_ax = sns.heatmap(
        corr_matrix,
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.05,
        linecolor="white",
        cbar=True,
        ax=ax,
        annot=False,
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Pearson r"},
    )

    n_rows, n_cols = corr_matrix.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = corr_matrix.iat[row_idx, col_idx]
            if np.isfinite(value):
                label_text = f"{value:.2f}"
                text_color = "white" if abs(value) > 0.4 else "black"
            else:
                label_text = "N/A"
                text_color = "black"

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label_text,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.tick_params(axis="both", which="both", width=0.5, length=1.5, labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel(f"Global row factor ({anchor_label})")

    cbar = heat_ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson r", size=7)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_best_match_corr_all(
    factor_corr_matrix_by_variant,
    weight_corr_matrix_gene,
    weight_corr_matrix_protein,
    out_file,
    anchor_label,
):
    panel_data = [
        ("Factors combined", factor_corr_matrix_by_variant["combined"]),
        ("Factors CFM", factor_corr_matrix_by_variant["cfm"]),
        ("Factors non-CFM", factor_corr_matrix_by_variant["non-cfm"]),
        ("Weights gene expression", weight_corr_matrix_gene),
        ("Weights protein expression", weight_corr_matrix_protein),
    ]

    fig = plt.figure(figsize=(cm(18), cm(7)))
    grid = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    empty_ax = fig.add_subplot(grid[1, 2])
    empty_ax.axis("off")

    for panel_idx, (panel_title, matrix) in enumerate(panel_data):
        ax = axes[panel_idx]
        matrix_plot = matrix.T
        sns.heatmap(
            matrix_plot,
            cmap="seismic",
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.05,
            linecolor="white",
            cbar=False,
            ax=ax,
            annot=False,
        )

        n_rows, n_cols = matrix_plot.shape
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                value = matrix_plot.iat[row_idx, col_idx]
                if np.isfinite(value):
                    label_text = f"{value:.2f}"
                    text_color = "white" if abs(value) > 0.4 else "black"
                else:
                    label_text = "N/A"
                    text_color = "black"

                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )

        ax.set_title(panel_title, fontsize=6, pad=4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=7)
        if panel_idx in [0, 3]:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.tick_params(axis="both", which="both", width=0.5, length=1.5, labelsize=7)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    plt.tight_layout(w_pad=0.5, h_pad=0.8)
    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_correlation_heatmap(corr_df, block_sizes, out_file):
    corr_signed = corr_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(cm(34), cm(26)))
    sns.heatmap(
        corr_signed,
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.05,
        linecolor="white",
        cbar=False,
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    ax.tick_params(axis="both", which="both", width=0.5, length=1.5, labelsize=5)

    boundaries = np.cumsum(block_sizes)[:-1]
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.5)
        ax.axvline(boundary, color="black", linewidth=0.5)

    total_n = corr_signed.shape[0]
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            total_n,
            total_n,
            fill=False,
            edgecolor="black",
            linewidth=0.5,
        )
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_variance_heatmaps(
    variance_df,
    row_factor_maps,
    model_specs,
    max_factor,
    out_file_prefix,
    anchor_label,
    total_variance_by_model,
):
    variant_rank = {"combined": 0, "cfm": 1, "non-cfm": 2}
    ordered_specs = sorted(
        model_specs,
        key=lambda s: (variant_rank.get(s.variant, 99), -s.factor_count),
    )
    model_order = [spec.label for spec in ordered_specs]

    views = [view for view in ["gene_exp", "protein"] if view in variance_df["view"].unique()]
    if not views:
        views = sorted(variance_df["view"].unique().tolist())

    view_display_names = {
        "gene_exp": "gene expression",
        "protein": "protein expression",
    }

    out_files = []
    for view_name in views:
        subset = variance_df[variance_df["view"] == view_name].copy()
        view_label = view_display_names.get(str(view_name), str(view_name).replace("_", " "))

        row_labels = [f"F{i}" for i in range(1, max_factor + 1)]
        heatmap_df = pd.DataFrame(index=row_labels, columns=model_order, dtype=float)

        for model_label in model_order:
            for row_pos in range(1, max_factor + 1):
                source_factor_idx = row_factor_maps[model_label][row_pos - 1]
                if source_factor_idx is None:
                    continue

                row = subset[(subset["model"] == model_label) & (subset["factor_idx"] == source_factor_idx)]
                if row.empty:
                    continue

                value_pct = float(row["variance_ratio"].iloc[0]) * 100.0
                heatmap_df.loc[f"F{row_pos}", model_label] = value_pct

        finite_values = heatmap_df.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        vmax = float(np.max(finite_values)) if finite_values.size > 0 else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(cm(18), cm(8.8)))
        heat_ax = sns.heatmap(
            heatmap_df,
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            linewidths=0.05,
            linecolor="white",
            cbar=True,
            ax=ax,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 6},
            cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Variance explained (%)"},
        )

        for row_idx in range(heatmap_df.shape[0]):
            for col_idx in range(heatmap_df.shape[1]):
                if not np.isfinite(heatmap_df.iat[row_idx, col_idx]):
                    ax.text(
                        col_idx + 0.5,
                        row_idx + 0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )

        # Display per-model total variance above each heatmap column.
        column_totals = []
        for model_label in model_order:
            model_totals = total_variance_by_model.get(model_label, {})
            column_totals.append(model_totals.get(view_name, np.nan))

        top_axis = ax.secondary_xaxis("top")
        top_axis.set_xticks(np.arange(heatmap_df.shape[1]) + 0.5)
        top_axis.set_xticklabels(
            [f"{val:.1f}" if np.isfinite(val) else "N/A" for val in column_totals],
            fontsize=6,
        )
        top_axis.tick_params(axis="x", which="both", length=0, pad=1, labelsize=6)
        for spine in top_axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        cbar = heat_ax.collections[0].colorbar
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(width=0.5, length=1.5)
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        ax.set_xlabel("")
        ax.set_ylabel(f"Global row factor ({anchor_label})")
        ax.set_title(f"Variance explained heatmap ({view_label})", fontsize=7, pad=4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        ax.tick_params(axis="both", which="both", width=0.5, length=1.5)

        for boundary in [4, 8]:
            if boundary < len(model_order):
                ax.axvline(boundary, color="black", linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        plt.tight_layout()
        safe_view = re.sub(r"[^A-Za-z0-9_]+", "_", str(view_label))
        out_file = out_file_prefix.parent / f"{out_file_prefix.stem}_{safe_view}{out_file_prefix.suffix}"
        fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        out_files.append(out_file)

    return out_files


def main():
    args = get_parser().parse_args()
    set_plot_style()

    model_specs = get_model_specs()
    max_supported_factor = max(spec.factor_count for spec in model_specs)
    max_factor = min(args.max_factor, max_supported_factor)

    if args.max_factor != max_factor:
        print(f"Warning: max_factor={args.max_factor} reduced to supported max_factor={max_factor}.")

    args.summary_dir.mkdir(parents=True, exist_ok=True)

    mdata = mu.read_h5mu(args.data_dir / "processed_mdata.h5mu")
    adata_plot = mdata.mod["gene_exp"].copy()
    adata_plot.obsm["spatial"] = mdata.obs.loc[adata_plot.obs_names, ["spatial1", "spatial2"]].to_numpy()

    models = {}
    score_dfs = {}
    aligned_scores = {}
    variance_frames = []
    spatial_row_stats_by_variant = {}
    weight_p_by_view = {}

    for spec in model_specs:
        model_path = spec.run_dir / spec.filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = mofax.mofa_model(model_path)
        models[spec.label] = model

        factor_df = get_first_n_factor_df(model, spec.factor_count)
        score_dfs[spec.label] = factor_df
        aligned_scores[spec.label] = align_scores_to_obs(adata_plot.obs_names, factor_df, spec.label)

        variance_frames.append(extract_variance_by_view(model, spec.label, n_factors=spec.factor_count))

    total_variance_by_model = {
        spec.label: extract_total_variance_by_view(models[spec.label]) for spec in model_specs
    }

    row_factor_maps, row_factor_ids, ref_label = build_global_row_factor_maps(aligned_scores, model_specs)
    for spec in model_specs:
        print(f"{spec.label}: row mapping -> source factors {row_factor_maps[spec.label]}")
    print(f"Global row reference model: {ref_label}")

    adata_plot_mi = prepare_rowstd_spatial_neighbors(adata_plot.copy())

    spatial_variants = ["combined", "cfm", "non-cfm"]
    for variant in spatial_variants:
        grid_out = args.summary_dir / f"mofty_spatial_grid_{variant.replace('-', '_')}.pdf"
        row_stats = plot_spatial_grid(
            adata_plot=adata_plot,
            adata_plot_mi=adata_plot_mi,
            aligned_scores=aligned_scores,
            row_factor_maps=row_factor_maps,
            row_factor_ids=row_factor_ids,
            model_specs=model_specs,
            variant=variant,
            max_factor=max_factor,
            out_file=grid_out,
        )
        spatial_row_stats_by_variant[variant] = row_stats
        print(f"Saved spatial grid ({variant}) to: {grid_out}")

    corr_df, block_sizes = build_correlation_dataframe(aligned_scores, row_factor_maps, row_factor_ids, model_specs)
    corr_csv_out = args.summary_dir / "mofty_factor_correlation_abs_input.csv"
    corr_df.to_csv(corr_csv_out, index=True)

    corr_out = args.summary_dir / "mofty_factor_correlation_heatmap.pdf"
    plot_correlation_heatmap(corr_df, block_sizes=block_sizes, out_file=corr_out)
    print(f"Saved correlation input matrix to: {corr_csv_out}")
    print(f"Saved correlation heatmap to: {corr_out}")

    factor_counts = sorted({spec.factor_count for spec in model_specs})
    shared_row_max = factor_counts[-2] if len(factor_counts) >= 2 else max_factor
    shared_row_max = min(max_factor, shared_row_max)
    best_match_row_positions = list(range(1, shared_row_max + 1))

    factor_best_corr_by_variant = {}
    for variant in ["combined", "cfm", "non-cfm"]:
        factor_best_corr_matrix, _ = compute_best_match_factor_signed_corr_matrix(
            aligned_scores=aligned_scores,
            row_factor_maps=row_factor_maps,
            model_specs=model_specs,
            max_factor=max_factor,
            variant=variant,
            row_positions=best_match_row_positions,
        )
        factor_best_corr_by_variant[variant] = factor_best_corr_matrix

        factor_best_corr_csv_out = args.summary_dir / f"mofty_factor_correlation_best_match_{variant}.csv"
        factor_best_corr_matrix.to_csv(factor_best_corr_csv_out, index=True)
        print(f"Saved signed best-match factor correlation matrix ({variant}) to: {factor_best_corr_csv_out}")

    # Backward-compatible combined-only CSV path.
    factor_best_corr_csv_out = args.summary_dir / "mofty_factor_correlation_best_match.csv"
    factor_best_corr_by_variant["combined"].to_csv(factor_best_corr_csv_out, index=True)
    print(f"Saved signed best-match factor correlation matrix to: {factor_best_corr_csv_out}")

    weight_view_display_names = {
        "gene_exp": "gene expression",
        "protein": "protein expression",
    }
    weight_corr_by_view = {}
    for view_name in ["gene_exp", "protein"]:
        display_name = weight_view_display_names[view_name]

        weight_corr_matrix, weight_p_matrix = compute_best_match_weight_signed_corr_matrix(
            models=models,
            row_factor_maps=row_factor_maps,
            model_specs=model_specs,
            max_factor=max_factor,
            view_name=view_name,
            row_positions=best_match_row_positions,
        )

        weight_corr_csv_out = args.summary_dir / f"mofty_weight_correlation_best_match_{view_name}.csv"
        weight_corr_matrix.to_csv(weight_corr_csv_out, index=True)
        weight_p_by_view[display_name] = weight_p_matrix.copy()
        weight_corr_by_view[display_name] = weight_corr_matrix.copy()
        print(f"Saved signed best-match weight correlation matrix ({view_name}) to: {weight_corr_csv_out}")

    best_match_triptych_out = args.summary_dir / "mofty_best_match_correlation.pdf"
    plot_best_match_corr_all(
        factor_corr_matrix_by_variant=factor_best_corr_by_variant,
        weight_corr_matrix_gene=weight_corr_by_view["gene expression"],
        weight_corr_matrix_protein=weight_corr_by_view["protein expression"],
        out_file=best_match_triptych_out,
        anchor_label=ref_label,
    )
    print(f"Saved best-match correlation to: {best_match_triptych_out}")

    p_txt_out = args.summary_dir / "mofty_significance_values.txt"
    write_p_values_summary_txt(
        out_file=p_txt_out,
        spatial_row_stats_by_variant=spatial_row_stats_by_variant,
        weight_p_by_view=weight_p_by_view,
    )
    print(f"Saved significance values to: {p_txt_out}")

    variance_df = pd.concat(variance_frames, axis=0, ignore_index=True)
    variance_df["variance_ratio"] = pd.to_numeric(variance_df["variance_ratio"], errors="coerce")
    variance_df = variance_df.dropna(subset=["variance_ratio"])

    variance_csv_out = args.summary_dir / "mofty_variance_explained_by_view.csv"
    variance_df.to_csv(variance_csv_out, index=False)

    variance_out_prefix = args.summary_dir / "mofty_variance_explained_heatmap.pdf"
    variance_out_files = plot_variance_heatmaps(
        variance_df=variance_df,
        row_factor_maps=row_factor_maps,
        model_specs=model_specs,
        max_factor=max_factor,
        out_file_prefix=variance_out_prefix,
        anchor_label=ref_label,
        total_variance_by_model=total_variance_by_model,
    )
    print(f"Saved variance summary to: {variance_csv_out}")
    for variance_out in variance_out_files:
        print(f"Saved variance plot to: {variance_out}")


if __name__ == "__main__":
    main()
