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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
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
from sklearn.decomposition import PCA


def cm(value):
    return value / 2.54


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fullGP", help="MEFISTO full GP model suffix.")
    parser.add_argument("--model_3k", type=str, default="3000", help="MEFISTO 3000 model suffix.")
    parser.add_argument("--model_1k", type=str, default="1000", help="MEFISTO 1000 model suffix.")
    parser.add_argument("--n_factors", type=int, default=4, help="Number of factors/PCs to compare.")
    return parser


def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
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


def get_first_n_factor_df(model, n_factors, prefix):
    factors_df = model.get_factors(df=True).copy()
    factor_cols = [
        col for col in sort_factor_columns(factors_df.columns) if re.match(r"^Factor\d+$", str(col))
    ]
    if len(factor_cols) < n_factors:
        factor_cols = sort_factor_columns(list(factors_df.columns))[:n_factors]

    if len(factor_cols) < n_factors:
        raise ValueError(f"Model has only {len(factor_cols)} factors, but {n_factors} are required.")

    out = factors_df[factor_cols].copy()
    out.columns = [f"{prefix}F{i + 1}" for i in range(n_factors)]
    return out


def get_first_n_weight_df(model, n_factors, view_name):
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
            f"Could not load weights for view '{view_name}'. Tried views: {candidate_views}. Last error: {last_error}"
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
            f"Weight matrix has only {len(factor_cols)} factor columns, but {n_factors} are required "
            f"for view '{view_name}'."
        )

    out = weights_df[factor_cols[:n_factors]].copy()
    out.columns = [f"F{i + 1}" for i in range(n_factors)]
    return out


def reorder_factor_columns(factor_df, source_factor_order):
    ordered = pd.DataFrame(index=factor_df.index)
    for display_pos, source_factor_idx in enumerate(source_factor_order, start=1):
        source_col = f"F{source_factor_idx}"
        if source_col not in factor_df.columns:
            raise KeyError(f"Missing expected factor column '{source_col}' in factor table.")
        ordered[f"F{display_pos}"] = pd.to_numeric(factor_df[source_col], errors="coerce")
    return ordered


def build_standard_factor_df_from_cols(obs_df, cols):
    out = pd.DataFrame(index=obs_df.index)
    for row_pos, col in enumerate(cols, start=1):
        out[f"F{row_pos}"] = pd.to_numeric(obs_df[col], errors="coerce")
    return out


def build_pca_weight_dfs(adata_pca, pca_model, feature_mask, protein_mask, pca_row_order, n_factors):
    feature_mask_values = np.asarray(getattr(feature_mask, "values", feature_mask), dtype=bool)
    protein_mask_values = np.asarray(getattr(protein_mask, "values", protein_mask), dtype=bool)

    selected_feature_names = adata_pca.var_names[feature_mask_values]
    selected_is_protein = protein_mask_values[feature_mask_values]

    weights_all = pd.DataFrame(index=selected_feature_names)
    for row_pos, pc_idx in enumerate(pca_row_order[:n_factors], start=1):
        weights_all[f"F{row_pos}"] = pca_model.components_[pc_idx - 1, :]

    gene_exp_weights = weights_all.loc[~selected_is_protein].copy()
    protein_weights = weights_all.loc[selected_is_protein].copy()
    return {
        "gene expression": gene_exp_weights,
        "protein expression": protein_weights,
    }


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


def compute_abs_corr_matrix(reference_df, target_df):
    n_ref = reference_df.shape[1]
    n_target = target_df.shape[1]
    corr_abs = np.zeros((n_ref, n_target), dtype=float)

    for i in range(n_ref):
        ref_values = pd.to_numeric(reference_df.iloc[:, i], errors="coerce")
        for j in range(n_target):
            target_values = pd.to_numeric(target_df.iloc[:, j], errors="coerce")
            corr_val, _ = safe_pearson_corr_and_p(ref_values.to_numpy(dtype=float), target_values.to_numpy(dtype=float))
            corr_abs[i, j] = abs(corr_val) if np.isfinite(corr_val) else 0.0

    return corr_abs


def match_to_reference_rows(reference_df, target_df):
    corr_abs = compute_abs_corr_matrix(reference_df, target_df)
    row_idx, col_idx = linear_sum_assignment(-corr_abs)
    return {int(r) + 1: int(c) + 1 for r, c in zip(row_idx, col_idx)}


def build_best_match_row_maps(reference_label, score_matrices):
    if reference_label not in score_matrices:
        raise KeyError(f"Reference label '{reference_label}' is missing from score matrices.")

    reference_df = score_matrices[reference_label]
    n_rows = reference_df.shape[1]
    row_factor_maps = {reference_label: list(range(1, n_rows + 1))}

    for model_label, target_df in score_matrices.items():
        if model_label == reference_label:
            continue
        row_to_factor = match_to_reference_rows(reference_df, target_df)
        factor_map = [None] * n_rows
        for row_idx, target_factor in row_to_factor.items():
            factor_map[row_idx - 1] = target_factor
        row_factor_maps[model_label] = factor_map

    return row_factor_maps


def compute_best_match_signed_corr_matrix(
    value_matrices,
    row_factor_maps,
    row_labels,
    align_on_index=False,
    anchor_label="PCA",
):
    model_labels = list(value_matrices.keys())
    if anchor_label not in model_labels:
        raise KeyError(f"Anchor label '{anchor_label}' is missing from value matrices.")

    comparison_labels = [label for label in model_labels if label != anchor_label]
    pair_entries = []

    # Keep anchor comparisons first for readability, then include all non-anchor pairs.
    for model_b in comparison_labels:
        pair_entries.append((anchor_label, model_b))
    for i in range(len(comparison_labels)):
        for j in range(i + 1, len(comparison_labels)):
            pair_entries.append((comparison_labels[i], comparison_labels[j]))

    pair_labels = [f"{model_a} vs. {model_b}" for model_a, model_b in pair_entries]

    corr_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)
    p_matrix = pd.DataFrame(index=row_labels, columns=pair_labels, dtype=float)

    for row_pos, row_label in enumerate(row_labels, start=1):
        for model_a, model_b in pair_entries:
            factor_a = row_factor_maps[model_a][row_pos - 1]
            factor_b = row_factor_maps[model_b][row_pos - 1]
            pair_label = f"{model_a} vs. {model_b}"

            if factor_a is None or factor_b is None:
                continue

            col_a = f"F{factor_a}"
            col_b = f"F{factor_b}"
            if col_a not in value_matrices[model_a].columns or col_b not in value_matrices[model_b].columns:
                continue

            series_a = pd.to_numeric(value_matrices[model_a][col_a], errors="coerce")
            series_b = pd.to_numeric(value_matrices[model_b][col_b], errors="coerce")

            if align_on_index:
                common_index = series_a.index.intersection(series_b.index)
                if len(common_index) < 3:
                    continue
                x_vals = series_a.loc[common_index].to_numpy(dtype=float)
                y_vals = series_b.loc[common_index].to_numpy(dtype=float)
            else:
                x_vals = series_a.to_numpy(dtype=float)
                y_vals = series_b.to_numpy(dtype=float)

            corr_val, p_val = safe_pearson_corr_and_p(x_vals, y_vals)
            if np.isfinite(corr_val):
                corr_matrix.loc[row_label, pair_label] = float(corr_val)
            if np.isfinite(p_val):
                p_matrix.loc[row_label, pair_label] = float(p_val)

    return corr_matrix, p_matrix


def plot_best_match_signed_correlation_panels(corr_matrices, panel_titles, out_file):
    if len(corr_matrices) != len(panel_titles):
        raise ValueError("corr_matrices and panel_titles must have the same length.")

    n_panels = len(corr_matrices)
    fig, axes = plt.subplots(1, n_panels, figsize=(cm(5.2 * n_panels), cm(7.4)), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, (ax, corr_matrix, title) in enumerate(zip(axes, corr_matrices, panel_titles)):
        # Rotate orientation so model pairs are on the shared y-axis.
        plot_matrix = corr_matrix.T

        heat_ax = sns.heatmap(
            plot_matrix,
            cmap="seismic",
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.05,
            linecolor="white",
            cbar=None,
            ax=ax,
            annot=False,
        )

        n_rows, n_cols = plot_matrix.shape
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                value = plot_matrix.iat[row_idx, col_idx]
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

        ax.set_title(title, fontsize=6, pad=4)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        if panel_idx == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="both", which="both", width=0.5, length=1.5, labelsize=7)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    plt.tight_layout(w_pad=0.4)
    fig.savefig(out_file, transparent=True, bbox_inches="tight")
    plt.close(fig)


def format_float_for_txt(value):
    if not np.isfinite(value):
        return "N/A"
    return f"{float(value):.1e}"


def format_matrix_for_txt(matrix):
    return matrix.apply(lambda col: col.map(format_float_for_txt))


def write_best_match_pvalues_txt(out_file, score_p_matrix, weight_p_by_view):
    lines = []
    lines.append("Best-match significance summary")
    lines.append("")
    lines.append("Factor/PC best-match p-values")
    lines.append(format_matrix_for_txt(score_p_matrix).to_csv(sep="\t", index=True))

    for view_name in ["gene expression", "protein expression"]:
        lines.append("")
        lines.append(f"Weight best-match p-values ({view_name})")
        p_matrix = weight_p_by_view.get(view_name)
        if p_matrix is None or p_matrix.empty:
            lines.append("N/A")
        else:
            lines.append(format_matrix_for_txt(p_matrix).to_csv(sep="\t", index=True))

    out_file.write_text("\n".join(lines), encoding="utf-8")


def join_obs_columns(adata, score_df):
    aligned = score_df.reindex(adata.obs_names)
    if aligned.isna().any().any():
        missing_rows = int(aligned.isna().any(axis=1).sum())
        print(f"Warning: {missing_rows} samples had missing scores after alignment; filling with 0.")
        aligned = aligned.fillna(0.0)

    overlap = [col for col in aligned.columns if col in adata.obs.columns]
    if overlap:
        adata.obs = adata.obs.drop(columns=overlap)

    adata.obs = adata.obs.join(aligned)
    return adata


def get_mefisto_scales(model, n_factors):
    scales = np.asarray(model.model["training_stats"]["scales"][()])
    scales = np.squeeze(scales)
    scales = np.ravel(scales).astype(float)

    if scales.size < n_factors:
        scales = np.pad(scales, (0, n_factors - scales.size), constant_values=np.nan)

    return scales[:n_factors]


def compute_joint_pca_scores(adata_pca, n_factors):
    protein_mask = adata_pca.var["feat_modality"] == "protein"
    feature_mask = adata_pca.var["highly_variable"] | protein_mask

    X_concat = adata_pca[:, feature_mask].to_df().to_numpy(dtype=np.float64)

    feature_mask_idx = np.where(feature_mask.values)[0]
    modality_is_gene_exp = adata_pca.var["highly_variable"].values & ~protein_mask.values
    modality_is_protein = protein_mask.values
    gene_exp_idx = np.where(modality_is_gene_exp)[0]
    protein_idx = np.where(modality_is_protein)[0]

    gene_exp_cols = np.isin(feature_mask_idx, gene_exp_idx)
    protein_cols = np.isin(feature_mask_idx, protein_idx)

    # 1) Center each feature individually across samples.
    feature_means = np.nanmean(X_concat, axis=0)
    feature_means = np.where(np.isnan(feature_means), 0.0, feature_means)
    X_concat = X_concat - feature_means

    gene_exp_std = np.nanstd(X_concat[:, gene_exp_cols])
    protein_std = np.nanstd(X_concat[:, protein_cols])

    if gene_exp_std > 0:
        X_concat[:, gene_exp_cols] = X_concat[:, gene_exp_cols] / gene_exp_std
    if protein_std > 0:
        X_concat[:, protein_cols] = X_concat[:, protein_cols] / protein_std

    print(
        "Applied preprocessing: feature-wise centering, then modality-wise scaling with "
        f"global stds (gene_exp={gene_exp_std:.4g}, protein={protein_std:.4g})"
    )
    print(f"Concatenated matrix for PCA shape: {X_concat.shape}")

    max_comps = min(30, X_concat.shape[0], X_concat.shape[1])
    if max_comps < n_factors:
        raise ValueError(f"Only {max_comps} PCA components are available, but {n_factors} are required.")

    pca_model = PCA(n_components=max_comps, random_state=42)
    X_pca = pca_model.fit_transform(X_concat)

    pc_cols = [f"PCA_PC{i + 1}" for i in range(n_factors)]
    pca_scores_df = pd.DataFrame(X_pca[:, :n_factors], index=adata_pca.obs_names, columns=pc_cols)

    return {
        "pca_scores_df": pca_scores_df,
        "pca_model": pca_model,
        "X_concat": X_concat,
        "X_pca": X_pca,
        "feature_mask": feature_mask,
        "protein_mask": protein_mask,
    }


def compute_pca_modality_variance(adata_pca, X_concat, X_pca, pca_model, feature_mask, protein_mask, n_factors):
    feature_mask_idx = np.where(feature_mask.values)[0]

    modality_is_gene_exp = adata_pca.var["highly_variable"].values & ~protein_mask.values
    modality_is_protein = protein_mask.values

    gene_exp_idx = np.where(modality_is_gene_exp)[0]
    protein_idx = np.where(modality_is_protein)[0]

    gene_exp_cols = np.isin(feature_mask_idx, gene_exp_idx)
    protein_cols = np.isin(feature_mask_idx, protein_idx)

    X_gene_exp = X_concat[:, gene_exp_cols]
    X_protein = X_concat[:, protein_cols]

    mean_all = pca_model.mean_
    mean_gene_exp = mean_all[gene_exp_cols]
    mean_protein = mean_all[protein_cols]

    components_top = pca_model.components_[:n_factors]
    components_gene_exp = components_top[:, gene_exp_cols]
    components_protein = components_top[:, protein_cols]

    ss_tot_gene_exp = np.sum((X_gene_exp - mean_gene_exp) ** 2)
    ss_tot_protein = np.sum((X_protein - mean_protein) ** 2)

    gene_exp_cum = []
    protein_cum = []

    for k in range(1, n_factors + 1):
        scores_k = X_pca[:, :k]

        recon_gene_exp = mean_gene_exp + scores_k @ components_gene_exp[:k, :]
        recon_protein = mean_protein + scores_k @ components_protein[:k, :]

        ss_res_gene_exp = np.sum((X_gene_exp - recon_gene_exp) ** 2)
        ss_res_protein = np.sum((X_protein - recon_protein) ** 2)

        if ss_tot_gene_exp > 0:
            gene_exp_cum.append(1.0 - (ss_res_gene_exp / ss_tot_gene_exp))
        else:
            gene_exp_cum.append(0.0)

        if ss_tot_protein > 0:
            protein_cum.append(1.0 - (ss_res_protein / ss_tot_protein))
        else:
            protein_cum.append(0.0)

    gene_exp_var = [gene_exp_cum[0]] + [gene_exp_cum[i] - gene_exp_cum[i - 1] for i in range(1, len(gene_exp_cum))]
    protein_var = [protein_cum[0]] + [protein_cum[i] - protein_cum[i - 1] for i in range(1, len(protein_cum))]

    return {
        "gene_exp": gene_exp_var,
        "protein": protein_var,
    }


def parse_factor_idx(value):
    if isinstance(value, (int, np.integer)):
        if value >= 1:
            return int(value)
        return int(value) + 1

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if value >= 1.0:
            return int(round(value))
        return int(round(value)) + 1

    text = str(value)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return np.nan


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
    out["factor_idx"] = out["factor"].apply(parse_factor_idx)
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
    out["component"] = [f"F{i}" for i in out["factor_idx"]]
    return out[["model", "view", "factor_idx", "component", "variance_ratio"]]


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


def clean_spatial_axis(ax, keep_ylabel=False):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    if not keep_ylabel:
        ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, labelsize=7)
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


def draw_vertical_model_boundaries(fig, axes, boundaries_after_cols):
    if not boundaries_after_cols:
        return

    y_top = axes[0, 0].get_position().y1
    y_bottom = axes[-1, 0].get_position().y0

    for boundary_idx in boundaries_after_cols:
        left_ax = axes[0, boundary_idx - 1]
        right_ax = axes[0, boundary_idx]
        x_mid = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2.0
        fig.add_artist(
            plt.Line2D(
                [x_mid, x_mid],
                [y_bottom, y_top],
                transform=fig.transFigure,
                color="black",
                linewidth=1.0,
            )
        )


def plot_spatial_grid(adata_plot, adata_plot_mi, column_specs, pca_row_order, out_file):
    n_factors = len(column_specs[0]["cols"])
    n_rows = len(column_specs)
    n_cols = n_factors

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cm(11), cm(4 * n_rows)))
    fig.subplots_adjust(wspace=0.0, hspace=0.1)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # For each component column, use one shared color scale across all MOFTy variants.
    mofty_col_vmax = [None] * n_cols
    for col_idx in range(n_cols):
        mofty_col_values = []
        for spec in column_specs:
            if spec["kind"] == "mofty":
                mofty_col_values.append(adata_plot.obs[spec["cols"][col_idx]].to_numpy(dtype=float))
        if mofty_col_values:
            mofty_concat = np.concatenate(mofty_col_values)
            vmax = np.nanmax(np.abs(mofty_concat))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1e-6
            mofty_col_vmax[col_idx] = vmax

    for row_idx, row_spec in enumerate(column_specs):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            score_col = row_spec["cols"][col_idx]
            values = adata_plot.obs[score_col].to_numpy(dtype=float)

            if row_spec["kind"] == "mofty" and mofty_col_vmax[col_idx] is not None:
                vmax = mofty_col_vmax[col_idx]
                vmin = -mofty_col_vmax[col_idx]
            else:
                vmax = np.nanmax(np.abs(values))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1e-6
                vmin = -vmax

            mi = sc.metrics.morans_i(adata_plot_mi, vals=values)

            sc.pl.spatial(
                adata_plot,
                img_key=None,
                color=score_col,
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

            if row_spec["kind"] == "pca":
                label_text = f"$I$: {mi:.2f}"
            elif row_spec["kind"] == "mefisto":
                scale = row_spec["scales"][col_idx]
                if np.isfinite(scale):
                    label_text = f"$S$: {scale:.2f}, $I$: {mi:.2f}"
                else:
                    label_text = f"$S$: n/a, $I$: {mi:.2f}"
            else:
                label_text = f"$I$: {mi:.2f}"

            ax.text(
                0.02,
                0.98,
                label_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                bbox={"facecolor": "none", "edgecolor": "none", "pad": 1.2},
            )

            if row_spec["kind"] == "pca":
                pca_id = pca_row_order[col_idx] if len(pca_row_order) == n_cols else (col_idx + 1)
                col_heading = f"PC{pca_id}"
            else:
                factor_labels = row_spec.get("factor_labels")
                factor_id = factor_labels[col_idx] if factor_labels is not None else (col_idx + 1)
                col_heading = f"F{factor_id}"

            ax.text(
                0.5,
                1.01,
                col_heading,
                transform=ax.transAxes,
                va="bottom",
                ha="center",
                fontsize=7,
            )

            clean_spatial_axis(ax, keep_ylabel=(col_idx == 0))

            if col_idx == 0:
                ax.set_ylabel(row_spec["header"], fontsize=7, rotation=90, labelpad=8)
                ax.yaxis.set_label_coords(-0.01, 0.5)

    fig.savefig(out_file, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_correlation_heatmap(corr_df, out_file, block_sizes):
    corr_signed = corr_df.corr(method="pearson")
    corr_values = corr_signed.to_numpy()

    fig, ax = plt.subplots(figsize=(cm(18), cm(18)))
    sns.heatmap(
        corr_signed,
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.2,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # Draw model block separators.
    boundaries = np.cumsum(block_sizes)[:-1]
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.5)
        ax.axvline(boundary, color="black", linewidth=0.5)

    # Draw an outer border around the full heatmap.
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

    # Keep correlation tick marks short and thin.
    ax.tick_params(axis="both", which="both", width=0.5, length=1.5)

    for row_idx in range(total_n):
        for col_idx in range(total_n):
            value = corr_values[row_idx, col_idx]
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

    plt.tight_layout()
    fig.savefig(out_file, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_correlation_colorbar(out_file):
    # Keep the color strip exactly 0.5 cm wide and 4 cm high,
    # but use a slightly wider canvas so tick labels are not clipped.
    bar_width_cm = 0.2
    fig_width_cm = 1
    fig_height_cm = 1.5
    fig = plt.figure(figsize=(cm(fig_width_cm), cm(fig_height_cm)))
    cax = fig.add_axes([0.05, 0.0, bar_width_cm / fig_width_cm, 1.0])

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="seismic")
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Pearson r", fontsize=6)
    cbar_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=5, width=0, length=0, pad=1)
    cbar.outline.set_visible(False)

    fig.savefig(out_file, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_variance_bars_by_modality(
    variance_df,
    summary_dir,
    model_name,
    n_factors,
    pca_row_order,
    model_factor_orders=None,
):
    model_order = [
        "PCA",
        "MEFISTO full GP",
        "MEFISTO 3K",
        "MEFISTO 1K",
        "MOFTy combined",
        "MOFTy CFM",
        "MOFTy non-CFM",
    ]
    model_colors = {
        "PCA": "#9b9b9b",
        "MEFISTO full GP": "#DA0000",
        "MEFISTO 3K": "#FF7C30",
        "MEFISTO 1K": "#FFE100",
        "MOFTy combined": "#0143A6",
        "MOFTy CFM": "#1182E6",
        "MOFTy non-CFM": "#71C0EE",
    }

    views = [v for v in ["gene_exp", "protein"] if v in variance_df["view"].unique()]
    factor_ids = list(range(1, n_factors + 1))
    if model_factor_orders is None:
        mefisto_1k_order = list(factor_ids)
        if n_factors >= 4:
            mefisto_1k_order[2], mefisto_1k_order[3] = mefisto_1k_order[3], mefisto_1k_order[2]
        model_factor_orders = {"MEFISTO 1K": mefisto_1k_order}
    n_models = len(model_order)
    bar_width = 0.1
    group_spacing = 0.8

    for view in views:
        subset = variance_df[variance_df["view"] == view].copy()
        if subset.empty:
            continue

        x = np.arange(n_factors, dtype=float) * group_spacing
        all_values = []

        fig, ax = plt.subplots(figsize=(cm(18), cm(6)))
        for model_idx, model_name_label in enumerate(model_order):
            offsets = (model_idx - (n_models - 1) / 2.0) * bar_width
            x_model = x + offsets
            y_model = []
            inside_labels = []
            source_factor_order = model_factor_orders.get(model_name_label, factor_ids)
            for display_pos, source_factor_idx in enumerate(source_factor_order, start=1):
                row = subset[(subset["model"] == model_name_label) & (subset["factor_idx"] == source_factor_idx)]
                if row.empty:
                    y_model.append(np.nan)
                else:
                    y_model.append(float(row["variance_ratio"].iloc[0]) * 100.0)

                if model_name_label == "PCA" and len(pca_row_order) == n_factors:
                    inside_labels.append(f"PC{pca_row_order[display_pos - 1]}")
                elif model_name_label == "PCA":
                    inside_labels.append(f"PC{display_pos}")
                else:
                    inside_labels.append(f"F{source_factor_idx}")

            y_plot = np.nan_to_num(np.asarray(y_model, dtype=float), nan=0.0)
            all_values.extend([v for v in y_model if np.isfinite(v)])

            bars = ax.bar(
                x_model,
                y_plot,
                width=bar_width,
                color=model_colors.get(model_name_label, "#7f7f7f"),
                edgecolor="black",
                linewidth=0.3,
                label=model_name_label,
            )

            for bar, value, inbar_label in zip(bars, y_model, inside_labels):
                if not np.isfinite(value):
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.25,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

        ymax = max(1.0, np.nanmax(all_values) if all_values else 1.0)
        ypad_bottom = ymax * 0.10
        ax.set_ylim(-ypad_bottom, ymax * 1.25)

        # Place PCx/Fx labels below each bar.
        label_y = -ypad_bottom * 0.14
        for model_idx, model_name_label in enumerate(model_order):
            offsets = (model_idx - (n_models - 1) / 2.0) * bar_width
            x_model = x + offsets

            source_factor_order = model_factor_orders.get(model_name_label, factor_ids)
            below_labels = []
            for display_pos, source_factor_idx in enumerate(source_factor_order, start=1):
                if model_name_label == "PCA" and len(pca_row_order) == n_factors:
                    below_labels.append(f"PC{pca_row_order[display_pos - 1]}")
                elif model_name_label == "PCA":
                    below_labels.append(f"PC{display_pos}")
                else:
                    below_labels.append(f"F{source_factor_idx}")

            for x_pos, lbl in zip(x_model, below_labels):
                ax.text(
                    x_pos,
                    label_y,
                    lbl,
                    ha="center",
                    va="top",
                    fontsize=6,
                    rotation=0,
                    color="black",
                    clip_on=False,
                )

        ax.axhline(0.0, color="black", linewidth=0.5)

        half_group_span = ((n_models - 1) / 2.0) * bar_width + (bar_width / 2.0)
        side_padding = 0.05
        ax.set_xlim(x[0] - half_group_span - side_padding, x[-1] + half_group_span + side_padding)
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.set_ylabel("Variance explained (%)", fontsize=6)
        ax.tick_params(axis="y", labelsize=6, width=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        legend = ax.legend(
            ncol=1,
            frameon=True,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=6,
        )
        if legend is not None:
            legend.get_frame().set_linewidth(0.5)

        plt.tight_layout()
        out_file = summary_dir / f"variance_explained_{view}.pdf"
        fig.savefig(out_file, transparent=True, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


def plot_total_variance_bars_by_modality(total_variance_by_model, summary_dir):
    model_order = [
        "PCA",
        "MEFISTO full GP",
        "MEFISTO 3K",
        "MEFISTO 1K",
        "MOFTy combined",
        "MOFTy CFM",
        "MOFTy non-CFM",
    ]
    model_colors = {
        "PCA": "#9b9b9b",
        "MEFISTO full GP": "#DA0000",
        "MEFISTO 3K": "#FF7C30",
        "MEFISTO 1K": "#FFE100",
        "MOFTy combined": "#0143A6",
        "MOFTy CFM": "#1182E6",
        "MOFTy non-CFM": "#71C0EE",
    }

    view_groups = [
        ("gene_exp", "gene expression"),
        ("protein", "protein expression"),
    ]

    n_models = len(model_order)
    group_centers = np.arange(len(view_groups), dtype=float)
    group_width = 0.8
    bar_width = group_width / n_models

    totals_by_model = {}
    finite_values = []
    for model_name_label in model_order:
        values = []
        model_totals = total_variance_by_model.get(model_name_label, {})
        for view_key, _ in view_groups:
            value = model_totals.get(view_key, np.nan)
            values.append(value)
            if np.isfinite(value):
                finite_values.append(value)
        totals_by_model[model_name_label] = values

    if not finite_values:
        return

    fig, ax = plt.subplots(figsize=(cm(14), cm(6)))

    for model_idx, model_name_label in enumerate(model_order):
        offsets = (model_idx - (n_models - 1) / 2.0) * bar_width
        x_pos = group_centers + offsets
        y_vals = totals_by_model[model_name_label]
        y_plot = np.nan_to_num(np.asarray(y_vals, dtype=float), nan=0.0)

        bars = ax.bar(
            x_pos,
            y_plot,
            width=bar_width,
            color=model_colors.get(model_name_label, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.3,
            label=model_name_label,
        )

        for bar, value in zip(bars, y_vals):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.25,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    ymax = max(1.0, np.nanmax(finite_values))
    ax.set_ylim(0.0, ymax * 1.2)
    ax.set_ylabel("Total variance explained (%)", fontsize=6)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in view_groups], fontsize=6)
    ax.tick_params(axis="y", labelsize=6, width=0.5)
    ax.tick_params(axis="x", width=0.5, length=1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    legend = ax.legend(
        ncol=1,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    out_file = summary_dir / "total_variance_explained_grouped.pdf"
    fig.savefig(out_file, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)



def main():
    args = get_parser().parse_args()
    set_plot_style()

    n_factors = args.n_factors
    if n_factors != 4:
        print(f"Warning: requested n_factors={n_factors}. The requested layout is 4 rows.")

    data_dir = Path("../input/input_gbm")
    mofty_output_dir = Path("../output/output_gbm_cfm_4/runs_combined_factors_4")
    mefisto_output_dir = Path("../output/output_gbm_mefisto")
    summary_dir = Path("../paper/supp_fig_comparison_pca_mefisto_mofty")
    summary_dir.mkdir(parents=True, exist_ok=True)

    model_file_z = mofty_output_dir / "mofty_model_Z.hdf5"
    model_file_z_cfm = mofty_output_dir / "mofty_model_Z_cfm.hdf5"
    model_file_z_non_cfm = mofty_output_dir / "mofty_model_Z_non_cfm.hdf5"
    mefisto_model_file_full = mefisto_output_dir / f"mefisto_model_trained_{args.model}.hdf5"
    mefisto_model_file_3k = mefisto_output_dir / f"mefisto_model_trained_{args.model_3k}.hdf5"
    mefisto_model_file_1k = mefisto_output_dir / f"mefisto_model_trained_{args.model_1k}.hdf5"

    mdata = mu.read_h5mu(data_dir / "processed_mdata.h5mu")
    adata_pca = sc.read_h5ad(data_dir / "processed_adata.h5ad")

    m_z = mofax.mofa_model(model_file_z)
    m_z_cfm = mofax.mofa_model(model_file_z_cfm)
    m_z_non_cfm = mofax.mofa_model(model_file_z_non_cfm)
    m_mefisto_full = mofax.mofa_model(mefisto_model_file_full)
    m_mefisto_3k = mofax.mofa_model(mefisto_model_file_3k)
    m_mefisto_1k = mofax.mofa_model(mefisto_model_file_1k)

    pca_row_order = list(range(1, n_factors + 1))
    factor_order = list(range(1, n_factors + 1))

    adata_plot = mdata.mod["gene_exp"].copy()
    adata_plot.obsm["spatial"] = mdata.obs.loc[adata_plot.obs_names, ["spatial1", "spatial2"]].to_numpy()

    pca_out = compute_joint_pca_scores(adata_pca, n_factors=n_factors)
    pca_scores = pca_out["pca_scores_df"]

    mefisto_scores_full = get_first_n_factor_df(m_mefisto_full, n_factors=n_factors, prefix="MEFISTO_FULL_")
    mefisto_scores_3k = get_first_n_factor_df(m_mefisto_3k, n_factors=n_factors, prefix="MEFISTO_3K_")
    mefisto_scores_1k = get_first_n_factor_df(m_mefisto_1k, n_factors=n_factors, prefix="MEFISTO_1K_")
    mofty_comb_scores = get_first_n_factor_df(m_z, n_factors=n_factors, prefix="MOFTY_COMB_")
    mofty_cfm_scores = get_first_n_factor_df(m_z_cfm, n_factors=n_factors, prefix="MOFTY_CFM_")
    mofty_non_cfm_scores = get_first_n_factor_df(m_z_non_cfm, n_factors=n_factors, prefix="MOFTY_NON_CFM_")

    adata_plot = join_obs_columns(adata_plot, pca_scores)
    adata_plot = join_obs_columns(adata_plot, mefisto_scores_full)
    adata_plot = join_obs_columns(adata_plot, mefisto_scores_3k)
    adata_plot = join_obs_columns(adata_plot, mefisto_scores_1k)
    adata_plot = join_obs_columns(adata_plot, mofty_comb_scores)
    adata_plot = join_obs_columns(adata_plot, mofty_cfm_scores)
    adata_plot = join_obs_columns(adata_plot, mofty_non_cfm_scores)

    # Build best-match row maps from natural (unreordered) score columns using PCA as anchor.
    pca_score_cols = [f"PCA_PC{i}" for i in factor_order]
    mefisto_full_score_cols = [f"MEFISTO_FULL_F{i}" for i in factor_order]
    mefisto_3k_score_cols = [f"MEFISTO_3K_F{i}" for i in factor_order]
    mefisto_1k_score_cols = [f"MEFISTO_1K_F{i}" for i in factor_order]
    mofty_comb_score_cols = [f"MOFTY_COMB_F{i}" for i in factor_order]

    score_matrices_best_match = {
        "PCA": build_standard_factor_df_from_cols(adata_plot.obs, pca_score_cols),
        "MEFISTO full GP": build_standard_factor_df_from_cols(adata_plot.obs, mefisto_full_score_cols),
        "MEFISTO 3K": build_standard_factor_df_from_cols(adata_plot.obs, mefisto_3k_score_cols),
        "MEFISTO 1K": build_standard_factor_df_from_cols(adata_plot.obs, mefisto_1k_score_cols),
        "MOFTy combined": build_standard_factor_df_from_cols(adata_plot.obs, mofty_comb_score_cols),
    }
    best_match_row_labels = [f"PC{pc_idx}" for pc_idx in pca_row_order]
    best_match_row_maps = build_best_match_row_maps(
        reference_label="PCA",
        score_matrices=score_matrices_best_match,
    )

    def sanitize_factor_order(order_values):
        out = []
        for row_pos, source_factor_idx in enumerate(order_values, start=1):
            if source_factor_idx is None:
                out.append(row_pos)
            else:
                out.append(int(source_factor_idx))
        return out

    def build_mapped_obs_cols(prefix, row_map, model_label):
        cols = []
        factor_labels = []
        for row_pos, source_factor_idx in enumerate(row_map, start=1):
            if source_factor_idx is None:
                raise ValueError(
                    f"Missing best-match source factor for {model_label} at PCA row {row_pos}."
                )
            cols.append(f"{prefix}{source_factor_idx}")
            factor_labels.append(int(source_factor_idx))
        return cols, factor_labels

    adata_plot_mi = prepare_rowstd_spatial_neighbors(adata_plot.copy())

    pca_row_cols = [f"PCA_PC{i}" for i in pca_row_order]
    mefisto_full_row_cols, mefisto_full_factor_labels = build_mapped_obs_cols(
        "MEFISTO_FULL_F",
        best_match_row_maps["MEFISTO full GP"],
        "MEFISTO full GP",
    )
    mefisto_3k_row_cols, mefisto_3k_factor_labels = build_mapped_obs_cols(
        "MEFISTO_3K_F",
        best_match_row_maps["MEFISTO 3K"],
        "MEFISTO 3K",
    )
    mefisto_1k_row_cols, mefisto_1k_factor_labels = build_mapped_obs_cols(
        "MEFISTO_1K_F",
        best_match_row_maps["MEFISTO 1K"],
        "MEFISTO 1K",
    )
    mofty_comb_row_cols, mofty_comb_factor_labels = build_mapped_obs_cols(
        "MOFTY_COMB_F",
        best_match_row_maps["MOFTy combined"],
        "MOFTy combined",
    )
    # Keep CFM and non-CFM rows aligned to PCA via MOFTy-combined row assignment.
    mofty_cfm_row_cols = [f"MOFTY_CFM_F{i}" for i in mofty_comb_factor_labels]
    mofty_non_cfm_row_cols = [f"MOFTY_NON_CFM_F{i}" for i in mofty_comb_factor_labels]

    variance_model_factor_orders = {
        "MEFISTO full GP": sanitize_factor_order(best_match_row_maps["MEFISTO full GP"]),
        "MEFISTO 3K": sanitize_factor_order(best_match_row_maps["MEFISTO 3K"]),
        "MEFISTO 1K": sanitize_factor_order(best_match_row_maps["MEFISTO 1K"]),
        "MOFTy combined": sanitize_factor_order(best_match_row_maps["MOFTy combined"]),
        "MOFTy CFM": sanitize_factor_order(mofty_comb_factor_labels),
        "MOFTy non-CFM": sanitize_factor_order(mofty_comb_factor_labels),
    }

    mefisto_scales_full_all = get_mefisto_scales(m_mefisto_full, n_factors=n_factors)
    mefisto_scales_3k_all = get_mefisto_scales(m_mefisto_3k, n_factors=n_factors)
    mefisto_scales_1k_all = get_mefisto_scales(m_mefisto_1k, n_factors=n_factors)

    mefisto_scales_full_row = np.asarray([mefisto_scales_full_all[i - 1] for i in mefisto_full_factor_labels], dtype=float)
    mefisto_scales_3k_row = np.asarray([mefisto_scales_3k_all[i - 1] for i in mefisto_3k_factor_labels], dtype=float)
    mefisto_scales_1k_row = np.asarray([mefisto_scales_1k_all[i - 1] for i in mefisto_1k_factor_labels], dtype=float)

    column_specs = [
        {
            "header": "PCA",
            "kind": "pca",
            "group_size": 1,
            "cols": pca_row_cols,
        },
        {
            "header": "MEFISTO full GP",
            "kind": "mefisto",
            "group_size": 3,
            "scales": mefisto_scales_full_row,
            "factor_labels": mefisto_full_factor_labels,
            "cols": mefisto_full_row_cols,
        },
        {
            "header": "MEFISTO 3000",
            "kind": "mefisto",
            "group_size": 0,
            "scales": mefisto_scales_3k_row,
            "factor_labels": mefisto_3k_factor_labels,
            "cols": mefisto_3k_row_cols,
        },
        {
            "header": "MEFISTO 1000",
            "kind": "mefisto",
            "group_size": 0,
            "scales": mefisto_scales_1k_row,
            "factor_labels": mefisto_1k_factor_labels,
            "cols": mefisto_1k_row_cols,
        },
        {
            "header": "MOFTy combined",
            "kind": "mofty",
            "group_size": 3,
            "factor_labels": mofty_comb_factor_labels,
            "cols": mofty_comb_row_cols,
        },
        {
            "header": "MOFTy CFM",
            "kind": "mofty",
            "group_size": 0,
            "factor_labels": mofty_comb_factor_labels,
            "cols": mofty_cfm_row_cols,
        },
        {
            "header": "MOFTy non-CFM",
            "kind": "mofty",
            "group_size": 0,
            "factor_labels": mofty_comb_factor_labels,
            "cols": mofty_non_cfm_row_cols,
        },
    ]

    grid_out = summary_dir / f"factors_grid.pdf"
    plot_spatial_grid(
        adata_plot=adata_plot,
        adata_plot_mi=adata_plot_mi,
        column_specs=column_specs,
        pca_row_order=pca_row_order,
        out_file=grid_out,
    )
    print(f"Saved spatial comparison grid to: {grid_out}")

    corr_df = pd.DataFrame(index=adata_plot.obs_names)
    for i in range(n_factors):
        corr_df[f"PC{i + 1}"] = adata_plot.obs[f"PCA_PC{i + 1}"]
    for i in range(n_factors):
        corr_df[f"MEFISTO full GP F{i + 1}"] = adata_plot.obs[f"MEFISTO_FULL_F{i + 1}"]
    for i in range(n_factors):
        corr_df[f"MEFISTO 3K F{i + 1}"] = adata_plot.obs[f"MEFISTO_3K_F{i + 1}"]
    for factor_idx in factor_order:
        corr_df[f"MEFISTO 1K F{factor_idx}"] = adata_plot.obs[f"MEFISTO_1K_F{factor_idx}"]
    for i in range(n_factors):
        corr_df[f"MOFTy combined F{i + 1}"] = adata_plot.obs[f"MOFTY_COMB_F{i + 1}"]
    for i in range(n_factors):
        corr_df[f"MOFTy CFM F{i + 1}"] = adata_plot.obs[f"MOFTY_CFM_F{i + 1}"]
    for i in range(n_factors):
        corr_df[f"MOFTy non-CFM F{i + 1}"] = adata_plot.obs[f"MOFTY_NON_CFM_F{i + 1}"]

    corr_matrix_out = summary_dir / f"factors_correlation.csv"
    corr_df.corr(method="pearson").to_csv(corr_matrix_out)

    corr_plot_out = summary_dir / f"factors_correlation.pdf"
    corr_cbar_out = summary_dir / f"correlation_colorbar.pdf"
    plot_correlation_heatmap(corr_df, corr_plot_out, block_sizes=[n_factors] * 7)
    plot_correlation_colorbar(corr_cbar_out)
    print(f"Saved correlation matrix CSV to: {corr_matrix_out}")
    print(f"Saved correlation heatmap to: {corr_plot_out}")
    print(f"Saved correlation colorbar to: {corr_cbar_out}")

    # Best-match analysis across PCA + MEFISTO variants + MOFTy combined only.
    # Uses row maps derived above from natural score matrices anchored to PCA.

    score_best_match_corr, score_best_match_p = compute_best_match_signed_corr_matrix(
        value_matrices=score_matrices_best_match,
        row_factor_maps=best_match_row_maps,
        row_labels=best_match_row_labels,
        align_on_index=False,
        anchor_label="PCA",
    )
    score_best_match_csv = (
        summary_dir
        / f"factors_best_match_correlation.csv"
    )
    score_best_match_corr.to_csv(score_best_match_csv, index=True)
    print(f"Saved best-match factor/PC signed correlation matrix to: {score_best_match_csv}")

    pca_weight_by_view = build_pca_weight_dfs(
        adata_pca=adata_pca,
        pca_model=pca_out["pca_model"],
        feature_mask=pca_out["feature_mask"],
        protein_mask=pca_out["protein_mask"],
        pca_row_order=pca_row_order,
        n_factors=n_factors,
    )

    weight_p_by_view = {}
    weight_corr_by_view = {}
    for view_label, view_key in [("gene expression", "gene_exp"), ("protein expression", "protein")]:
        weight_matrices_best_match = {
            "PCA": pca_weight_by_view[view_label],
            "MEFISTO full GP": reorder_factor_columns(
                get_first_n_weight_df(m_mefisto_full, n_factors=n_factors, view_name=view_key),
                factor_order,
            ),
            "MEFISTO 3K": reorder_factor_columns(
                get_first_n_weight_df(m_mefisto_3k, n_factors=n_factors, view_name=view_key),
                factor_order,
            ),
            "MEFISTO 1K": reorder_factor_columns(
                get_first_n_weight_df(m_mefisto_1k, n_factors=n_factors, view_name=view_key),
                factor_order,
            ),
            "MOFTy combined": reorder_factor_columns(
                get_first_n_weight_df(m_z, n_factors=n_factors, view_name=view_key),
                factor_order,
            ),
        }

        weight_best_match_corr, weight_best_match_p = compute_best_match_signed_corr_matrix(
            value_matrices=weight_matrices_best_match,
            row_factor_maps=best_match_row_maps,
            row_labels=best_match_row_labels,
            align_on_index=True,
            anchor_label="PCA",
        )
        weight_corr_by_view[view_label] = weight_best_match_corr.copy()
        weight_p_by_view[view_label] = weight_best_match_p.copy()

        weight_best_match_csv = (
            summary_dir
            / f"weights_best_match_correlation_{view_label}.csv"
        )
        weight_best_match_corr.to_csv(weight_best_match_csv, index=True)
        print(f"Saved best-match weight signed correlation matrix ({view_label}) to: {weight_best_match_csv}")

    combined_best_match_plot = summary_dir / "best_match_correlation_heatmaps_combined.pdf"
    plot_best_match_signed_correlation_panels(
        corr_matrices=[
            score_best_match_corr,
            weight_corr_by_view["gene expression"],
            weight_corr_by_view["protein expression"],
        ],
        panel_titles=[
            "Factors/PC scores",
            "Weights gene expression",
            "Weights protein expression",
        ],
        out_file=combined_best_match_plot,
    )
    print(f"Saved combined best-match heatmaps to: {combined_best_match_plot}")

    p_values_txt_out = (
        summary_dir
        / f"best_match_significance_values.txt"
    )
    write_best_match_pvalues_txt(
        out_file=p_values_txt_out,
        score_p_matrix=score_best_match_p,
        weight_p_by_view=weight_p_by_view,
    )
    print(f"Saved best-match p-values to: {p_values_txt_out}")

    pca_var = compute_pca_modality_variance(
        adata_pca=adata_pca,
        X_concat=pca_out["X_concat"],
        X_pca=pca_out["X_pca"],
        pca_model=pca_out["pca_model"],
        feature_mask=pca_out["feature_mask"],
        protein_mask=pca_out["protein_mask"],
        n_factors=n_factors,
    )

    pca_var_rows = []
    for view_name, values in pca_var.items():
        for factor_idx, pc_idx in enumerate(pca_row_order, start=1):
            value = values[pc_idx - 1]
            pca_var_rows.append(
                {
                    "model": "PCA",
                    "view": view_name,
                    "factor_idx": factor_idx,
                    "component": f"PC{pc_idx}",
                    "variance_ratio": value,
                }
            )
    pca_var_df = pd.DataFrame(pca_var_rows)

    mefisto_var_df = extract_variance_by_view(m_mefisto_full, "MEFISTO full GP", n_factors=n_factors)
    mefisto_3k_var_df = extract_variance_by_view(m_mefisto_3k, "MEFISTO 3K", n_factors=n_factors)
    mefisto_1k_var_df = extract_variance_by_view(m_mefisto_1k, "MEFISTO 1K", n_factors=n_factors)
    mofty_comb_var_df = extract_variance_by_view(m_z, "MOFTy combined", n_factors=n_factors)
    mofty_cfm_var_df = extract_variance_by_view(m_z_cfm, "MOFTy CFM", n_factors=n_factors)
    mofty_non_cfm_var_df = extract_variance_by_view(m_z_non_cfm, "MOFTy non-CFM", n_factors=n_factors)

    variance_df = pd.concat(
        [
            pca_var_df,
            mefisto_var_df,
            mefisto_3k_var_df,
            mefisto_1k_var_df,
            mofty_comb_var_df,
            mofty_cfm_var_df,
            mofty_non_cfm_var_df,
        ],
        axis=0,
        ignore_index=True,
    )

    variance_df["variance_ratio"] = pd.to_numeric(variance_df["variance_ratio"], errors="coerce")
    variance_df = variance_df.dropna(subset=["variance_ratio"])

    variance_out = summary_dir / f"variance_explained_by_modality_and_factor.csv"
    variance_df.to_csv(variance_out, index=False)
    print(f"Saved variance explained summary to: {variance_out}")

    plot_variance_bars_by_modality(
        variance_df,
        summary_dir=summary_dir,
        model_name=args.model,
        n_factors=n_factors,
        pca_row_order=pca_row_order,
        model_factor_orders=variance_model_factor_orders,
    )
    print("Saved modality-specific variance bar plots.")

    total_variance_by_model = {
        "PCA": {
            "gene_exp": float(np.nansum(pca_var.get("gene_exp", [])) * 100.0),
            "protein": float(np.nansum(pca_var.get("protein", [])) * 100.0),
        },
        "MEFISTO full GP": extract_total_variance_by_view(m_mefisto_full),
        "MEFISTO 3K": extract_total_variance_by_view(m_mefisto_3k),
        "MEFISTO 1K": extract_total_variance_by_view(m_mefisto_1k),
        "MOFTy combined": extract_total_variance_by_view(m_z),
        "MOFTy CFM": extract_total_variance_by_view(m_z_cfm),
        "MOFTy non-CFM": extract_total_variance_by_view(m_z_non_cfm),
    }

    plot_total_variance_bars_by_modality(
        total_variance_by_model=total_variance_by_model,
        summary_dir=summary_dir,
    )
    print("Saved modality-specific total variance bar plots.")


if __name__ == "__main__":
    main()
