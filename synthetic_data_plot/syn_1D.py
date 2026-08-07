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

from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    output_dir = Path("../paper/syn_1D")
    output_dir.mkdir(exist_ok=True, parents=True)

    Z_ground_truth = "../input/input_syn_cfm_dim1/Z.parquet"
    Z_cfm_ground_truth = "../input/input_syn_cfm_dim1/Z_cfm.parquet"
    Z_non_cfm_ground_truth = "../input/input_syn_cfm_dim1/Z_non_cfm.parquet"

    Z_rec = "../output/output_syn_rec_cfm_dim1/runs_combined_factors_5/Z.parquet"
    Z_cfm_rec = "../output/output_syn_rec_cfm_dim1/runs_combined_factors_5/Z_cfm.parquet"
    Z_non_cfm_rec = "../output/output_syn_rec_cfm_dim1/runs_combined_factors_5/Z_non_cfm.parquet"


    # --- Data Loading ---
    # Ground Truth
    Z_ground_truth_df = pd.read_parquet(Z_ground_truth).T
    Z_cfm_ground_truth_df = pd.read_parquet(Z_cfm_ground_truth).T
    Z_non_cfm_ground_truth_df = pd.read_parquet(Z_non_cfm_ground_truth).T

    # Reconstructed
    Z_rec_df = pd.read_parquet(Z_rec).T
    Z_cfm_rec_df = pd.read_parquet(Z_cfm_rec).T
    Z_non_cfm_rec_df = pd.read_parquet(Z_non_cfm_rec).T

    # Convert index to numeric for all dataframes
    def extract_numeric_from_index(df):
        if df.index.dtype == 'object':
            df.index = df.index.str.replace(r'sample_', '', regex=True)
        df.index = pd.to_numeric(df.index)
        return df

    Z_ground_truth_df = extract_numeric_from_index(Z_ground_truth_df)
    Z_cfm_ground_truth_df = extract_numeric_from_index(Z_cfm_ground_truth_df)
    Z_non_cfm_ground_truth_df = extract_numeric_from_index(Z_non_cfm_ground_truth_df)
    Z_rec_df = extract_numeric_from_index(Z_rec_df)
    Z_cfm_rec_df = extract_numeric_from_index(Z_cfm_rec_df)
    Z_non_cfm_rec_df = extract_numeric_from_index(Z_non_cfm_rec_df)

    # --- Plotting Setup ---
    fig, axes = plt.subplots(5, 2, figsize=(7/2.54, 10/2.54), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    colors = {"combined": "#006EFF", "cfm": "#FF0000", "non_cfm": "#5D5C5C"}
    z_orders = {"combined": 1, "non_cfm": 0, "cfm": 2}

    # --- Find global limits for x and y axes ---
    all_data = [
        Z_ground_truth_df, Z_cfm_ground_truth_df, Z_non_cfm_ground_truth_df,
        Z_rec_df, Z_cfm_rec_df, Z_non_cfm_rec_df
    ]
    # Convert all data to numeric, coercing errors to NaN
    all_data_numeric = [df.apply(pd.to_numeric, errors='coerce') for df in all_data]

    min_val = min(df.min().min() for df in all_data_numeric)
    max_val = max(df.max().max() for df in all_data_numeric)
    bufer = (max_val - min_val) * 0.05

    # --- Plotting Loop ---
    for factor_idx in range(5):
        # --- Ground Truth Plot ---
        ax_gt = axes[factor_idx, 0]

        # Plot combined, stochastic, CFM
        ax_gt.scatter(Z_ground_truth_df.index, Z_ground_truth_df[factor_idx],
                    color=colors["combined"], s=0.3, linewidths=0, zorder=z_orders["combined"])
        ax_gt.scatter(Z_non_cfm_ground_truth_df.index, Z_non_cfm_ground_truth_df[factor_idx],
                    color=colors["non_cfm"], s=0.3, linewidths=0, zorder=z_orders["non_cfm"])
        ax_gt.scatter(Z_cfm_ground_truth_df.index, Z_cfm_ground_truth_df[factor_idx],
                    color=colors["cfm"], s=0.3, linewidths=0, zorder=z_orders["cfm"])
        if factor_idx == 0:
            ax_gt.set_title("Ground truth", fontsize=6, pad=3)

        # --- Reconstructed Plot ---

        # --- Correlation Calculation & Legend ---
        data_pairs = {
            "combined": (Z_ground_truth_df, Z_rec_df),
            "cfm": (Z_cfm_ground_truth_df, Z_cfm_rec_df),
            "non_cfm": (Z_non_cfm_ground_truth_df, Z_non_cfm_rec_df)
        }

        correlations = {}
        p_values = []
        for name, (gt_df, rec_df) in data_pairs.items():
            aligned_gt, aligned_rec = gt_df[[factor_idx]].align(rec_df[[factor_idx]], join='inner', axis=0)
            corr, p_value = pearsonr(aligned_gt.iloc[:, 0], aligned_rec.iloc[:, 0])
            correlations[name] = corr
            p_values.append(p_value)

        max_p_value = max(p_values)
        for p in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            if max_p_value < p:
                max_p_value = p
                break
            if p == 1e-2 and max_p_value >= p:
                raise ValueError(f"Check p-value threshold: p-value is {max_p_value:.2e}, which is above the threshold of {p}.")

        ax_rec = axes[factor_idx, 1]

        ax_rec.scatter(Z_rec_df.index, Z_rec_df[factor_idx],
                    color=colors["combined"], s=0.3, linewidths=0, zorder=z_orders["combined"])
        ax_rec.scatter(Z_non_cfm_rec_df.index, Z_non_cfm_rec_df[factor_idx],
                    color=colors["non_cfm"], s=0.3, linewidths=0, zorder=z_orders["non_cfm"])
        ax_rec.scatter(Z_cfm_rec_df.index, Z_cfm_rec_df[factor_idx],
                    color=colors["cfm"], s=0.3, linewidths=0, zorder=z_orders["cfm"])
        if factor_idx == 0:
            ax_rec.set_title(f"Reconstruction", fontsize=6, pad=3)
        ax_rec.set_ylabel(f"F{factor_idx + 1}", fontsize=6, rotation=0, labelpad=5)
        ax_rec.yaxis.set_label_position("right")


        # Add text for each correlation horizontally
        x_pos = 0.2
        for name in ["combined", "cfm", "non_cfm"]:
            corr = correlations[name]
            ax_rec.text(x_pos, 0.05, f"r={corr:.2f}", transform=ax_rec.transAxes,
                        fontsize=5, verticalalignment='bottom', horizontalalignment='center',
                        color=colors[name])
            x_pos += 0.3

        ax_rec.text(0.98, 0.95, f"p<{max_p_value:.0e}", transform=ax_rec.transAxes,
            fontsize=5, verticalalignment='top', horizontalalignment='right',
            color='black')


    # --- Set global limits and outer labels ---
    plt.setp(axes, ylim=(-4, 4), yticks=[ -3.0, -1.5, 0.0, 1.5, 3], xticks=[])
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=5, width=0.25, length=1, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    # for ax in axes[-1,:]:
    #     # ax.set_xlabel("Sample Index", fontsize=5)

    # Set x-ticks to be in steps of 100
    max_index = Z_ground_truth_df.index.max()
    # axes[-1, 0].set_xticks(np.arange(0, max_index, 100))
    # axes[-1, 1].set_xticks(np.arange(0, max_index, 100))


    # --- Legends ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Combined', markerfacecolor=colors["combined"], markersize=5),
        Line2D([0], [0], marker='o', color='w', label='CFM', markerfacecolor=colors["cfm"], markersize=5),
        Line2D([0], [0], marker='o', color='w', label='Non-CFM', markerfacecolor=colors["non_cfm"], markersize=5)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.055), fontsize=6, frameon=False, handletextpad=-0.1, columnspacing=0.75)
    plt.savefig(output_dir / "reconstruction_plot_dim1.pdf", bbox_inches='tight', transparent=True)

if __name__ == "__main__":
    main()
