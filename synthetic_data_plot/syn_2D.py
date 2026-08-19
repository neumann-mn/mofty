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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ident", type=str, default="syn_cfm_dim2")
    parser.add_argument("--output_ident", type=str, default="syn_rec_cfm_dim2")
    parser.add_argument("--pix_shape", type=int, required=True)
    parser.add_argument("--n_factors", type=int, default=5)
    return parser


def main():
    args = get_parser().parse_args()
    pix_shape = args.pix_shape
    n_factors = args.n_factors
    input_ident = args.input_ident
    output_ident = args.output_ident
    output_dir = Path(f"../paper/syn_2D_{pix_shape}")
    output_dir.mkdir(exist_ok=True, parents=True)

    Z_ground_truth = f"../input/input_{input_ident}_{pix_shape}/Z.parquet"
    Z_cfm_ground_truth = f"../input/input_{input_ident}_{pix_shape}/Z_cfm.parquet"
    Z_non_cfm_ground_truth = f"../input/input_{input_ident}_{pix_shape}/Z_non_cfm.parquet"

    Z_rec = f"../output/output_{output_ident}_{pix_shape}/runs_combined_factors_5/Z.parquet"
    Z_cfm_rec = f"../output/output_{output_ident}_{pix_shape}/runs_combined_factors_5/Z_cfm.parquet"
    Z_non_cfm_rec = f"../output/output_{output_ident}_{pix_shape}/runs_combined_factors_5/Z_non_cfm.parquet"

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
    fig, axes = plt.subplots(n_factors, 6, figsize=(15/2.54, 12.5/2.54))
    plt.subplots_adjust(wspace=-0.4, hspace=0.25)

    # --- Plotting Loop ---
    for factor_idx in range(n_factors):
        # Data for the current factor
        data_gt = {
            "combined": Z_ground_truth_df[factor_idx].values.reshape(pix_shape, pix_shape).T,
            "cfm": Z_cfm_ground_truth_df[factor_idx].values.reshape(pix_shape, pix_shape).T,
            "non_cfm": Z_non_cfm_ground_truth_df[factor_idx].values.reshape(pix_shape, pix_shape).T
        }
        data_rec = {
            "combined": Z_rec_df[factor_idx].values.reshape(pix_shape, pix_shape).T,
            "cfm": Z_cfm_rec_df[factor_idx].values.reshape(pix_shape, pix_shape).T,
            "non_cfm": Z_non_cfm_rec_df[factor_idx].values.reshape(pix_shape, pix_shape).T
        }

        # Determine color scale for the entire row
        all_row_data = np.concatenate([
            data_gt["combined"].flatten(), data_rec["combined"].flatten(),
            data_gt["cfm"].flatten(), data_rec["cfm"].flatten(),
            data_gt["non_cfm"].flatten(), data_rec["non_cfm"].flatten()
        ])
        vmax = np.max(np.abs(all_row_data))
        vmin = -vmax

        # Titles for columns
        titles = [
            "Combined\n(ground truth)", "Combined\n(reconstruction)",
            "CFM\n(ground truth)", "CFM\n(reconstruction)",
            "Non-CFM\n(ground truth)", "Non-CFM\n(reconstruction)"
        ]

        plot_data_map = [
            data_gt["combined"], data_rec["combined"],
            data_gt["cfm"], data_rec["cfm"],
            data_gt["non_cfm"], data_rec["non_cfm"]
        ]

        for col_idx in range(6):
            ax = axes[factor_idx, col_idx]

            im = ax.imshow(
                plot_data_map[col_idx],
                cmap='seismic',
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='none',
                rasterized=True,
                norm=None,
                origin='lower'
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
            ax.set_xticks([])
            ax.set_yticks([])

            if factor_idx == 0:
                # Split title into two lines
                title_parts = titles[col_idx].split('\n')
                # Add first line with a larger font size
                ax.text(0.5, 1.22, title_parts[0], ha='center', va='center',
                        transform=ax.transAxes, fontsize=6)
                # Add second line with a smaller font size
                if len(title_parts) > 1:
                    ax.text(0.5, 1.1, title_parts[1], ha='center', va='center',
                            transform=ax.transAxes, fontsize=5)

            if col_idx % 2 == 1:
                # Calculate Pearson correlation
                gt_data = plot_data_map[col_idx - 1]
                rec_data = plot_data_map[col_idx]
                corr, p_value = pearsonr(gt_data.flatten(), rec_data.flatten())
                for p in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                    if p_value < p:
                        p_value = p
                        break
                    if p == 1e-2 and p_value >= p:
                        raise ValueError(f"Check p-value threshold: p-value is {p_value:.2e}, which is above the threshold of {p}.")

                ax.set_xlabel(f"r={corr:.2f}, p<{p_value:.0e}", fontsize=5, labelpad=3)

        # Set y-label for the factor on the first plot of the row

        # Add a single colorbar for the row
        # Set y-label for the factor on the first plot of the row
        ax_last = axes[factor_idx, 5]
        ax_last.set_ylabel(f"F{factor_idx + 1}", fontsize=6, labelpad=5, rotation=0)
        ax_last.yaxis.set_label_position("right")

    plt.savefig(output_dir / f"reconstruction_plot_dim2.pdf", dpi=2400, bbox_inches='tight', transparent=True)

if __name__ == "__main__":
    main()
