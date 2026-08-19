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

import h5py
import matplotlib
import matplotlib.ticker as mticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_SLICES = 10
FIG_WIDTH_CM = 18.0
ROW_HEIGHT_CM = 1.5
FONT_SIZE_PT = 3.5
TIGHT_LAYOUT_PAD = 0.2
TIGHT_LAYOUT_W_PAD = 0.2
TIGHT_LAYOUT_H_PAD = 0.2


def cm(value):
	return value / 2.54


def load_posterior_samples(hdf5_path):
	with h5py.File(hdf5_path, "r") as handle:
		sample_group = handle["samples"]
		sample_keys = sorted(sample_group.keys(), key=lambda key: int(key))
		samples = np.stack([sample_group[key][()] for key in sample_keys], axis=0)

		if "stats" in handle and "mean" in handle["stats"]:
			posterior_mean = handle["stats"]["mean"][()]
		else:
			posterior_mean = samples.mean(axis=0)

	return samples, posterior_mean


def select_slice_at_y(covariates_df, y_value):
	x_column = covariates_df.columns[0]
	y_column = covariates_df.columns[1]
	slice_mask = np.isclose(covariates_df[y_column].to_numpy(dtype=float), float(y_value))
	slice_df = covariates_df.loc[slice_mask, [x_column, y_column]].sort_values(x_column, kind="mergesort")
	return slice_df.index.to_numpy(), slice_df[x_column].to_numpy(), x_column, y_column


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pix_shape",
		type=int,
		default=30,
		help="Pixel-grid side length used to pick slice rows (e.g. 30 or 100).",
	)
	return parser.parse_args()


def build_paths(pix_shape):
	run_dir = Path(f"../output/output_syn_rec_cfm_dim2_{pix_shape}/runs_combined_factors_5")
	latent_paths = {
		"Z": run_dir / "Z/latest.hdf5",
		"Z_cfm": run_dir / "Z_cfm/latest.hdf5",
		"Z_non_cfm": run_dir / "Z_non_cfm/latest.hdf5",
	}
	covariates_path = run_dir / "covariates.parquet"
	output_dir = Path(f"../paper/syn_2D_{pix_shape}")
	return latent_paths, covariates_path, output_dir


def plot_slice_grid(covariates_df, latent_name, latent_path, pix_shape, output_dir):
	samples, posterior_mean = load_posterior_samples(latent_path)

	if samples.ndim != 3:
		raise ValueError(f"Expected samples with shape (n_samples, n_rows, n_points), got {samples.shape}.")

	if covariates_df.shape[1] < 2:
		raise ValueError("Expected at least two covariate columns for the 2D slice plot.")
	if pix_shape < N_SLICES:
		raise ValueError(f"pix_shape must be >= {N_SLICES}, got {pix_shape}.")
	if pix_shape % N_SLICES != 0:
		raise ValueError(f"pix_shape must be divisible by {N_SLICES} for exact slice steps, got {pix_shape}.")

	y_column = covariates_df.columns[1]
	y_values = np.unique(covariates_df[y_column].to_numpy(dtype=float))
	step = pix_shape // N_SLICES
	slice_positions = [ii * step for ii in range(N_SLICES)]
	if max(slice_positions) >= len(y_values):
		raise ValueError(
			f"pix_shape={pix_shape} implies max slice index {max(slice_positions)} but only {len(y_values)} y-levels are available."
		)
	if len(slice_positions) < N_SLICES:
		raise ValueError(f"Need at least {N_SLICES} distinct slice positions to build the panel.")

	n_factors = samples.shape[1]
	fig, axes = plt.subplots(
		n_factors,
		N_SLICES,
		figsize=(cm(FIG_WIDTH_CM), cm(ROW_HEIGHT_CM * n_factors)),
		sharex=True,
		sharey=True,
	)
	if n_factors == 1:
		axes = np.asarray([axes])

	for ax in axes.flat:
		ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
		ax.set_xlim(1.0, float(pix_shape))
		ax.set_xticks([1.0, pix_shape / 2.0, float(pix_shape)])
		ax.tick_params(labelsize=FONT_SIZE_PT, width=0.25, length=1.0, pad=0.5)
		for spine in ax.spines.values():
			spine.set_linewidth(0.25)

	for factor_idx in range(n_factors):
		for slice_idx, slice_position in enumerate(slice_positions):
			ax = axes[factor_idx, slice_idx]
			y_value = y_values[slice_position]
			slice_indices, x_values, x_label, _ = select_slice_at_y(covariates_df, y_value)

			if slice_indices.size == 0:
				raise ValueError(f"No covariate rows matched slice position {slice_position}.")

			slice_samples = samples[:, factor_idx, :][:, slice_indices]
			slice_mean = posterior_mean[factor_idx, slice_indices]
			x_values_plot = x_values + 1.0

			for sample_values in slice_samples:
				ax.plot(x_values_plot, sample_values, color="lightsteelblue", linewidth=0.15)

			ax.plot(x_values_plot, slice_mean, color="blue", linewidth=0.3)
			if factor_idx == 0:
				ax.set_title(f"{y_column} = {y_value:.3g}", fontsize=FONT_SIZE_PT, pad=1.0)
			if slice_idx == 0:
				ax.set_ylabel(f"Factor {factor_idx + 1}", fontsize=FONT_SIZE_PT, labelpad=1.0)
			if factor_idx == n_factors - 1:
				ax.set_xlabel(x_label, fontsize=FONT_SIZE_PT, labelpad=1.0)
			ax.grid(False)

	out_file = output_dir / f"slice_grid_{pix_shape}_{latent_name}.pdf"
	fig.tight_layout(pad=TIGHT_LAYOUT_PAD, w_pad=TIGHT_LAYOUT_W_PAD, h_pad=TIGHT_LAYOUT_H_PAD)
	fig.savefig(out_file, bbox_inches="tight", transparent=True)
	plt.close(fig)

	print(f"Saved {out_file}")


def main():
	args = parse_args()
	latent_paths, covariates_path, output_dir = build_paths(args.pix_shape)
	for latent_name, latent_path in latent_paths.items():
		if not latent_path.exists():
			raise FileNotFoundError(f"Missing latent file for {latent_name}: {latent_path}")
	if not covariates_path.exists():
		raise FileNotFoundError(f"Missing covariates file: {covariates_path}")

	output_dir.mkdir(parents=True, exist_ok=True)

	covariates_df = pd.read_parquet(covariates_path)

	for latent_name, latent_path in latent_paths.items():
		plot_slice_grid(covariates_df, latent_name, latent_path, args.pix_shape, output_dir)


if __name__ == "__main__":
	main()
