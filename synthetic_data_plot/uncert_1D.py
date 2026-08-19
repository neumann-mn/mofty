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

import h5py
import matplotlib
import matplotlib.ticker as mticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


N_FACTORS = 5
RUN_DIR = Path("../output/output_syn_rec_cfm_dim1/runs_combined_factors_5")
MODEL_PATHS = {
	"Combined": RUN_DIR / "Z/latest.hdf5",
	"CFM": RUN_DIR / "Z_cfm/latest.hdf5",
	"Non-CFM": RUN_DIR / "Z_non_cfm/latest.hdf5",
}
OUTPUT_DIR = Path("../paper/syn_1D")
FONT_SIZE_PT = 7
TIGHT_LAYOUT_PAD = 0.5
TIGHT_LAYOUT_W_PAD = 1
TIGHT_LAYOUT_H_PAD = 0.5


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


def plot_dim1_uncertainty_grid(model_samples):
	model_names = list(model_samples.keys())
	n_models = len(model_names)

	fig, axes = plt.subplots(
		N_FACTORS,
		n_models,
		figsize=(18/2.54, (3 * N_FACTORS)/2.54),
		sharex=True,
		sharey=True,
		squeeze=False,
	)

	for ax in np.asarray(axes).flat:
		ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
		ax.tick_params(labelsize=FONT_SIZE_PT, pad=2, length=2)

	for model_idx, model_name in enumerate(model_names):
		samples, posterior_mean = model_samples[model_name]
		n_points = samples.shape[2]
		x = np.arange(1, n_points + 1, dtype=float)
		cov_max = float(n_points)

		for factor_idx in range(N_FACTORS):
			ax = axes[factor_idx, model_idx]
			factor_samples = samples[:, factor_idx, :]
			factor_mean = posterior_mean[factor_idx, :]

			for sample_values in factor_samples:
				ax.plot(x, sample_values, color="lightsteelblue", linewidth=0.25)

			ax.plot(x, factor_mean, color="blue", linewidth=0.75)
			if factor_idx == 0:
				ax.set_title(model_name, fontsize=FONT_SIZE_PT, pad=0.5)
			if model_idx == 0:
				ax.set_ylabel(f"Factor {factor_idx + 1}", fontsize=FONT_SIZE_PT, labelpad=1.0)
			ax.set_xlim(1.0, cov_max)
			ax.set_xticks([1.0, 300.0, 600.0, 900.0])
			ax.grid(False)

	return fig


def main():
	for latent_name, latent_path in MODEL_PATHS.items():
		if not latent_path.exists():
			raise FileNotFoundError(f"Missing latent file for {latent_name}: {latent_path}")

	model_samples = {}
	for latent_name, latent_path in MODEL_PATHS.items():
		samples, posterior_mean = load_posterior_samples(latent_path)
		if samples.ndim != 3:
			raise ValueError(f"Expected samples with shape (n_samples, n_factors, n_points), got {samples.shape} for {latent_name}.")
		if samples.shape[1] < N_FACTORS:
			raise ValueError(f"Model {latent_name} has only {samples.shape[1]} factors, expected at least {N_FACTORS}.")
		model_samples[latent_name] = (samples, posterior_mean)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig = plot_dim1_uncertainty_grid(model_samples)

	out_file = OUTPUT_DIR / "uncertainty_grid_dim1.pdf"
	fig.tight_layout(pad=TIGHT_LAYOUT_PAD, w_pad=TIGHT_LAYOUT_W_PAD, h_pad=TIGHT_LAYOUT_H_PAD)
	fig.savefig(out_file, bbox_inches="tight", transparent=True)
	plt.close(fig)

	print(f"Saved {out_file}")


if __name__ == "__main__":
	main()
