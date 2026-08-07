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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FIG_WIDTH_CM = 5.6
FIG_HEIGHT_CM = 2.6
FONT_SIZE = 7
P_THRESHOLDS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def load_weight_table(csv_path):
	df = pd.read_csv(csv_path)
	weights = df[df["name"].str.startswith("W_view_")].copy()
	if weights.empty:
		raise ValueError(f"No weight rows found in {csv_path}")

	max_p_value = float(weights["p_value"].max())

	weights["view"] = weights["name"].str.replace("W_view_", "", regex=False)
	pivot = (
		weights.pivot(index="factor", columns="view", values="corr")
		.sort_index()
		.reindex(columns=["1", "2", "3"])
	)
	return pivot, max_p_value


def threshold_p_value(p_value):
	threshold = p_value
	for p in P_THRESHOLDS:
		if threshold < p:
			threshold = p
			break
		if p == 1e-2 and threshold >= p:
			raise ValueError(
				f"Check p-value threshold: p-value is {threshold:.2e}, which is above the threshold of {p}."
			)
	return threshold


def p_threshold_label(p_value):
	threshold = threshold_p_value(p_value)
	return f"{threshold:.0e}".replace("e-0", "e-")


def build_view_dataframe(weight_table):
	return pd.DataFrame({"View": [f"F{idx}" for idx in weight_table.index]})


def build_values_dataframe(weight_table):
	return pd.DataFrame(
		{
			"1": [f"{val:.2f}" for val in weight_table["1"].to_numpy()],
			"2": [f"{val:.2f}" for val in weight_table["2"].to_numpy()],
			"3": [f"{val:.2f}" for val in weight_table["3"].to_numpy()],
		}
	)


def draw_table(ax, display_df):
	ax.axis("off")
	table = ax.table(
		cellText=display_df.values,
		colLabels=display_df.columns,
		loc="center",
		cellLoc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(FONT_SIZE)
	table.scale(1.0, 1.10)

	for (row, col), cell in table.get_celld().items():
		cell.set_linewidth(0.6)
		if row == 0:
			cell.set_facecolor("#e8e8e8")


def save_combined_table(dim1_table, dim2_30_table, max_p_value, output_dir):
	fig, axes = plt.subplots(
		1,
		3,
		figsize=(FIG_WIDTH_CM / 2.54, FIG_HEIGHT_CM / 2.54),
		gridspec_kw={"width_ratios": [1, 3, 3]},
	)

	draw_table(axes[0], build_view_dataframe(dim1_table))
	draw_table(axes[1], build_values_dataframe(dim1_table))
	draw_table(axes[2], build_values_dataframe(dim2_30_table))

	axes[1].set_title("1D", fontsize=FONT_SIZE, pad=10)
	axes[2].set_title("2D", fontsize=FONT_SIZE, pad=10)

	p_label = p_threshold_label(max_p_value)
	fig.text(
		0.5,
		0.02,
		f"Weight reconstruction (Pearson r, p<{p_label})",
		ha="center",
		va="bottom",
		fontsize=FONT_SIZE,
	)
	fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0], pad=0.05, w_pad=0.25, h_pad=0.05)

	out_pdf = output_dir / "weight_corr_table_dim1_dim2_30.pdf"
	fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
	plt.close(fig)
	print(f"Saved {out_pdf}")


def save_single_table(weight_table, max_p_value, output_dir, suffix):
	fig, axes = plt.subplots(
		1,
		2,
		figsize=(FIG_WIDTH_CM / 2.54, FIG_HEIGHT_CM / 2.54),
		gridspec_kw={"width_ratios": [1, 3]},
	)
	draw_table(axes[0], build_view_dataframe(weight_table))
	draw_table(axes[1], build_values_dataframe(weight_table))

	p_label = p_threshold_label(max_p_value)
	fig.text(
		0.5,
		0.0,
		f"Weight reconstruction (Pearson r, p<{p_label})",
		ha="center",
		va="bottom",
		fontsize=FONT_SIZE,
	)
	fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0], pad=0.05, w_pad=0.25, h_pad=0.05)

	out_pdf = output_dir / f"weight_corr_table_{suffix}.pdf"
	fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
	plt.close(fig)
	print(f"Saved {out_pdf}")


def main():
	base = Path("../output")
	dim1_csv = base / "output_syn_rec_cfm_dim1/runs_combined_factors_5/recovery_metrics.csv"
	dim2_30_csv = base / "output_syn_rec_cfm_dim2_30/runs_combined_factors_5/recovery_metrics.csv"
	dim2_100_csv = base / "output_syn_rec_cfm_dim2_100/runs_combined_factors_5/recovery_metrics.csv"

	for csv_path in [dim1_csv, dim2_30_csv, dim2_100_csv]:
		if not csv_path.exists():
			raise FileNotFoundError(f"Missing file: {csv_path}")

	output_dir = Path("../paper/syn_tables")
	output_dir.mkdir(parents=True, exist_ok=True)

	dim1_table, dim1_max_p = load_weight_table(dim1_csv)
	dim2_30_table, dim2_30_max_p = load_weight_table(dim2_30_csv)
	dim2_100_table, dim2_100_max_p = load_weight_table(dim2_100_csv)

	save_combined_table(dim1_table, dim2_30_table, max(dim1_max_p, dim2_30_max_p), output_dir)
	save_single_table(dim2_100_table, dim2_100_max_p, output_dir, "dim2_100")


if __name__ == "__main__":
	main()
