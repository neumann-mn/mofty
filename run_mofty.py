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
import copy
import cProfile
import pstats
import os
import pickle
import shutil
import subprocess
import sys
from functools import reduce
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nifty.cl as ift
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from scipy.stats import gamma, norm, pearsonr

import mofty_modules.custom_op as custom_op
import mofty_modules.plot as plot_utils
import mofty_modules.utils as utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--rp", type=int, default=1)
    parser.add_argument(
        "--np",
        type=int,
        default=None,
        help="Number of MPI processes to use for each task. If 0, no MPI is used.",
    )
    parser.add_argument(
        "--device", type=str, default="CPU", help="Device to use: CPU or GPU."
    )
    parser.add_argument(
        "--run_task", type=str, help="Internal use: run a single serialized task."
    )
    parser.add_argument(
        "--only_summarize", action="store_true", help="Only summarize runs."
    )
    parser.add_argument(
        "--profile_worker",
        action="store_true",
        help="Enable cProfile for the worker task.",
    )
    parser.add_argument(
        "--grid_check",
        type=float,
        default=0,
        help="Check grid parameters for CFM initialization.",
    )
    parser.add_argument(
        "--force_grid",
        action="store_true",
        help="Force grid creation even if estimated grid points exceed limit.",
    )
    return parser


def mofty_main(args):
    sn = SimpleNamespace(**args)

    if sn.syn_data_flag and sn.syn_data_rec_flag:
        raise ValueError(
            "Cannot set both syn_data_flag and syn_data_rec_flag to True."
        )

    if not sn.summarize_runs_flag:
        ift.random.push_sseq_from_seed(sn.seed)

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            master = comm.Get_rank() == 0
        except ImportError:
            comm = None
            master = True

        if master:
            if not sn.syn_data_flag:
                print(
                    f"Starting run for seed: {sn.seed}, factors: {sn.n_factors}"
                )
                print()
            else:
                print(f"Generating synthetic data for seed: {sn.seed}, factors: {sn.n_factors}")

        if sn.syn_data_flag:
            output_dir = sn.input_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        elif not sn.syn_data_flag:
            output_dir = (
                sn.output_base_dir / f"run_seed_{sn.seed}_factors_{sn.n_factors}"
            )
            (output_dir / "pickle").mkdir(parents=True, exist_ok=True)
            if master:
                print(f"Number of factors: {sn.n_factors}")
                print(f"Output directory: {output_dir}")
                output_config_path = output_dir / f"config.yaml"
                run_config = copy.deepcopy(sn.config)
                run_config["SEEDS"] = [sn.seed]
                if sn.only_ref_seed_init and sn.seed != sn.ref_seed:
                    run_config["FACTOR_INIT"] = "random"

                run_config["N_FACTORS_LIST"] = [sn.n_factors]

                run_config.pop("REF_SEED_DICT", None)
                run_config.pop("N_FACTORS_INIT", None)
                run_config.pop("N_FACTORS_FINAL", None)
                run_config.pop("N_FACTORS_STEP", None)
                run_config.pop("N_SEEDS_PER_STEP", None)

                output_config_path = output_dir / "master_config.yaml"
                with open(output_config_path, "w") as f:
                    yaml.safe_dump(run_config, f, sort_keys=False)
        if sn.only_ref_seed_init and sn.seed != sn.ref_seed:
            sn.factor_init = "random"
            print(
                f"Random factor initialization since only_ref_seed_init is set and seed {sn.seed} not equal to {sn.ref_seed}"
            )

    else:
        master = True
        comm = None
        output_dir = sn.output_base_dir / f"runs_combined_factors_{sn.n_factors}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        (output_dir / "model_plots").mkdir(parents=True, exist_ok=True)
        (output_dir / "pickle").mkdir(parents=True, exist_ok=True)

    # =============================
    # ======= Forward model =======
    # =============================

    mask = {kk: ~np.isnan(v) for kk, v in sn.data.items()}
    n_data_obs = list(sn.data.values())[0].shape[1]
    modalities = {kk: v.shape[0] for kk, v in sn.data.items()}

    export_operator_outputs = {}
    dom_samples = ift.UnstructuredDomain(n_data_obs)
    dom_latent_factors = ift.UnstructuredDomain(sn.n_factors)
    Z_space = (dom_latent_factors, dom_samples)

    # ======= Correlated field model =======

    if sn.init_cfm:
        pix, centered_cov_samples, rg_coords, RG_dom = utils.calculate_grid_parameters(
            sn.cov_samples_transformed,
            sn.covariate_keys,
            sn.grid_scaling,
            sn.padding,
            force_grid=sn.force_grid,
        )
        cfm = ift.CorrelatedFieldMaker(prefix="Z_cfm_op_", total_N=sn.n_factors)
        cfm.set_amplitude_total_offset(
            offset_mean=sn.offset_mean,
            offset_std=sn.offset_std,
            dofdex=tuple(range(sn.n_factors)),
        )
        cfm.add_fluctuations(
            prefix="",
            target_subdomain=RG_dom,
            fluctuations=tuple(sn.fluctuations),
            loglogavgslope=tuple(sn.loglogavgslope),
            flexibility=tuple(sn.flexibility),
            asperity=tuple(sn.asperity),
            dofdex=tuple(range(sn.n_factors)),
        )
        Z_cfm_op = cfm.finalize()

        export_operator_outputs["Z_cfm_op"] = Z_cfm_op
        Z_cfm_op_pspec = cfm.power_spectrum
        export_operator_outputs["Z_cfm_op_pspec"] = Z_cfm_op_pspec

        if master:
            print()
            print("CFM configuration:")
            print(f"- CFM grid parameters: pix={pix}, padding={sn.padding}, grid_scaling={sn.grid_scaling}")
            print(f"- Offset mean: {sn.offset_mean}")
            print(f"- Offset std: {sn.offset_std}")
            print(f"- Fluctuations: {sn.fluctuations}")
            print(f"- Log-log average slope: {sn.loglogavgslope}")
            print(f"- Flexibility: {sn.flexibility}")
            print(f"- Asperity: {sn.asperity}")
            print()

        L_cfm = custom_op.FactorwiseLinearInterpolator(
            tuple(Z_cfm_op.target), sampling_points=centered_cov_samples.T
        )
        ift.extra.check_linear_operator(L_cfm, rtol=1e-12, atol=1e-12)
        Z_cfm = L_cfm @ Z_cfm_op

        if sn.init_cfm_scaling:
            bc = ift.ContractionOperator(Z_space, (1,)).adjoint
            sf = ift.BetaOperator(
                dom_latent_factors, a=sn.beta_param, b=sn.beta_param
            ).ducktape("Z_cfm_op_scaling")
            if sn.syn_data_flag:
                sf = bc @ sf
                export_operator_outputs["Z_cfm_op_scaling"] = sf
            else:
                # reduce memory/storage by avoiding broadcasted values
                export_operator_outputs["Z_cfm_op_scaling"] = sf
                sf = bc @ sf
            Z_cfm = sf.clip(1e-12, None).sqrt() * Z_cfm

        if sn.init_non_cfm:
            export_operator_outputs["Z_cfm"] = Z_cfm
            Z_non_cfm = ift.ScalingOperator(Z_space, 1).ducktape("Z_non_cfm")
            if sn.init_cfm_scaling:
                Z_non_cfm = (1 - sf).clip(1e-12, None).sqrt() * Z_non_cfm
            export_operator_outputs["Z_non_cfm"] = Z_non_cfm
            Z = Z_cfm + Z_non_cfm
        else:
            Z = Z_cfm

    # ======= Standard factor initialization =======

    elif not sn.init_cfm:
        Z = ift.ScalingOperator(Z_space, 1).ducktape("Z")

    if sn.igamma_Z:
        Z_igamma = (
            ift.InverseGammaOperator(
                Z_space,
                alpha=1 + sn.igamma_param,
                q=sn.igamma_param,
            )
            .clip(0, None)
            .ducktape("Z_igamma")
        )
        Z = Z * Z_igamma

    if sn.include_intercept:
        dom_latent_factors = ift.UnstructuredDomain(sn.n_factors)
        Z_space = (dom_latent_factors, dom_samples)
        arr = np.ones((sn.n_factors, n_data_obs))
        arr[-1, :] = 0
        op0 = ift.makeOp(ift.makeField(Z_space, arr))
        arr = np.zeros((sn.n_factors, n_data_obs))
        arr[-1, :] = 1
        op1 = ift.Adder(ift.makeField(Z_space, arr))
        Z = op1 @ op0 @ Z

    if sn.softplus_Z:
        Z = Z.softplus()
    if sn.exp_Z:
        Z = Z.exp()

    export_operator_outputs["Z"] = Z

    model_data = {}
    lh = {}
    likelihood_energy = []
    mask_ops = []

    for kk, D_m in modalities.items():
        dom_features = ift.UnstructuredDomain(D_m)
        W_space = (dom_features, dom_latent_factors)
        W = ift.ScalingOperator(W_space, 1).ducktape(f"W_gaussian_{kk}")
        if sn.igamma_W:
            igamma_w = (
                ift.InverseGammaOperator(
                    W_space,
                    alpha=1 + sn.igamma_param,
                    q=sn.igamma_param,
                )
                .clip(0, None)
                .ducktape(f"W_igamma_{kk}")
            )
            W = igamma_w * W

        if sn.softplus_W:
            W = W.softplus()
        if sn.exp_W:
            W = W.exp()

        export_operator_outputs[f"W_{kk}"] = W

        # Model data
        model_data[kk] = custom_op.MatrixMultiply(W_space, f"W_{kk}", Z.target, "Z") @ (
            Z.ducktape_left("Z") + W.ducktape_left(f"W_{kk}")
        )

        d_field = ift.makeField(model_data[kk].target, sn.data[kk])
        mask_op = custom_op.MaskOperator(ift.makeField(model_data[kk].target, mask[kk]))

        if sn.likelihood_options[kk] == "gaussian":
            residuals = ift.Adder(d_field, neg=True) @ model_data[kk]
            residuals = (mask_op @ residuals).ducktape_left("residuals")

            if sn.noise_axis == "row_wise":
                aa = 1
                bc = ift.ContractionOperator(mask_op.domain, (aa,)).adjoint
                icov = ift.GammaOperator(
                    dom_features,
                    alpha=sn.icov_gamma_param,
                    beta=sn.icov_gamma_param
                ).clip(1e-12, None).ducktape(f"{kk}_icov_gamma")
                if sn.syn_data_flag:
                    icov = bc @ icov
                    export_operator_outputs[f"icov_{kk}"] = icov
                    export_operator_outputs[f"std_{kk}"] = icov.reciprocal().sqrt()
                else:
                    # reduce memory/storage by avoiding broadcasted values
                    export_operator_outputs[f"icov_{kk}"] = icov
                    export_operator_outputs[f"std_{kk}"] = icov.reciprocal().sqrt()
                    icov = bc @ icov

            elif sn.noise_axis == "simple":
                bc = ift.ContractionOperator(mask_op.domain, spaces=None).adjoint
                icov = ift.GammaOperator(bc.domain,
                    alpha=sn.icov_gamma_param,
                    beta=sn.icov_gamma_param,
                ).clip(1e-12, None).ducktape(f"{kk}_icov_gamma")
                if sn.syn_data_flag:
                    icov = bc @ icov
                    export_operator_outputs[f"icov_{kk}"] = icov
                    export_operator_outputs[f"std_{kk}"] = icov.reciprocal().sqrt()
                else:
                    # reduce memory/storage by avoiding broadcasted values
                    export_operator_outputs[f"icov_{kk}"] = icov
                    export_operator_outputs[f"std_{kk}"] = icov.reciprocal().sqrt()
                    icov = bc @ icov

            icov = (mask_op @ icov).ducktape_left("icov")
            lh[kk] = ift.VariableCovarianceGaussianEnergy(
                mask_op.target, "residuals", "icov", d_field.dtype
            ) @ (residuals + icov)

        elif sn.likelihood_options[kk] == "bernoulli":
            model_data[kk] = ift.sigmoid(model_data[kk].clip(-4.5, 4.5))
            d_field = mask_op(d_field)
            d_field = ift.makeField(d_field.domain, d_field.asnumpy().astype(int))
            lh[kk] = ift.BernoulliEnergy(d_field) @ mask_op @ model_data[kk]

        elif sn.likelihood_options[kk] == "poisson":
            if sn.poisson_exp:
                model_data[kk] = model_data[kk].exp()
            if sn.poisson_softplus:
                model_data[kk] = model_data[kk].softplus()
            d_field = mask_op(d_field)
            d_field = ift.makeField(d_field.domain, d_field.asnumpy().astype(int))
            lh[kk] = ift.PoissonianEnergy(d_field) @ mask_op @ model_data[kk]

        else:
            raise RuntimeError(f"Unknown modality {kk}")

        lh[kk].name = kk
        likelihood_energy.append(lh[kk])
        if len(modalities) == 1:
            mask_op = mask_op.ducktape(kk)
        else:
            mask_op = mask_op.ducktape(kk).ducktape_left(kk)

        mask_ops.append(mask_op)

    likelihood_energy = reduce(lambda x, y: x + y, likelihood_energy)
    mask_ops = reduce(lambda x, y: x + y, mask_ops)

    # =========================================
    # ======= SYNTHETIC DATA GENERATION =======
    # =========================================

    if sn.syn_data_flag:
        a_beta, b_beta = sn.beta_param, sn.beta_param
        x_beta_vals = []
        for val in sn.cfm_scaling_values:
            x_beta = utils.beta_to_latent(val, a_beta, b_beta)
            x_beta_vals.append(x_beta)
        obs_names = [f"sample_{i}" for i in range(n_data_obs)]
        syn_pos = ift.from_random(likelihood_energy.domain).to_dict()
        for kk, vv in syn_pos.items():
            if "W_igamma" in kk:
                c = sn.W_igamma_syn_scaling
            else:
                c = sn.syn_scaling
            sf = ift.full(vv.domain, c).asnumpy()
            syn_pos[kk] = ift.makeField(vv.domain, sf * syn_pos[kk].asnumpy())
            if kk == "Z_cfm_op_scaling":
                syn_pos[kk] = ift.makeField(syn_pos[kk].domain, np.array(x_beta_vals))
        syn_pos = ift.MultiField.from_dict(syn_pos)

        for op_name, op in export_operator_outputs.items():
            if "W" in op_name or "icov" in op_name:
                continue
            syn_op = op.force(syn_pos)
            if op_name == "Z_cfm_op_scaling":
                syn_op_df = pd.DataFrame(
                    syn_op.asnumpy(),
                    index=[i for i in range(sn.n_factors)],
                    columns=obs_names,
                )
                syn_op_df.to_parquet(output_dir / f"{op_name}.parquet", index=True)
            elif op_name.startswith("Z") and op_name not in ["Z_cfm_op_pspec", "Z_cfm_op"]:
                coords = sn.cov_samples_transformed
                if len(sn.covariate_keys) in [2, 3]:
                    plot_utils.cfm_plotting(
                        syn_op.asnumpy(),
                        op_name,
                        coords=coords,
                        covariate_keys=sn.covariate_keys,
                        z_slices=sn.z_slices,
                        flip_image=sn.flip_image,
                        plot_dot_size=sn.plot_dot_size,
                        padding=sn.padding,
                        output_dir=output_dir / "model_plots",
                        master=master,
                    )
                else:
                    plot_utils.standard_plotting(
                        syn_op.asnumpy(),
                        op_name,
                        xlabel="Samples",
                        ylabel="Factors",
                        output_dir=output_dir / "model_plots",
                        master=master,
                        percentile=100,
                    )

                syn_op_df = pd.DataFrame(
                    syn_op.asnumpy(),
                    index=[i for i in range(sn.n_factors)],
                    columns=obs_names,
                )
                syn_op_df.to_parquet(output_dir / f"{op_name}.parquet", index=True)

            elif op_name == "Z_cfm_op_pspec":
                fig, ax = plt.subplots()
                power_spectrum_data = syn_op.asnumpy()
                for i in range(power_spectrum_data.shape[0]):
                    ax.plot(power_spectrum_data[i, :], label=f"Factor {i+1}")
                ax.set_xlabel("|k|")
                ax.set_ylabel("p(|k|)")
                ax.set_title("")
                ax.set_yscale("log")
                ax.set_xscale("log")
                if sn.n_factors <= 10:
                    ax.legend()
                plt.tight_layout()
                plot_path = output_dir / "model_plots"
                plot_path.mkdir(parents=True, exist_ok=True)
                plot_path = plot_path / f"{op_name}_synthetic.png"
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close(fig)
                syn_op_df = pd.DataFrame(syn_op.asnumpy())
                syn_op_df.to_parquet(output_dir / f"{op_name}.parquet", index=True)

        for kk, vv in sn.likelihood_options.items():
            feature_names = [f"{kk}_feature_{i}" for i in range(modalities[kk])]
            syn_data = model_data[kk].force(syn_pos)
            for op_name, op in export_operator_outputs.items():
                if f"W_{kk}" not in op_name:
                    continue
                syn_op = op.force(syn_pos)
                val = syn_op.asnumpy()
                plot_utils.standard_plotting(
                    syn_op.asnumpy(),
                    op_name,
                    xlabel="Factors",
                    ylabel="Features",
                    output_dir=output_dir / "model_plots",
                    master=master,
                    percentile=99.95,
                )
                syn_op_df = pd.DataFrame(
                    syn_op.asnumpy(),
                    index=feature_names,
                    columns=[i for i in range(sn.n_factors)],
                )
                syn_op_df.to_parquet(output_dir / f"{op_name}.parquet", index=True)

            if vv == "gaussian":
                plot_utils.standard_plotting(
                    syn_data.asnumpy(),
                    f"{kk}_model_data",
                    xlabel="Samples",
                    ylabel="Features",
                    output_dir=output_dir / "model_plots",
                    master=master,
                )
                icov = export_operator_outputs[f"icov_{kk}"]
                N_syn = icov.force(syn_pos).reciprocal().sqrt()
                plot_utils.standard_plotting(
                    N_syn.asnumpy(),
                    f"{kk}_std_unscaled",
                    xlabel="Samples",
                    ylabel="Features",
                    output_dir=output_dir / "model_plots",
                    master=master,
                )
                noise_syn = ift.makeOp(N_syn, sampling_dtype=float).draw_sample()
                syn_data_centered = syn_data.asnumpy() - np.mean(
                    syn_data.asnumpy(), axis=1, keepdims=True
                )
                signal_std = np.std(syn_data_centered)
                noise_std = np.std(noise_syn.asnumpy())
                current_snr = signal_std / noise_std
                scale_factor = current_snr / sn.target_snr
                scaled_noise = scale_factor * noise_syn
                noisy_syn_data = syn_data + scaled_noise

                plot_utils.standard_plotting(
                    scaled_noise.asnumpy(),
                    f"{kk}_noise_unmasked",
                    xlabel="Samples",
                    ylabel="Features",
                    output_dir=output_dir / "model_plots",
                    master=master,
                )
                noise_syn_df = pd.DataFrame(
                    scaled_noise.asnumpy(), index=feature_names, columns=obs_names
                )
                noise_syn_df.to_parquet(output_dir / f"{kk}_noise_unmasked.parquet", index=True)

                scaled_noise_masked = sn.data_mask[kk] * scaled_noise.asnumpy()
                plot_utils.standard_plotting(
                    scaled_noise_masked,
                    f"{kk}_noise",
                    xlabel="Samples",
                    ylabel="Features",
                    output_dir=output_dir / "model_plots",
                    master=master,
                )

            elif vv == "poisson":
                rounded_val = np.round(syn_data.asnumpy()).astype(int)
                syn_data = ift.makeField(syn_data.domain, rounded_val)
                noisy_syn_data = syn_data
            elif vv == "bernoulli":
                rounded_val = np.round(syn_data.asnumpy()).astype(int)
                syn_data = ift.makeField(syn_data.domain, rounded_val)
                noisy_syn_data = syn_data
            else:
                raise RuntimeError(
                    f"Unknown likelihood option {vv} for synthetic data generation."
                )

            syn_data_df = pd.DataFrame(
                syn_data.asnumpy(), index=feature_names, columns=obs_names
            )
            syn_data_df.to_parquet(output_dir / f"{kk}_model_data.parquet", index=True)

            noisy_syn_data_masked = sn.data_mask[kk] * noisy_syn_data.asnumpy()
            plot_utils.standard_plotting(
                noisy_syn_data_masked,
                f"{kk}",
                xlabel="Samples",
                ylabel="Features",
                output_dir=output_dir / "model_plots",
                master=master,
            )
            noisy_syn_data_masked_df = pd.DataFrame(
                noisy_syn_data_masked, index=feature_names, columns=obs_names
            )
            noisy_syn_data_masked_df.to_parquet(output_dir / f"{kk}.parquet", index=True)
        return

    # ====================================
    # ======= Callback optimize KL =======
    # ====================================

    def callback(samples_object, iteration):
        if sn.summarize_runs_flag:
            samples, samples_orig = samples_object
        else:
            samples = samples_object

        if sn.summarize_runs_flag:
            normres_avg = {}
            for ii in samples_orig.keys():
                normres_avg[ii] = samples_orig[ii].average(
                    lambda x: mask_ops.adjoint(likelihood_energy.normalized_residual(x))
                )
        else:
            normres_avg = samples.average(
                lambda x: mask_ops.adjoint(likelihood_energy.normalized_residual(x))
            )

        for kk in modalities:
            if sn.summarize_runs_flag:
                normres_avg_kk = {}
                for seed in samples_orig.keys():
                    normres_avg_kk[seed] = normres_avg[seed][kk]
                normres_avg_kk = reduce(lambda x, y: x + y, normres_avg_kk.values())
                normres_avg_kk = normres_avg_kk / len(samples_orig.keys())
                val = normres_avg_kk.asnumpy()
            else:
                val = normres_avg[kk].asnumpy()

            # Build full residual matrix (features x samples) with NaNs where masked
            n_features = modalities[kk]
            n_obs_samples = len(sn.obs_names)
            full_residuals = np.full((n_features, n_obs_samples), np.nan)
            full_residuals[mask[kk]] = val[mask[kk]]

            if iteration == sn.n_iterations - 1:
                feature_names = sn.feature_names_dict.get(kk)
                if not (
                    feature_names and len(feature_names) == full_residuals.shape[0]
                ):
                    feature_names = [
                        f"feature_{j}_{kk}" for j in range(full_residuals.shape[0])
                    ]
                df = pd.DataFrame(
                    full_residuals, index=feature_names, columns=sn.obs_names
                )
                parquet_path = output_dir / f"normalized_residuals_{kk}.parquet"
                df.to_parquet(parquet_path, index=True)

            # Plot the full 2D residuals (not the flattened vector)
            plot_utils.standard_plotting(
                full_residuals,
                f"normalized_residuals_{kk}",
                xlabel="Samples",
                ylabel="Features",
                output_dir=output_dir / "model_plots",
                master=master,
                iteration=iteration,
            )

        for kk, op_kk in export_operator_outputs.items():
            if sn.summarize_runs_flag:
                if kk in {"Z_cfm_op", "Z_cfm_op_scaling"}:
                    continue
                current_samples = samples[kk]
            else:
                current_samples = samples
            vals_to_plot = {}
            if sn.summarize_runs_flag:
                mean, var = current_samples.sample_stat(op=None)
            else:
                mean, var = current_samples.sample_stat(op=op_kk)
            std_val = var.sqrt().asnumpy()
            mean_val = mean.asnumpy()
            vals_to_plot["mean"] = mean_val
            vals_to_plot["std"] = std_val
            if kk == "Z_cfm_op_pspec":
                if sn.summarize_runs_flag:
                    op_kk = None
                plot_utils.pspec_plotting(
                    op_kk,
                    current_samples,
                    kk,
                    n_factors=sn.n_factors,
                    iteration=iteration,
                    output_dir=output_dir / "model_plots",
                    master=master
                )
                continue

            for stat_name, val in vals_to_plot.items():
                if kk.startswith("Z"):
                    if (
                        sn.use_covariates
                        and len(sn.covariate_keys) in [2, 3]
                        and kk not in ["Z_cfm_op_pspec", "Z_cfm_op_scaling"]
                    ):
                        if kk == "Z_cfm_op":
                            coords = rg_coords
                        else:
                            coords = sn.cov_samples_transformed
                        plot_utils.cfm_plotting(
                            val,
                            kk,
                            coords=coords,
                            covariate_keys=sn.covariate_keys,
                            stat_name=stat_name,
                            z_slices=sn.z_slices,
                            flip_image=sn.flip_image,
                            plot_dot_size=sn.plot_dot_size,
                            padding=sn.padding,
                            output_dir=output_dir / "model_plots",
                            master=master,
                            iteration=iteration,
                        )
                        continue
                    else:
                        if kk == "Z_cfm_op_scaling" and stat_name == "std":
                            continue
                        if kk == "Z_cfm_op_scaling":
                            val = val.reshape(-1,1)
                        if kk != "Z_cfm_op_pspec":
                            xlabel = "Samples"
                        else:
                            xlabel = "Frequency"
                        ylabel = "Factors"
                elif kk.startswith("W"):
                    xlabel = "Factors"
                    ylabel = "Features"
                elif kk.startswith("std"):
                    if stat_name == "std":
                        continue
                    val = val.reshape(-1,1)
                    xlabel = "Samples"
                    ylabel = "Features"
                else:
                    continue

                plot_utils.standard_plotting(
                    val,
                    kk,
                    stat_name=stat_name,
                    iteration=iteration,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    output_dir=output_dir / "model_plots",
                    master=master,
                )

    # =========================================
    # ======= Sampling and Optimization =======
    # =========================================

    if not sn.summarize_runs_flag:

        def ic_sampling(limit, name="Linear_sampling"):
            return ift.AbsDeltaEnergyController(
                name=name, deltaE=0.5, convergence_level=1, iteration_limit=limit
            )

        def minimizer_VL_BFGS(limit, name="VL_BFGS"):
            return ift.VL_BFGS(
                ift.AbsDeltaEnergyController(
                    name=name, deltaE=0.5, convergence_level=2, iteration_limit=limit
                )
            )

        def ic_sampling_cont(iteration):
            if iteration < sn.n_iterations_low_opt:
                return ic_sampling(200)
            else:
                return ic_sampling(1000)

        def minimizer(iteration):
            if iteration < sn.n_iterations_low_opt:
                return minimizer_VL_BFGS(200)
            else:
                return minimizer_VL_BFGS(1000)

        def geoVI_minimizer(iteration):
            if iteration < sn.n_iterations_low_opt:
                return minimizer_VL_BFGS(50, name="VL_BFGS_non_linear")
            else:
                return minimizer_VL_BFGS(300, name="VL_BFGS_non_linear")

        if sn.inference_method == "geoVI":
            minimizer_non_linear = geoVI_minimizer
        elif sn.inference_method == "MGVI":
            minimizer_non_linear = None

        setup_point_estimates = [
            "gamma",
            "Z_cfm_op_scaling",
            "Z_cfm_op_loglogavgslope",
            "Z_cfm_op_asperity",
            "Z_cfm_op_flexibility",
            "Z_cfm_op_fluctuations",
            "Z_cfm_op_zeromode",
        ]

        point_estimates = tuple(
            filter(
                lambda x: any(substr in x for substr in setup_point_estimates),
                likelihood_energy.domain.keys(),
            )
        )

        if master:
            print(f"Point estimates: {point_estimates}")
            print(f"Inference method: {sn.inference_method}")
            print(
                f"Samples: {sn.n_start_samples} for the first "
                f"{sn.n_start_samples_iterations} iterations, then {sn.n_final_samples}"
            )
            print(f"Initialization: {sn.factor_init}")
            print()

        initial_position = 0.1 * ift.from_random(likelihood_energy.domain)
        initial_position = initial_position.to_dict()

        x_gamma = norm.ppf(
            gamma.cdf(
                sn.noise_icov_initial,
                a=sn.icov_gamma_param,
                scale= 1 / sn.icov_gamma_param,
            )
        )

        a_beta, b_beta = sn.beta_param, sn.beta_param
        target_beta_value = float(sn.beta_init)
        x_beta = utils.beta_to_latent(target_beta_value, a_beta, b_beta)

        initial_position = utils.initialization_setup(
            initial_position,
            ref_seed=sn.ref_seed,
            x_gamma=x_gamma,
            x_beta=x_beta,
            only_ref_seed_init=sn.only_ref_seed_init,
            seed=sn.seed,
            factor_init=sn.factor_init,
            data_keys=sn.data_keys,
            skip_init=sn.skip_init,
            data=sn.data,
            center_data=sn.center_data,
            scale_views=sn.scale_views,
            n_factors=sn.n_factors,
            cfm_scaling_init_values=sn.cfm_scaling_init_values,
            master=master,
        )

        initial_position = ift.MultiField.from_dict(initial_position)
        setup = ["icov_gamma"]
        lh_keys = likelihood_energy.domain.keys()

        def n_samples_fun(iteration):
            if iteration < sn.n_start_samples_iterations:
                return sn.n_start_samples
            else:
                return sn.n_final_samples

        def constants(iteration):
            if iteration < sn.n_iterations_noise_constant:
                return tuple(
                    filter(lambda x: any(substr in x for substr in setup), lh_keys)
                )
            return []

        np.seterr(invalid="raise")

        if sn.device == "CPU":
            device_id = -1
        elif sn.device == "GPU":
            device_id = 0

        samples = ift.optimize_kl(
            likelihood_energy,
            total_iterations=sn.n_iterations,
            n_samples=n_samples_fun,
            kl_minimizer=minimizer,
            sampling_iteration_controller=ic_sampling_cont,
            nonlinear_sampling_minimizer=minimizer_non_linear,
            export_operator_outputs=export_operator_outputs,
            initial_position=initial_position,
            inspect_callback=callback,
            output_directory=output_dir,
            point_estimates=point_estimates,
            constants=constants,
            resume=sn.resume_training,
            device_id=device_id,
            save_strategy="latest",
        )

    # =================================
    # ======= Summarize Results =======
    # =================================

    if sn.summarize_runs_flag:
        print(f"Summarizing results for factors: {sn.n_factors}")

        sample_dict_summary = sn.sample_dict_summary

        if sn.syn_data_rec_flag:
            permutation_matrix = np.eye(sn.n_factors)
        else:
            factor_r2 = {}

            for kk in sn.data_keys:
                # Read the CSV file
                df = pd.read_csv(
                    sn.output_base_dir
                    / f"run_seed_{sn.ref_seed}_factors_{sn.n_factors}"
                    / f"Z_r2_stats_{kk}.csv"
                )
                factor_r2[kk] = df.set_index("factor_index")["r2_per_factor"].to_dict()

            # Sum the R2 values across all data keys
            sum_factor_r2 = pd.DataFrame(factor_r2).sum(axis=1).to_dict()
            # Sort factors by their total R2 values in descending order
            ordered_factors = sorted(sum_factor_r2, key=sum_factor_r2.get, reverse=True)
            n_factors = len(ordered_factors)

            # Initialize a permutation matrix
            if sn.include_intercept:
                # Assume intercept is last factor; remove it from permutation
                intercept_idx = n_factors - 1
                non_intercept_order = [i for i in ordered_factors if i != intercept_idx]
                if len(non_intercept_order) != n_factors - 1:
                    raise ValueError("Mismatch in non-intercept factor ordering.")
                permutation_matrix = np.eye(n_factors)
                permutation_matrix[:-1, :-1] = np.eye(n_factors - 1)[
                    non_intercept_order
                ]
            else:
                permutation_matrix = np.eye(n_factors)[ordered_factors]

        Z_op = export_operator_outputs["Z"]
        Z_mean_dict = {}
        op_samples = {}
        for seed in sample_dict_summary.keys():
            mean = sample_dict_summary[seed].average(op=Z_op)
            Z_val = mean.asnumpy()
            Z_mean_dict[seed] = Z_val
        W_mean_dict = {}
        for seed in sample_dict_summary.keys():
            W_list = []
            for kk, op_kk in export_operator_outputs.items():
                if kk.startswith("W"):
                    mean = sample_dict_summary[seed].average(op=op_kk)
                    W_val = mean.asnumpy()
                    W_list.append(W_val)
            W_mean_dict[seed] = np.concatenate([W_val for W_val in W_list], axis=0)

        ot_matrices = {}

        if sn.exp_Z or sn.softplus_Z:
            for seed in sample_dict_summary.keys():
                if not sn.syn_data_rec_flag and seed == sn.ref_seed:
                    ot_matrices[seed] = permutation_matrix.T
                    continue

                Z_mean = Z_mean_dict[seed]

                if sn.syn_data_rec_flag:
                    Z_parquet_path = sn.input_dir / "Z.parquet"
                    Z_ref = pd.read_parquet(Z_parquet_path).to_numpy()
                else:
                    Z_ref = permutation_matrix @ Z_mean_dict[sn.ref_seed]

                # If intercept is included, don't align the last row/column (keep it fixed)
                if sn.include_intercept:
                    Z_mean = Z_mean[:-1, :]
                    Z_ref = Z_ref[:-1, :]

                # Standardize per row
                Z1 = Z_mean - Z_mean.mean(axis=1, keepdims=True)
                Z2 = Z_ref - Z_ref.mean(axis=1, keepdims=True)

                z1_std = Z1.std(axis=1, ddof=1, keepdims=True)
                z1_std[z1_std == 0] = 1
                z2_std = Z2.std(axis=1, ddof=1, keepdims=True)
                z2_std[z2_std == 0] = 1

                Z1 = Z1 / z1_std
                Z2 = Z2 / z2_std

                # Correlation / similarity matrix
                C = Z1 @ Z2.T  # shape (k, k)

                # Nonnegative orthogonal => permutation. Choose permutation maximizing trace(P @ C).
                # linear_sum_assignment minimizes cost, so minimize -C.
                row_ind, col_ind = linear_sum_assignment(-C)

                k = C.shape[0]
                P = np.zeros((k, k), dtype=float)
                P[row_ind, col_ind] = 1

                # Re-expand if intercept included (keep last dimension fixed)
                if sn.include_intercept:
                    R_full = np.eye(sn.n_factors, dtype=float)
                    R_full[:-1, :-1] = P
                    R = R_full
                else:
                    R = P

                ot_matrices[seed] = R
                csv_path = output_dir / f"ot_matrix_seed_{seed}.csv"
                np.savetxt(csv_path, R, delimiter=",")

        else:
            for seed in sample_dict_summary.keys():
                if not sn.syn_data_rec_flag and seed == sn.ref_seed:
                    ot_matrices[seed] = permutation_matrix.T
                    continue
                Z_mean = Z_mean_dict[seed]
                if sn.syn_data_rec_flag:
                    Z_parquet_path = sn.input_dir / f"Z.parquet"
                    Z_ref = pd.read_parquet(Z_parquet_path).to_numpy()
                else:
                    Z_ref = permutation_matrix @ Z_mean_dict[sn.ref_seed]
                if sn.include_intercept:
                    Z_mean = Z_mean[:-1, :]
                    Z_ref = Z_ref[:-1, :]
                Z1 = Z_mean.copy()
                Z2 = Z_ref.copy()
                Z1 = Z1 - Z1.mean(axis=1, keepdims=True)
                Z2 = Z2 - Z2.mean(axis=1, keepdims=True)
                z1_std = Z1.std(axis=1, ddof=1, keepdims=True)
                z1_std[z1_std == 0] = 1
                z2_std = Z2.std(axis=1, ddof=1, keepdims=True)
                z2_std[z2_std == 0] = 1
                Z1 = Z1 / z1_std
                Z2 = Z2 / z2_std
                C = Z1 @ Z2.T
                U, _, Vt = np.linalg.svd(C, full_matrices=False)
                R = U @ Vt
                if sn.include_intercept:
                    R_full = np.eye(sn.n_factors)
                    R_full[:-1, :-1] = R
                    R = R_full
                ot_matrices[seed] = R

                csv_path = output_dir / f"ot_matrix_seed_{seed}.csv"
                np.savetxt(csv_path, R, delimiter=",")

        if "Z_cfm_op_scaling" in export_operator_outputs:
            # point estimate therefore constant over samples
            s_seed = {
                seed: sample_dict_summary[seed]
                .average(op=export_operator_outputs["Z_cfm_op_scaling"])
                .asnumpy()
                .ravel()
                for seed in sample_dict_summary
            }
        else:
            s_seed = {seed: np.ones(sn.n_factors) for seed in sample_dict_summary}

        for kk in export_operator_outputs.keys():
            if kk in {"Z_cfm_op", "Z_cfm_op_scaling"}:
                continue
            op_samples[kk] = {}
            for seed in sample_dict_summary.keys():
                R = ot_matrices[seed]
                RT = R.T
                op_samples[kk][seed] = []
                for sample in sample_dict_summary[seed].iterator(
                    export_operator_outputs[kk]
                ):
                    sample_val = sample.asnumpy()
                    if kk.startswith("W"):
                        sample_val = sample_val @ R
                    elif kk == "Z":
                        sample_val = RT @ sample_val
                    elif kk in {"Z_cfm", "Z_non_cfm"}:
                        original_shape = sample_val.shape
                        sample_val = sample_val.reshape(RT.shape[0], -1)
                        sample_val = RT @ sample_val  # Apply RT
                        sample_val = sample_val.reshape(original_shape)
                    elif kk == "Z_cfm_op_pspec":
                        smix = np.square(RT) * s_seed[seed][None, :]
                        sample_val = (smix @ sample_val) / smix.sum(axis=1, keepdims=True)

                    op_samples[kk][seed].append(sample_val)

                op_samples[kk][seed] = np.array(op_samples[kk][seed])

        log_counter = 0
        for seed in sample_dict_summary.keys():
            if seed == sn.ref_seed:
                continue

            current_Z_mean = Z_mean_dict[seed]
            used_indices = [False] * sn.n_factors

            for ref_idx in range(sn.n_factors):
                ref_factor_data = Z_mean_dict[sn.ref_seed][ref_idx, :]
                best_match_idx = -1
                max_abs_corr = -1
                best_corr_sign = 1

                for i in range(sn.n_factors):
                    if not used_indices[i]:
                        current_factor_data = current_Z_mean[i, :]
                        if (
                            np.std(ref_factor_data) < 1e-9
                            or np.std(current_factor_data) < 1e-9
                        ):
                            corr = 0
                        else:
                            corr = np.corrcoef(ref_factor_data, current_factor_data)[
                                0, 1
                            ]
                        if np.isnan(corr):
                            corr = 0
                        abs_corr = np.abs(corr)

                        if abs_corr > max_abs_corr:
                            max_abs_corr = abs_corr
                            max_corr = corr
                            best_match_idx = i
                            best_corr_sign = np.sign(corr)

                log_path = output_dir / "factor_matching_log.txt"
                if log_counter == 0:
                    header_line1 = f"#reference_seed: {sn.ref_seed}"
                    header_line2 = (
                        f"seed, ref_factor_id, max_abs_corr, best_match_id, corr_sign"
                    )
                    with open(log_path, "a") as f:
                        f.write(header_line1 + "\n")
                        f.write(header_line2 + "\n")
                    log_counter += 1
                log_line = (
                    f"{seed}, {ref_idx}, {max_corr}, {best_match_idx}, {best_corr_sign}"
                )
                with open(log_path, "a") as f:
                    f.write(log_line + "\n")

        op_samples_final = {}

        for kk in export_operator_outputs.keys():
            if kk in {"Z_cfm_op", "Z_cfm_op_scaling"}:
                continue
            op_samples_final[kk] = []
            for seed in sample_dict_summary.keys():
                for sample_val in op_samples[kk][seed]:
                    field_sample = ift.makeField(
                        export_operator_outputs[kk].target, sample_val
                    )
                    op_samples_final[kk].append(field_sample)
            op_samples_final[kk] = ift.SampleList(op_samples_final[kk])
        callback([op_samples_final, sample_dict_summary], iteration=sn.n_iterations - 1)

        for kk in export_operator_outputs.keys():
            if kk in {"Z_cfm_op", "Z_cfm_op_scaling"}:
                continue
            sample_dir = output_dir / kk
            sample_dir.mkdir(exist_ok=True)
            file_name = sample_dir / f"latest.hdf5"
            op_samples_final[kk].save_to_hdf5(
                file_name, op=None, samples=True, mean=True, std=True, overwrite=True
            )

    if master:
        print()
        if sn.summarize_runs_flag:
            print(f"Saving model for factors: {sn.n_factors}")
        else:
            print(
                f"Finished run for seed: {sn.seed}, factors: {sn.n_factors}"
            )
            print()
            print(
                f"Saving model for seed: {sn.seed}, factors: {sn.n_factors}"
            )

        # =======================================================
        # ======= Save MOFTy model in MOFA/MEFISTO format =======
        # =======================================================

        if sn.init_cfm:
            Z_latent_spaces = ["Z"]
            if sn.init_non_cfm:
                Z_latent_spaces.append("Z_cfm")
                Z_latent_spaces.append("Z_non_cfm")
        else:
            Z_latent_spaces = ["Z"]

        for Z_latent in Z_latent_spaces:
            if sn.init_cfm and sn.init_non_cfm:
                output_file_path = output_dir / f"mofty_model_{Z_latent}.hdf5"
                if output_file_path.exists():
                    output_file_path.unlink()
            else:
                output_file_path = output_dir / "mofty_model.hdf5"
                if output_file_path.exists():
                    output_file_path.unlink()

            with h5py.File(output_file_path, "w") as f_out:
                string_dt = h5py.string_dtype(encoding="utf-8")
                M_views = len(sn.data.keys())
                K_factors = sn.n_factors
                N_samples = n_data_obs

                # --- Data & Intercepts (adapted from user snippet) ---
                data_group_hdf5 = f_out.create_group("data")
                intercepts_group_hdf5 = f_out.create_group("intercepts")

                for (
                    kk_view_data,
                    vv_data_orig,
                ) in sn.data.items():  # Iterate over sn.data
                    vv_data_float = vv_data_orig.astype(float)

                    # MOFA2 expects data as (N_samples, D_features_view)
                    D_to_save_transposed = vv_data_float.T
                    data_view_group = data_group_hdf5.create_group(kk_view_data)
                    data_view_group.create_dataset(
                        sn.group_ident, data=D_to_save_transposed
                    )

                # --- Expectations (Means of W and Z - adapted from user snippet) ---
                exp_group = f_out.create_group("expectations")
                exp_W_group = exp_group.create_group("W")
                exp_Z_group = exp_group.create_group("Z")

                # --- Samples and Samples Metadata ---
                samples_group_hdf5 = f_out.create_group("samples")
                samples_meta_group_hdf5 = f_out.create_group("samples_metadata")
                encoded_obs_names_list = [s.encode("utf-8") for s in sn.obs_names]

                samples_group_hdf5.create_dataset(
                    sn.group_ident, data=encoded_obs_names_list, dtype=string_dt
                )
                group_meta_group = samples_meta_group_hdf5.create_group(sn.group_ident)
                group_meta_group.create_dataset(
                    "sample", data=encoded_obs_names_list, dtype=string_dt
                )
                group_meta_group.create_dataset(
                    "group",
                    data=[sn.group_ident.encode("utf-8")] * N_samples,
                    dtype=string_dt,
                )

                if len(sn.covariate_keys) > 0:
                    cov_keys_group = f_out.create_group("covariates")
                    cov_keys_group.create_dataset(
                        "covariates",
                        data=[key.encode("utf-8") for key in sn.covariate_keys],
                        dtype=string_dt,
                    )
                    cov_group = f_out.create_group("cov_samples_transformed")
                    cov_group.create_dataset(
                        sn.group_ident, data=sn.cov_samples_transformed
                    )
                    cov_raw_group = f_out.create_group("cov_samples")
                    cov_raw_group.create_dataset(sn.group_ident, data=sn.cov_samples)
                    # Save covariates to CSV files
                    cov_transformed_df = pd.DataFrame(
                        sn.cov_samples_transformed, columns=sn.covariate_keys
                    )
                    cov_transformed_parquet_path = output_dir / "covariates.parquet"
                    cov_transformed_df.to_parquet(cov_transformed_parquet_path, index=False)

                # --- Groups, Views, Model Options, Training Options, Training Stats ---
                f_out.create_group("groups").create_dataset(
                    "groups", data=[sn.group_ident.encode("utf-8")], dtype=string_dt
                )
                f_out.create_group("views").create_dataset(
                    "views",
                    data=[name.encode("utf-8") for name in sn.data.keys()],
                    dtype=string_dt,
                )

                sample_hdf5 = "latest.hdf5"
                Z_path = output_dir / Z_latent / sample_hdf5
                if Z_path.exists():
                    with h5py.File(Z_path, "r") as f_in:
                        if sn.n_final_samples > 0:
                            Z_mean = f_in["stats/mean"][()]
                            Z_std = f_in["stats/standard deviation"][()]
                        else:
                            Z_mean = f_in["samples/0"][()]
                            Z_std = f_in["samples/0"][()]
                        exp_Z_group.create_dataset(sn.group_ident, data=Z_mean)

                W_means = {}
                W_stds = {}

                for kk in sn.data.keys():
                    W_key = f"W_{kk}"
                    W_path = output_dir / W_key / sample_hdf5
                    if W_path.exists():
                        with h5py.File(W_path, "r") as f_in:
                            if sn.n_final_samples > 0:
                                W_mean = f_in["stats/mean"][()]
                                W_std = f_in["stats/standard deviation"][()]
                            else:
                                W_mean = f_in["samples/0"][()]
                                W_std = f_in["samples/0"][()]
                            W_means[kk] = W_mean
                            W_stds[kk] = W_std
                            n_features = W_mean.shape[0]
                            exp_W_group.create_dataset(
                                kk, data=W_mean.T
                            )  # Exclude intercept
                            intercept_data_for_view = [0] * n_features
                            # Intercepts are not stored and set to zero
                            # if sn.center_data is True this is correct since we only save the centered data
                            # if sn.center_data is False the intercepts are non-zero and we can update the mofa_object in the downstream analysis if needed.
                            # nan would be better but this can lead to issues in MOFA2
                            current_intercepts_group = (
                                intercepts_group_hdf5.create_group(kk)
                            )
                            current_intercepts_group.create_dataset(
                                sn.group_ident, data=intercept_data_for_view
                            )

                # --- Calculate and Store Variance Explained ---
                var_exp_group = f_out.create_group("variance_explained")

                # Initialize lists for storing various R2/variance explained metrics
                r2_total_gaussian_list = []
                r2_per_factor_gaussian_matrix = np.full((M_views, K_factors), np.nan)

                for m_idx, kk_view_data in enumerate(sn.data.keys()):
                    if sn.likelihood_options.get(
                        kk_view_data, ""
                    ).lower() == "gaussian":
                        D_view = sn.data.get(
                            kk_view_data
                        )  # centered if sn.center_data is True (default)
                        W_view = W_means.get(kk_view_data)  # D_m x K_factors
                        D_view = D_view.astype(float)
                        D_view_centered = D_view
                        tot_sum_sq_view = np.nansum(D_view_centered**2)
                        if tot_sum_sq_view > 1e-10:
                            # Total R2 for the view
                            D_recon_view = W_view @ Z_mean
                            sum_sq_total_view = np.nansum((D_view - D_recon_view) ** 2)
                            view_r2_total_gaussian = 1 - (
                                sum_sq_total_view / tot_sum_sq_view
                            )
                            r2_total_gaussian_list.append(view_r2_total_gaussian)
                            view_r2_per_factor_values = np.full(K_factors, np.nan)
                            for k_factor_idx in range(K_factors):
                                Wk_factor_view = W_view[:, k_factor_idx : k_factor_idx + 1]
                                Zk_factor_view = Z_mean[k_factor_idx : k_factor_idx + 1, :]
                                D_recon_k = Wk_factor_view @ Zk_factor_view
                                sum_sq_factor_k = np.nansum((D_view - D_recon_k) ** 2)
                                view_r2_per_factor_values[k_factor_idx] = 1 - (
                                    sum_sq_factor_k / tot_sum_sq_view
                                )
                            r2_per_factor_gaussian_matrix[m_idx, :] = (
                                view_r2_per_factor_values
                            )
                        else:
                            r2_total_gaussian_list.append(0)
                            r2_per_factor_gaussian_matrix[m_idx, :].fill(0)
                            print(
                                f"TSS near zero for Gaussian view '{kk_view_data}', R2 set to 0."
                            )
                    elif (
                        sn.likelihood_options.get(kk_view_data, "").lower()
                        == "bernoulli"
                    ):
                        D_view = sn.data.get(
                            kk_view_data
                        )  # D_m x N, original Bernoulli data (0s and 1s)
                        W_view = W_means.get(kk_view_data)  # D_m x K_factors
                        D_view = D_view.astype(
                            float
                        )  # Ensure it's float for calculations
                        # View-wise Tjur R-squared
                        pred_total = W_view @ Z_mean  # (Dm, K+1) @ (K+1, N) -> (Dm, N)
                        pred_prob_total = expit(pred_total)
                        mean_prob_success_total = np.nanmean(
                            pred_prob_total[D_view == 1]
                        )
                        mean_prob_failure_total = np.nanmean(
                            pred_prob_total[D_view == 0]
                        )
                        tjur_r2_view = np.nan
                        if not (
                            np.isnan(mean_prob_success_total)
                            or np.isnan(mean_prob_failure_total)
                        ):
                            tjur_r2_view = (
                                mean_prob_success_total - mean_prob_failure_total
                            )
                        r2_total_gaussian_list.append(tjur_r2_view)
                        view_tjur_r2_per_factor_values = np.full(K_factors, np.nan)
                        for k_factor_idx in range(K_factors):
                            Wk_factor_view = W_view[
                                :, k_factor_idx : k_factor_idx + 1
                            ]  # (Dm, 1)
                            Zk_factor_view = Z_mean[
                                k_factor_idx : k_factor_idx + 1, :
                            ]  # (1, N)
                            pred_factor_k = Wk_factor_view @ Zk_factor_view  # (Dm, N)
                            pred_prob_factor_k = expit(pred_factor_k)
                            mean_prob_success_k = np.nanmean(
                                pred_prob_factor_k[D_view == 1]
                            )
                            mean_prob_failure_k = np.nanmean(
                                pred_prob_factor_k[D_view == 0]
                            )
                            tjur_r2_factor_k = np.nan
                            if not (
                                np.isnan(mean_prob_success_k)
                                or np.isnan(mean_prob_failure_k)
                            ):
                                tjur_r2_factor_k = (
                                    mean_prob_success_k - mean_prob_failure_k
                                )
                            view_tjur_r2_per_factor_values[k_factor_idx] = (
                                tjur_r2_factor_k
                            )
                        r2_per_factor_gaussian_matrix[m_idx, :] = (
                            view_tjur_r2_per_factor_values
                        )
                    elif (
                        sn.likelihood_options.get(kk_view_data, "").lower() == "poisson"
                    ):
                        print(
                            "For Poisson data, variance explained calculation is not implemented."
                        )
                        r2_total_gaussian_list.append(np.nan)

                # --- Save Gaussian R2 metrics ---
                var_exp_group.create_dataset(
                    f"r2_total/{sn.group_ident}",
                    data=np.array(r2_total_gaussian_list, dtype=np.float64),
                )
                var_exp_group.create_dataset(
                    f"r2_per_factor/{sn.group_ident}",
                    data=r2_per_factor_gaussian_matrix,
                )

                # --- Features and Features Metadata ---
                features_group_hdf5 = f_out.create_group("features")
                features_meta_group_hdf5 = f_out.create_group("features_metadata")
                for kk_view_feat in sn.data.keys():
                    nfeatures_view = modalities[kk_view_feat]
                    view_feature_names_list = []
                    if (
                        kk_view_feat in sn.feature_names_dict
                        and sn.feature_names_dict[kk_view_feat] is not None
                        and len(sn.feature_names_dict[kk_view_feat]) == nfeatures_view
                    ):
                        view_feature_names_list = sn.feature_names_dict[kk_view_feat]
                    encoded_feature_names = [
                        f.encode("utf-8") for f in view_feature_names_list
                    ]
                    features_group_hdf5.create_dataset(
                        kk_view_feat, data=encoded_feature_names, dtype=string_dt
                    )
                    view_meta_group = features_meta_group_hdf5.create_group(
                        kk_view_feat
                    )
                    view_meta_group.create_dataset(
                        "feature", data=encoded_feature_names, dtype=string_dt
                    )
                    view_meta_group.create_dataset(
                        "view",
                        data=[kk_view_feat.encode("utf-8")] * nfeatures_view,
                        dtype=string_dt,
                    )

                model_opts_group = f_out.create_group("model_options")
                model_opts_group.create_dataset("ard_factors", data=np.nan)
                model_opts_group.create_dataset("ard_weights", data=np.nan)
                model_opts_group.create_dataset("spikeslab_factors", data=np.nan)
                model_opts_group.create_dataset("spikeslab_weights", data=np.nan)
                likelihoods_opts_list = []
                for kk_view_opts in sn.data.keys():
                    likelihoods_opts_list.append(sn.likelihood_options[kk_view_opts])
                model_opts_group.create_dataset(
                    "likelihoods",
                    data=[lik.encode("utf-8") for lik in likelihoods_opts_list],
                    dtype=string_dt,
                )

                f_out.create_dataset("training_opts", data=[np.nan])
                train_stats_group = f_out.create_group("training_stats")
                train_stats_group.create_dataset("elbo", data=[np.nan])
                train_stats_group.create_dataset("number_factors", data=[K_factors])
                train_stats_group.create_dataset("time", data=[np.nan])

            # --- Save W, D, noise and R2 matrices to CSV files ---
            # Prepare factor names
            factor_names = list(range(K_factors))
            # Get sample names, generating generic ones if needed
            # Z is (K, N)
            Z_matrix = Z_mean[factor_names, :]
            Z_std_matrix = Z_std[factor_names, :]
            # Transpose to (N, K) for DataFrame
            Z_df = pd.DataFrame(Z_matrix, index=factor_names, columns=sn.obs_names)
            Z_std_df = pd.DataFrame(Z_std_matrix, index=factor_names, columns=sn.obs_names)
            Z_parquet_path = output_dir / f"{Z_latent}.parquet"
            Z_df.to_parquet(Z_parquet_path, index=True)
            Z_std_parquet_path = output_dir / f"{Z_latent}_std.parquet"
            Z_std_df.to_parquet(Z_std_parquet_path, index=True)

            # Save W and D matrices for each view separately
            view_names = list(sn.data.keys())
            for m_idx, view in enumerate(view_names):
                # --- Save W matrix for the view ---
                if W_means and view in W_means:
                    W_matrix = W_means[view]  # Shape (D_m, K)
                    W_std_matrix = W_stds[view]
                    W_matrix_view = W_matrix[:, factor_names]
                    W_std_matrix_view = W_std_matrix[:, factor_names]
                    nfeatures_view = modalities[view]
                    view_feature_names = sn.feature_names_dict.get(view)
                    if not (
                        view_feature_names and len(view_feature_names) == nfeatures_view
                    ):
                        view_feature_names = [
                            f"feature_{j}_{view}" for j in range(nfeatures_view)
                        ]
                    # Create DataFrame with features as rows, factors as columns
                    W_df = pd.DataFrame(
                        W_matrix_view, index=view_feature_names, columns=factor_names
                    )
                    W_df.index.name = ""
                    W_std_df = pd.DataFrame(
                        W_std_matrix_view,
                        index=view_feature_names,
                        columns=factor_names,
                    )
                    W_std_df.index.name = ""
                    W_parquet_path = output_dir / f"W_{view}.parquet"
                    W_std_parquet_path = output_dir / f"W_{view}_std.parquet"
                    W_df.to_parquet(W_parquet_path, index=True)
                    W_std_df.to_parquet(W_std_parquet_path, index=True)

                # --- Save D matrix for the view ---
                nfeatures_view = modalities[view]
                view_feature_names = sn.feature_names_dict.get(view)
                # Create DataFrame with features as rows, samples as columns

                # --- Save noise matrix for the view (if Gaussian) ---
                if sn.likelihood_options.get(view) == "gaussian":
                    # Assuming noise precision (tau) is stored in a directory named after the view + _tau
                    noise_path = output_dir / f"std_{view}" / sample_hdf5
                    with h5py.File(noise_path, "r") as f_in:
                        if sn.n_final_samples > 0:
                            std = f_in["stats/mean"][()]
                        else:
                            std = f_in["samples/0"][()]
                        nfeatures_view = modalities[view]
                        view_feature_names = sn.feature_names_dict.get(view)
                        if not (
                            view_feature_names
                            and len(view_feature_names) == nfeatures_view
                        ):
                            view_feature_names = [
                                f"feature_{j}_{view}" for j in range(nfeatures_view)
                            ]
                        # Save noise vector with feature annotations
                        noise_vector = std
                        noise_df = pd.DataFrame(
                            noise_vector,
                            index=view_feature_names,
                            columns=["noise_std"],
                        )
                        noise_df.index.name = ""
                        noise_parquet_path = output_dir / f"noise_std_{view}.parquet"
                        noise_df.to_parquet(noise_parquet_path, index=True)

                # --- Save R2 statistics for the view ---
                r2_per_factor_view = r2_per_factor_gaussian_matrix[m_idx, :]
                total_r2_view = r2_total_gaussian_list[m_idx]

                r2_stats_df = pd.DataFrame(
                    {
                        "factor_index": factor_names,
                        "r2_per_factor": r2_per_factor_view,
                        "total_r2_view": total_r2_view,
                    }
                )
                if sn.init_cfm:
                    r2_csv_path = output_dir / f"{Z_latent}_r2_stats_{view}.csv"
                else:
                    r2_csv_path = output_dir / f"Z_r2_stats_{view}.csv"
                r2_stats_df.to_csv(r2_csv_path, index=False)

        if sn.syn_data_rec_flag and sn.summarize_runs_flag:
            print()
            print("Synthetic data recovery metrics")
            recovery_metrics = []
            matrices_to_check = ["Z"]
            if sn.init_cfm:
                matrices_to_check.extend(["Z_cfm", "Z_non_cfm"])

            # Create a directory for recovery plots
            plots_dir = output_dir / "recovery_plots"
            plots_dir.mkdir(exist_ok=True)

            # Helper function to calculate metrics for a given matrix
            def calculate_metrics(
                matrix_name, ref_path, est_path, n_factors, is_w_matrix=False
            ):
                if not ref_path.exists() or not est_path.exists():
                    print(f"Skipping {matrix_name}: files not found.")
                    return

                z_ref = pd.read_parquet(ref_path).to_numpy()
                z_est = pd.read_parquet(est_path).to_numpy()

                # W matrices have factors as columns, Z matrices have them as rows
                if is_w_matrix:
                    z_ref = z_ref.T
                    z_est = z_est.T

                # Ensure number of factors match before proceeding
                if z_ref.shape[0] != n_factors or z_est.shape[0] != n_factors:
                    print(
                        f"Warning: Factor number mismatch for {matrix_name}. Ref: {z_ref.shape[0]}, Est: {z_est.shape[0]}. Skipping."
                    )
                    return

                for k in range(n_factors):
                    z_ref_k = z_ref[k, :]
                    z_est_k = z_est[k, :]

                    # Handle cases with (near) zero variance
                    if np.std(z_ref_k) < 1e-9 or np.std(z_est_k) < 1e-9:
                        corr_k, p_val_k = np.nan, np.nan
                    else:
                        corr_k, p_val_k = pearsonr(z_ref_k, z_est_k)

                    if p_val_k < 1e-12:
                        pterm = "p<1e-12"
                    else:
                        pterm = f"p={p_val_k:.2e}"
                    # Generate and save scatter plot
                    fig, ax = plt.subplots(figsize=(5/2.54, 4.2/2.54))

                    ax.scatter(z_ref_k, z_est_k, edgecolors="none", s=1)

                    ax.set_xlabel("Ground Truth", fontsize=5, labelpad=3)
                    ax.set_ylabel("Reconstructed", fontsize=5, labelpad=3)
                    ax.tick_params(axis="both", which="major", labelsize=5, length=1.5, width=0.5, pad=3)
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.set_title(f"{matrix_name} - Factor {k+1}", fontsize=5, pad=3)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)
                    ax.axhline(0, color="grey", linestyle="--", alpha=0.6, linewidth=0.5)
                    ax.axvline(0, color="grey", linestyle="--", alpha=0.6, linewidth=0.5)

                    # Add correlation text
                    ax.text(
                        0.05,
                        0.95,
                        f"r={corr_k:.3f}\n{pterm}",
                        transform=ax.transAxes,
                        fontsize=5,
                        verticalalignment="top",
                    )

                    plt.tight_layout()
                    plot_path = plots_dir / f"recovery_{matrix_name}_factor_{k+1}.pdf"
                    plt.savefig(plot_path, bbox_inches="tight", transparent=True)
                    plt.close(fig)
                    # --- End of plotting ---

                    recovery_metrics.append(
                        {
                            "name": matrix_name,
                            "factor": k + 1,
                            "corr": corr_k,
                            "p_value": p_val_k,
                        }
                    )

            # Calculate for Z, Z_cfm, Z_non_cfm
            for matrix_name in matrices_to_check:
                ref_path = sn.input_dir / f"{matrix_name}.parquet"
                est_path = output_dir / f"{matrix_name}.parquet"
                calculate_metrics(matrix_name, ref_path, est_path, sn.n_factors)

            # Calculate for W matrices
            for kk in sn.data_keys:
                matrix_name = f"W_{kk}"
                ref_path = sn.input_dir / f"{matrix_name}.parquet"
                est_path = output_dir / f"{matrix_name}.parquet"
                calculate_metrics(
                    matrix_name, ref_path, est_path, sn.n_factors, is_w_matrix=True
                )

            # Save all metrics to a single CSV file
            if recovery_metrics:
                metrics_df = pd.DataFrame(recovery_metrics)
                metrics_csv_path = output_dir / "recovery_metrics.csv"
                metrics_df.to_csv(metrics_csv_path, index=False)


def main():
    parser = get_parser()
    args = parser.parse_args()
    only_summarize = args.only_summarize

    if args.run_task:
        # This is a worker process
        with open(args.run_task, "rb") as f:
            task_args = pickle.load(f)

        if args.profile_worker:
            print("--- PROFILING WORKER ---")

            profiler = cProfile.Profile()
            profiler.enable()

            mofty_main(task_args)

            profiler.disable()

            stats = pstats.Stats(profiler).strip_dirs()

            print("--- PROFILER RESULTS (top 20 by cumtime) ---")
            stats.sort_stats("cumtime").print_stats(20)

            print("--- PROFILER RESULTS (top 20 by tottime) ---")
            stats.sort_stats("tottime").print_stats(20)

            print("---------------------------------")
        else:
            mofty_main(task_args)
        return  # End worker process here

    # Load YAML configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if (
        config.get("INIT_CFM", False)
        and config.get("USE_COVARIATES", False)
        and args.grid_check
    ):
        padding = config.get("PADDING", 2)
        covariate_keys, cov_samples_transformed = utils.preprocessing(
            input_dir=config.get("INPUT_DIR", None),
            hdf5_file=config.get("HDF5_INPUT", None),
            group_ident=config.get("GROUP_IDENT", "group1"),
            use_covariates=True,
            grid_only=True,
        )

        pix, centered_cov_samples, rg_coords, RG_dom = utils.calculate_grid_parameters(
                cov_samples_transformed,
                covariate_keys,
                args.grid_check,
                padding,
                force_grid=args.force_grid,
            )
        print("Grid scaling factor:", args.grid_check)
        print(f"Grid padding factor in each coordinate direction: {padding}")
        print(f"RG pixels with padding in each coordinate direction: {pix}")
        rg_set = set(map(tuple, rg_coords))
        union_touched = set()
        r = 0.5
        for a in centered_cov_samples:
            a = np.asarray(a)
            lo = np.ceil(a - r).astype(int)
            hi = np.floor(a + r).astype(int)
            ranges = [range(lo[i], hi[i] + 1) for i in range(len(a))]
            candidates = product(*ranges)
            touched = [p for p in candidates if p in rg_set]
            union_touched.update(touched)

        print("Total RG pixels:", len(rg_coords))
        print("Total RG pixels covered/touched by sample covariates (union):", len(union_touched))
        print("Total centered sample covariates:", len(centered_cov_samples))
        print(f"Covered RG pixels (without padding) / total centered sample covariates: { len(union_touched) / len(centered_cov_samples):.4f}")

        return


    if not config.get("SYN_DATA", False):
        output_base_dir = Path(config.get("OUTPUT_BASE_DIR", None))
        output_base_dir.mkdir(parents=True, exist_ok=True)


    if config.get("FORCE", False) and output_base_dir.exists():
        print(f"Removing existing output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    n_factors_list = config.get("N_FACTORS_LIST", [])
    base_seeds = config.get("SEEDS", [])

    n_factors_list.sort(reverse=True)
    n_seeds = 0
    if len(n_factors_list) == 0:
        n_factors_init = config.get("N_FACTORS_INIT", 5)
        n_factors_final = config.get("N_FACTORS_FINAL", 20)
        n_factors_step = config.get("N_FACTORS_STEP", 1)
        n_factors_list = list(
            range(n_factors_init, n_factors_final + 1, n_factors_step)
        )
        n_factors_list.sort(reverse=True)
        n_seeds_per_step = config.get("N_SEEDS_PER_STEP", 1)
        n_seeds = len(n_factors_list) * n_seeds_per_step
        n_factors_list = [[n] * n_seeds_per_step for n in n_factors_list]
        n_factors_list = reduce(lambda x, y: x + y, n_factors_list)  # Flatten the list

    if n_seeds > 0:
        rng = np.random.default_rng(None)
        extra_seeds = rng.integers(0, 2**31 - 1, size=n_seeds).tolist()
    else:
        extra_seeds = []

    seeds = base_seeds + extra_seeds
    reps_per_factor = len(seeds) // len(n_factors_list)
    n_factors_list_ext = np.repeat(n_factors_list, reps_per_factor).tolist()


    ref_seed_dict = config.get("REF_SEED_DICT", {})
    if len(ref_seed_dict) > 0 and len(ref_seed_dict) != len(set(n_factors_list)):
        raise ValueError("REF_SEED_DICT length must match the number of unique factor counts in N_FACTORS_LIST.")
    elif len(ref_seed_dict) == 0:
        prev = None
        for seed, n_factors in zip(seeds, n_factors_list_ext):
            if prev == n_factors:
                continue
            ref_seed_dict[n_factors] = seed
            prev = n_factors

    syn_data_flag = config.get("SYN_DATA", False)
    input_dir = Path(config.get("INPUT_DIR", None))
    if syn_data_flag:
        print("Generating synthetic data.")
        print()
        input_dir_syn = Path(config.get("INPUT_DIR", None))
        if input_dir_syn.exists():
            print(f"Removing existing input directory: {input_dir_syn}")
            print()
            shutil.rmtree(input_dir_syn)
        input_dir_syn.mkdir(parents=True, exist_ok=True)
        data = {}
        data_keys = config.get("DATA_KEYS", ["view_1", "view_2", "view_3"])
        n_features = config.get("N_FEATURES", [300, 300, 300])
        use_covariates = config.get("USE_COVARIATES", False)
        if use_covariates:
            covariates_shape = config.get("COVARIATES_SHAPE", [30, 30])
            coord_ranges = [
                np.linspace(0, covariates_shape[i] - 1, s)
                for i, s in enumerate(covariates_shape)
            ]
            grid = np.meshgrid(*coord_ranges, indexing="ij")
            cov_samples_np = np.vstack([g.ravel() for g in grid]).T
            covariate_keys = [f"covariate{i+1}" for i in range(cov_samples_np.shape[1])]
            cov_samples_df = pd.DataFrame(cov_samples_np, columns=covariate_keys)
            cov_parquet_path = input_dir_syn / "covariates.parquet"
            cov_samples_df.to_parquet(cov_parquet_path, index=False)
            cov_samples = cov_samples_df.to_numpy()
            cov_samples_transformed = cov_samples_df.to_numpy()
            data_samples = cov_samples.shape[0]
        else:
            cov_samples = None
            covariate_keys = []
            cov_samples_transformed = None
            data_samples = config.get("DATA_SAMPLES", 200)

        rng = np.random.default_rng(config.get("SEED", 42))
        data_mask = {}
        for kk in data_keys:
            num_features_modality = n_features[data_keys.index(kk)]
            data[kk] = np.zeros((num_features_modality, data_samples))
            data_mask[kk] = np.ones((num_features_modality, data_samples))
            random_masking = config.get("MISSING_DATA_FRACTION", 0.2)
            if random_masking > 0:
                # Calculate the number of columns (samples) to mask for this modality
                n_cols_to_mask = int(data_samples * random_masking)
                if n_cols_to_mask > 0:
                    # Randomly select a distinct set of column indices for this modality
                    cols_to_mask = rng.choice(
                        data_samples, n_cols_to_mask, replace=False
                    )
                    # Set all values in the selected columns to NaN
                    data_mask[kk][:, cols_to_mask] = np.nan

        syn_args = {
            "syn_data_flag": syn_data_flag,
            "syn_data_rec_flag": False,
            "summarize_runs_flag": False,
            "n_factors": config.get("N_FACTORS", 5),
            "seed": config.get("SEED", 42),
            "ref_seed": config.get("SEED", 42),
            "only_ref_seed_init": False,
            "data": data,
            "covariate_keys": covariate_keys,
            "data_mask": data_mask,
            "cov_samples_transformed": cov_samples_transformed,
            "force_grid": args.force_grid,
            "grid_scaling": config.get("GRID_SCALING", 1),
            "include_intercept": config.get("INCLUDE_INTERCEPT", False),
            "input_dir": input_dir,
            "data_keys": config.get(
                "DATA_KEYS", ["view_1", "view_2", "view_3"]
            ),
            "likelihood_options": config.get(
                "LIKELIHOOD_OPTIONS", ["gaussian", "gaussian", "gaussian"]
            ),
            "data_samples": config.get("DATA_SAMPLES", 200),
            "init_cfm": config.get("INIT_CFM", False),
            "init_non_cfm": config.get("INIT_NON_CFM", True),
            "use_covariates": use_covariates,
            "target_snr": config.get("TARGET_SNR", 1),
            "fluctuations": config.get("FLUCTUATIONS", [3, 0.5]),
            "loglogavgslope": config.get("LOGLOGAVGSLOPE", [-5, 0.5]),
            "flexibility": config.get("FLEXIBILITY", [1, 0.5]),
            "asperity": config.get("ASPERITY", [1, 0.5]),
            "offset_mean": config.get("OFFSET_MEAN", 0),
            "offset_std": config.get("OFFSET_STD", [1, 0.5]),
            "padding": config.get("PADDING", 1),
            "init_cfm_scaling": config.get("INIT_CFM_SCALING", True),
            "cfm_scaling_values": config.get("CFM_SCALING_VALUES", []),
            "W_igamma_syn_scaling": config.get("W_IGAMMA_SYN_SCALING", 0.6),
            "syn_scaling": config.get("SYN_SCALING", 0.6),
            "noise_model": config.get("NOISE_MODEL", "icov_gamma"),
            "exp_Z": config.get("EXP_Z", False),
            "softplus_Z": config.get("SOFTPLUS_Z", False),
            "poisson_exp": config.get("POISSON_EXP", False),
            "poisson_softplus": config.get("POISSON_SOFTPLUS", False),
            "softplus_W": config.get("SOFTPLUS_W", False),
            "exp_W": config.get("EXP_W", False),
            "noise_axis": config.get("NOISE_AXIS", "row_wise"),
            "igamma_Z": config.get("IGAMMA_Z", False),
            "igamma_W": config.get("IGAMMA_W", True),
            "igamma_param": config.get("IGAMMA_PARAM", 0.1),
            "beta_param": config.get("BETA_PARAM", 1),
            "icov_gamma_param": config.get("ICOV_GAMMA_PARAM", 1),
            "flip_image": config.get("FLIP_IMAGE", False),
            "z_slices": config.get("Z_SLICES", 20),
            "plot_dot_size": config.get("PLOT_DOT_SIZE", 5),
        }
        mofty_main(syn_args)
        print("Synthetic data generation completed.")
        return

    config_master = copy.deepcopy(config)
    existing_configs = list(output_base_dir.glob("master_config*.yaml"))
    num_existing = len(existing_configs)
    if num_existing > 0:
        for cfg in existing_configs:
            cfg_data = yaml.safe_load(cfg.read_text())
            if seeds != cfg_data["SEEDS"]:
                if n_factors_list_ext != cfg_data["N_FACTORS_LIST"]:
                    raise ValueError(
                        f"Existing config with matching seeds and not matching factors found. Please resolve the conflict"
                    )
                output_config_path = (
                    output_base_dir / f"master_config_{num_existing + 1}.yaml"
                )
                if num_existing == 1 and (output_base_dir / "master_config.yaml").exists():
                    (output_base_dir / "master_config.yaml").rename(
                        output_base_dir / "master_config_1.yaml"
                    )
            else:
                output_config_path = output_base_dir / "master_config.yaml"
    else:
        output_config_path = output_base_dir / "master_config.yaml"

    with open(output_config_path, "w") as f:
        config_master["SEEDS"] = seeds
        config_master["REF_SEED_DICT"] = ref_seed_dict
        config_master["N_FACTORS_LIST"] = n_factors_list_ext
        config_master.pop("N_FACTORS_INIT", None)
        config_master.pop("N_FACTORS_FINAL", None)
        config_master.pop("N_FACTORS_STEP", None)
        config_master.pop("N_SEEDS_PER_STEP", None)
        yaml.safe_dump(config_master, f, sort_keys=False)

        print(f"Full run configuration saved to {output_config_path}")


    print()
    print("Running MOFTy")
    print()
    print(f"Run parameters (seed, n_factors): {list(zip(seeds, n_factors_list_ext))}")

    (
        data,
        covariate_keys,
        cov_samples,
        cov_samples_transformed,
        obs_names,
        feature_names_dict,
    ) = utils.preprocessing(
        input_dir=input_dir,
        data_keys=config.get("DATA_KEYS", []),
        use_covariates=config.get("USE_COVARIATES", False),
        likelihood_options=config.get("LIKELIHOOD_OPTIONS"),
        group_ident=config.get("GROUP_IDENT", "group1"),
        scale_views=config.get("SCALE_VIEWS", True),
        center_data=config.get("CENTER_DATA", True),
        hdf5_file=config.get("HDF5_INPUT", None)
    )

    for key, item in data.items():
        # Create DataFrame with features as rows, samples as columns
        feature_names = feature_names_dict.get(key)
        data_df = pd.DataFrame(item, index=feature_names, columns=obs_names)
        data_df.index.name = ""

    for kk in data.keys():
        plot_utils.standard_plotting(
            data[kk],
            kk,
            xlabel="Samples",
            ylabel="Features",
            output_dir=output_base_dir,
            master=True,
            stat_name="data_overview",
        )

    base_args = {
        "config": config,
        "output_base_dir": output_base_dir,
        "input_dir": input_dir,
        "syn_data_flag": False,
        "syn_data_rec_flag": config.get("SYN_DATA_REC", False),
        "summarize_runs_flag": False,
        "data": data,
        "covariate_keys": covariate_keys,
        "force_grid": args.force_grid,
        "noise_axis": config.get("NOISE_AXIS", "row_wise"),
        "cov_samples": cov_samples,
        "data_keys": config.get("DATA_KEYS", []),
        "only_ref_seed_init": config.get("ONLY_REF_SEED_INIT", False),
        "cov_samples_transformed": cov_samples_transformed,
        "obs_names": obs_names,
        "feature_names_dict": feature_names_dict,
        "resume_training": config.get("RESUME_TRAINING", False),
        "use_covariates": config.get("USE_COVARIATES", False),
        "likelihood_options": config.get("LIKELIHOOD_OPTIONS"),
        "group_ident": config.get("GROUP_IDENT", "group1"),
        "scale_views": config.get("SCALE_VIEWS", True),
        "center_data": config.get("CENTER_DATA", True),
        "include_intercept": config.get("INCLUDE_INTERCEPT", False),
        "init_cfm": config.get("INIT_CFM", False),
        "init_non_cfm": config.get("INIT_NON_CFM", True),
        "grid_scaling": config.get("GRID_SCALING", 1),
        "factor_init": config.get("FACTOR_INIT", "random"),  # or PCA or NMF
        "skip_init": config.get("SKIP_INIT", []),
        "noise_model": config.get("NOISE_MODEL", "icov_gamma"),  # or "variance"
        "fluctuations": config.get("FLUCTUATIONS", [2, 1]),
        "loglogavgslope": config.get("LOGLOGAVGSLOPE", [-2, 1]),
        "flexibility": config.get("FLEXIBILITY", [2, 1]),
        "asperity": config.get("ASPERITY", [2, 1]),
        "offset_mean": config.get("OFFSET_MEAN", 0),
        "offset_std": config.get("OFFSET_STD", [2, 1]),
        "padding": config.get("PADDING", 2),
        "init_cfm_scaling": config.get("INIT_CFM_SCALING", True),
        "cfm_scaling_init_values": config.get("CFM_SCALING_INIT_VALUES", "fixed"),
        "exp_Z_cfm": config.get("EXP_Z_CFM", False),
        "exp_Z": config.get("EXP_Z", False),
        "softplus_Z": config.get("SOFTPLUS_Z", False),
        "poisson_exp": config.get("POISSON_EXP", False),
        "poisson_softplus": config.get("POISSON_SOFTPLUS", False),
        "softplus_W": config.get("SOFTPLUS_W", False),
        "exp_W": config.get("EXP_W", False),
        "igamma_Z": config.get("IGAMMA_Z", False),
        "igamma_W": config.get("IGAMMA_W", True),
        "inference_method": config.get("INFERENCE_METHOD", "geoVI"),
        "igamma_param": config.get("IGAMMA_PARAM", 0.1),
        "beta_param": config.get("BETA_PARAM", 1),
        "beta_init": config.get("BETA_INIT", 0),
        "icov_gamma_param": config.get("ICOV_GAMMA_PARAM", 0.1),
        "noise_icov_initial": config.get("NOISE_ICOV_INITIAL", 1),
        "n_iterations": config.get("N_ITERATIONS", 30),
        "n_iterations_noise_constant": config.get("N_ITERATIONS_NOISE_CONSTANT", 5),
        "n_iterations_low_opt": config.get("N_ITERATIONS_LOW_OPT", 5),
        "n_start_samples": config.get("N_START_SAMPLES", 4),
        "n_final_samples": config.get("N_FINAL_SAMPLES", 4),
        "n_start_samples_iterations": config.get("N_START_SAMPLES_ITERATIONS", 5),
        "device": args.device,
        "flip_image": config.get("FLIP_IMAGE", False),
        "plot_dot_size": config.get("PLOT_DOT_SIZE", 5),
        "z_slices": config.get("Z_SLICES", 20),
    }

    if not only_summarize:
        tasks = [
            base_args
            | {"seed": seed, "n_factors": n_factors}
            | {"ref_seed": ref_seed_dict[n_factors]}
            for seed, n_factors in zip(seeds, n_factors_list_ext)
        ]

        n_tasks = len(tasks)
        rp = args.rp
        if n_tasks % rp != 0:
            raise ValueError(
                f"Number of tasks ({n_tasks}) must be divisible by number of processes ({rp})"
            )

        temp_task_dir = output_base_dir / "temp_tasks"
        temp_task_dir.mkdir(exist_ok=True)

        # Create a directory for log files
        log_dir = output_base_dir / "log_files"
        log_dir.mkdir(exist_ok=True)

        task_files = []
        for i, task in enumerate(tasks):
            task_file = temp_task_dir / f"task_{i}.pkl"
            with open(task_file, "wb") as f:
                pickle.dump(task, f)
            task_files.append(task_file)

        processes = []
        log_files = []
        for i in range(n_tasks):
            # Wait if we have reached the maximum number of parallel processes
            if len(processes) >= rp:
                # Wait for the oldest process to finish
                p, (stdout_f, stderr_f) = processes.pop(0)
                return_code = p.wait()
                stdout_f.close()
                stderr_f.close()
                if return_code != 0:
                    # Find the task name associated with the failed process
                    # This is a bit of a hack, we should ideally store the task name with the process
                    log_filename = os.path.basename(stdout_f.name)
                    task_name_from_log = log_filename.replace("_stdout.log", "")
                    raise RuntimeError(
                        f"Subprocess for {task_name_from_log} failed with exit code {return_code}. Check logs in {log_dir}."
                    )

            task_name = f"task_seed_{tasks[i]['seed']}_factors_{tasks[i]['n_factors']}"
            print()
            print(
                f"Starting subprocess for {task_name} ({i+1}/{n_tasks}). Log files are in {log_dir}."
            )

            # Base command for the worker task
            cmd_list = [sys.executable, __file__, "--run_task", str(task_files[i])]

            # Add profiling flag if requested
            if args.profile_worker:
                cmd_list.append("--profile_worker")

            if args.np:
                print("Running with MPI.")
                cmd = ["mpirun", "-np", str(args.np)] + cmd_list
            else:
                print("Running without MPI.")
                cmd = cmd_list

            stdout_log = open(log_dir / f"{task_name}_stdout.log", "w")
            stderr_log = open(log_dir / f"{task_name}_stderr.log", "w")
            log_files.append((stdout_log, stderr_log))

            sub_env = os.environ.copy()
            sub_env["OMP_NUM_THREADS"] = "1"

            p = subprocess.Popen(cmd, stdout=stdout_log, stderr=stderr_log, env=sub_env)
            processes.append((p, (stdout_log, stderr_log)))

        # Wait for all remaining processes to complete
        for p, (stdout_f, stderr_f) in processes:
            return_code = p.wait()
            stdout_f.close()
            stderr_f.close()
            if return_code != 0:
                log_filename = os.path.basename(stdout_f.name)
                task_name_from_log = log_filename.replace("_stdout.log", "")
                raise RuntimeError(
                    f"Subprocess for {task_name_from_log} failed with exit code {return_code}. Check logs in {log_dir}."
                )

        # Clean up temporary task files
        for task_file in task_files:
            task_file.unlink()
        temp_task_dir.rmdir()

        print()
        print("All runs finished.")

    master_factors_comb = []
    master_run_task_comb = []
    ref_seed_dict = {}
    for master_path in output_base_dir.glob("master*.yaml"):
        with open(master_path, "r") as f:
            master_config = yaml.safe_load(f)
        seeds = master_config.get("SEEDS", [])
        factors = master_config.get("N_FACTORS_LIST", [])
        ref_seed_dict |= master_config.get("REF_SEED_DICT", {})
        if len(seeds) == 0 or len(factors) == 0:
            raise ValueError(f"No seeds or no factors provided in {master_path.name}.")
        master_factors_comb.extend(factors)
        master_run_task_comb.extend(list(zip(seeds, factors)))

    with open(output_base_dir / "master_run_tasks.txt", "w") as f:
        f.write("Ref seed dict:\n")
        f.write(f"{ref_seed_dict}\n")

    if len(master_run_task_comb) / len(set(master_factors_comb)) == 1:
        print()
        print(
            "Found only one run task per factor in the master configurations. Skipping summarization."
        )
        return
    if len(master_run_task_comb) % len(set(master_factors_comb)) != 0:
        raise ValueError("Number of total run tasks is not a multiple of the number of distinct factors.")

    master_factors_comb = sorted(list(set(master_factors_comb)))
    for factor in master_factors_comb:
        sample_dict_summary = {}
        for task in master_run_task_comb:
            if task[1] != factor:
                continue
            else:
                sample_list = ift.ResidualSampleList.load(
                    output_base_dir
                    / f"run_seed_{task[0]}_factors_{task[1]}"
                    / "pickle"
                    / "latest"
                )
                sample_list = ift.SampleList(list(sample_list.iterator()))
                sample_dict_summary[task[0]] = sample_list

        mofty_args = (
            base_args
            | {"n_factors": factor}
            | {"summarize_runs_flag": True}
            | {"sample_dict_summary": sample_dict_summary}
            | {"ref_seed": ref_seed_dict[factor]}
        )
        mofty_main(mofty_args)

    print()
    print("All computations complete.")


if __name__ == "__main__":
    main()
