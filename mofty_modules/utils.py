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

import itertools
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
from scipy.stats import norm, beta
from sklearn.decomposition import PCA, NMF
from sklearn.impute import SimpleImputer
import nifty.cl as ift


def _resolve_table_path(input_dir, name):
    input_dir = Path(input_dir)

    candidates = [
        input_dir / f"{name}.parquet",
        input_dir / f"{name}.csv",
        input_dir / f"{name}.csv.gz",
        input_dir / f"{name}.csv.xz",
        input_dir / f"{name}.csv.bz2",
        input_dir / f"{name}.csv.zip",
        input_dir / f"{name}.csv.zst",
    ]

    matches = [path for path in candidates if path.exists()]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {name} as parquet or CSV in {input_dir}"
        )

    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple matching files found for {name}: {matches}"
        )

    return matches[0]


def _read_table(input_dir, name, **kwargs):
    path = _resolve_table_path(input_dir, name)

    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, **kwargs)
        except OSError as err:
            err_msg = str(err)
            if (
                "Couldn't deserialize thrift" not in err_msg
                or "Exceeded size limit" not in err_msg
            ):
                raise

            # Large parquet metadata can exceed PyArrow's default thrift limits.
            # Retry with larger limits before failing.
            thrift_limit = (2**31) - 1
            retry_kwargs = dict(kwargs)
            retry_kwargs.setdefault("engine", "pyarrow")
            retry_kwargs.setdefault("thrift_string_size_limit", thrift_limit)
            retry_kwargs.setdefault("thrift_container_size_limit", thrift_limit)

            print(
                "Parquet metadata exceeded default thrift size limits; retrying with larger limits."
            )
            return pd.read_parquet(path, **retry_kwargs)

    return pd.read_csv(path, **kwargs)


def preprocessing(
    input_dir,
    *,
    data_keys=None,
    likelihood_options=None,
    use_covariates=False,
    group_ident="group1",
    scale_views=True,
    center_data=True,
    hdf5_file=None,
    grid_only=False,
):
    print()
    print("Preprocessing")

    if not grid_only and not data_keys:
        raise ValueError("No data keys specified in config.")
    if not grid_only and not likelihood_options:
        raise ValueError("No likelihood options specified in config.")

    feature_names_dict = {}

    if hdf5_file:
        print()
        print("Loading data from MOFA/MEFISTO compatible HDF5 file.")

        model_file = Path(input_dir) / hdf5_file
        with h5py.File(model_file, "r") as f:
            if use_covariates:
                covariate_keys = f["covariates/covariates"][()].astype(str).tolist()
                cov_samples = f[f"cov_samples/{group_ident}"][()]
                print(f"Loaded sample covariates with shape {cov_samples.shape}")
                cov_samples_transformed = f[f"cov_samples_transformed/{group_ident}"][
                    ()
                ]
                if cov_samples.ndim == 1:
                    cov_samples = cov_samples.reshape(-1, 1)
                    cov_samples_transformed = cov_samples_transformed.reshape(-1, 1)
            else:
                covariate_keys = []
                cov_samples = []
                cov_samples_transformed = []
            if grid_only:
                return covariate_keys, cov_samples_transformed

            data = {}

            for view_name in data_keys:
                data[view_name] = f[f"data/{view_name}/{group_ident}"][()].T
                feature_names = [
                    feat.decode("utf-8") for feat in f[f"features/{view_name}"][()]
                ]
                feature_names_dict[view_name] = feature_names
                print(
                    f"- Loaded data for view '{view_name}' with shape {data[view_name].shape}"
                )
            sample_names = [s.decode("utf-8") for s in f[f"samples/{group_ident}"][()]]

    else:
        if use_covariates:
            cov_samples_df = _read_table(input_dir, "covariates")
            covariate_keys = cov_samples_df.columns.tolist()
            cov_samples = cov_samples_df.to_numpy()
            cov_samples_transformed = cov_samples_df.to_numpy()

        else:
            covariate_keys = []
            cov_samples = []
            cov_samples_transformed = []

        if grid_only:
            return covariate_keys, cov_samples_transformed

        if not likelihood_options:
            raise ValueError("No likelihood options specified in config.")

        sample_names = None
        data = {}
        for kk in data_keys:
            df = _read_table(input_dir, kk)
            current_feature_names = df.index.astype(str).tolist()
            current_sample_names = df.columns.astype(str).tolist()
            feature_names_dict[kk] = current_feature_names
            if sample_names is None:
                sample_names = current_sample_names
            elif sample_names != current_sample_names:
                print(
                    f"Warning: Sample names differ between '{data_keys[0]}' and '{kk}'. Check input files."
                )
            data[kk] = df.to_numpy()

    for kk, vv in data.items():
        if likelihood_options[kk] == "gaussian":
            if center_data:
                vv -= np.nanmean(vv, axis=1)[..., None]
            if scale_views:
                vv /= np.nanstd(vv)

    # Safe guard against potential issues with data types in covariate samples
    cov_samples = np.asarray(cov_samples, dtype=np.float64)
    cov_samples_transformed = np.asarray(cov_samples_transformed, dtype=np.float64)

    print()
    print("Data loading and preprocessing complete.")

    return (
        data,
        covariate_keys,
        cov_samples,
        cov_samples_transformed,
        sample_names,
        feature_names_dict,
    )


def beta_to_latent(val, a, b, eps=1e-12):
    val = np.clip(val, eps, 1 - eps)
    u = beta.cdf(val, a, b)
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)


def initialization_setup(
    initial_position,
    *,
    data_keys,
    data,
    ref_seed,
    seed,
    n_factors,
    x_gamma,
    x_beta,
    only_ref_seed_init,
    factor_init,
    skip_init,
    center_data,
    scale_views,
    cfm_scaling_init_values,
    master,
):
    for kk, vv in initial_position.items():
        if "icov_gamma" in kk:
            initial_position[kk] = ift.full(vv.domain, x_gamma)
        if kk == "Z_cfm_op_scaling" and cfm_scaling_init_values.lower() == "fixed":
            initial_position[kk] = ift.full(vv.domain, x_beta)
        if only_ref_seed_init and seed != ref_seed:
            continue
        if factor_init.lower() != "random":
            imputer = SimpleImputer(strategy="mean")
            X = []
            for dd in data_keys:
                if dd in skip_init:
                    continue
                else:
                    X.append(data[dd])
            X = np.vstack(X).astype(float)
            X = imputer.fit_transform(X.T).T
            if center_data:
                X_mean = np.mean(X, axis=1)[..., None]
                if np.isnan(X_mean).any():
                    raise ValueError("NaN encountered in mean during initialization.")
                X -= X_mean
            if scale_views:
                X_std = np.std(X)
                if np.isnan(X_std) or X_std == 0:
                    raise ValueError(
                        "NaN or zero encountered in std during initialization."
                    )
                X /= X_std

        if kk in ["Z", "Z_non_cfm"]:
            if factor_init.lower() == "pca":
                if master:
                    print(f"Initializing {kk} with PCA")
                pca = PCA(n_components=n_factors, random_state=seed)
                Z_init = pca.fit_transform(X.T).T
                initial_position[kk] = ift.makeField(vv.domain, Z_init)
            elif factor_init.lower() == "nmf":
                if master:
                    print(f"Initializing {kk} with NMF.")
                X[X < 0] = 0
                nmf = NMF(n_components=n_factors, random_state=seed, max_iter=1000)
                Z_init = nmf.fit_transform(X.T).T
                initial_position[kk] = ift.makeField(vv.domain, Z_init)

    return initial_position


def calculate_grid_parameters(
    covariates, covariate_keys, grid_scaling, padding, force_grid= False,
):
    """
    Calculates grid parameters for the correlated field model.

    Returns:
        tuple: A tuple containing pix, centered_cov_samples, rg_coords, and RG_dom.
    """
    mins = np.min(covariates, axis=0)
    maxs = np.max(covariates, axis=0)

    # Find the smallest spacing in any coordinate direction
    min_spacing = np.min(
        [
            np.min(np.diff(np.unique(covariates[:, i])))
            for i in range(len(covariate_keys))
        ]
    )

    # Use the same pixel size (dx) for all directions
    dx = np.full(len(covariate_keys), min_spacing) * grid_scaling
    L = maxs - mins + dx
    N = np.ceil(L / dx).astype(int)
    pix = np.ceil(padding * N).astype(int)

    if not force_grid and np.prod(pix) > 1_000_000:
       raise RuntimeError("Estimated grid points exceed 1_000_000. Use --force_grid to override this check and create the grid anyway.")

    padding_offset = (pix - N) / 2
    centered_covariates = covariates - mins
    for i in range(len(covariate_keys)):
        centered_covariates[:, i] = (
            centered_covariates[:, i] / dx[i]
        ) + padding_offset[i]

    ranges = [range(int(p)) for p in pix]
    rg_coords = np.array(list(itertools.product(*ranges)))

    RG_dom = ift.RGSpace(
        shape=tuple(pix.astype(int)), distances=[1] * len(covariate_keys)
    )

    return pix, centered_covariates, rg_coords, RG_dom
