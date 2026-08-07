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
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inducing", type=int, default=1000)
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--n_factors", type=int, default=4)
    parser.add_argument("--convergence_mode", type=str, default="slow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparseGP", action="store_true")
    parser.add_argument("--data_dir", type=str, default="../input/input_gbm")
    parser.add_argument("--mefisto_output_dir", type=str, default="../output/output_gbm_mefisto")
    parser.add_argument("--gpu_mode", action="store_true")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--mkl_threads", type=int, default=1)
    return parser


def main():
    args = get_parser().parse_args()
    try:
        os.environ["MKL_NUM_THREADS"] = str(args.mkl_threads)
        print(f"Setting MKL_NUM_THREADS to {os.environ['MKL_NUM_THREADS']}")
    except:
        print("Could not set MKL_NUM_THREADS, using default")

    import muon as mu

    sparseGP = args.sparseGP
    seed = args.seed
    n_inducing = args.n_inducing
    n_iterations = args.n_iterations
    gpu_mode = args.gpu_mode
    gpu_device = args.gpu_device
    convergence_mode = args.convergence_mode
    n_factors = args.n_factors
    data_dir = Path(args.data_dir)
    mefisto_output_dir = Path(args.mefisto_output_dir)
    mefisto_output_dir.mkdir(exist_ok=True)
    if sparseGP:
        mefisto_output_file = mefisto_output_dir / f"mefisto_model_trained_{n_inducing}.hdf5"
    else:
        mefisto_output_file = mefisto_output_dir / f"mefisto_model_trained_fullGP.hdf5"
    Path(mefisto_output_file).unlink(missing_ok=True)
    processed_mdata_file = data_dir / "processed_mdata.h5mu"
    mdata = mu.read_h5mu(processed_mdata_file)
    n_obs = mdata.n_obs

    print(f"Number of observations: {n_obs}")
    print(f"Number of factors: {n_factors}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Convergence mode: {convergence_mode}")
    print(f"GPU mode: {gpu_mode}")

    if gpu_mode:
        print(f"GPU device: {gpu_device}")

    if sparseGP:
        print("Using sparse GP")
        print(f"Number of inducing points: {n_inducing}")
        print(f"Fraction of inducing points: {n_inducing/n_obs:.4f}")
        smooth_kwargs = {"sparseGP": True, "frac_inducing": n_inducing/n_obs}
    else:
        print("Using full GP")
        smooth_kwargs = {"sparseGP": False}


    mu.tl.mofa(
        mdata,
        center_groups=True,
        scale_views=True,
        n_factors=n_factors,
        convergence_mode=convergence_mode,
        gpu_mode=gpu_mode,
        gpu_device=gpu_device,
        likelihoods=["gaussian", "gaussian"],
        use_var=None,
        ard_factors=False,
        outfile=mefisto_output_file,
        smooth_kwargs=smooth_kwargs,
        smooth_covariate=["spatial1", "spatial2"],
        n_iterations= n_iterations,
        seed=seed
    )

if __name__ == "__main__":
    main()
