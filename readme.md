[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## MOFTy

### Citing

If you use MOFTy in your research, please cite:

```
Neumann M, Arras P, Kaster AK, and Ott A. 
MOFTy: Multimodal Gaussian Process Factor Analysis with Numerical Information Field Theory. 
Preprint at bioRxiv (2026). DOI: https://doi.org/10.64898/2026.08.12.744240
```

---

### Content

- [Model description](#model-description)
- [Setup](#setup)
- [Model configuration](#model-configuration)
- [Output format](#output-format)
- [Integration with MOFA2, muon, and scverse](#integration-with-mofa2-muon-and-scverse)
- [Reproducing the MOFTy paper results](#reproducing-the-mofty-paper-results)
- [References](#references)


## Model description

MOFTy is a multimodal Bayesian factor analysis framework that jointly infers shared latent factors across multiple data modalities, while optionally incorporating continuous covariates such as time or spatial coordinates. It supports joint modeling with Gaussian, Bernoulli, and Poisson likelihoods, typically used for real-valued, binary, and count data, respectively, and naturally accommodates missing observations without prior imputation. Suppose we are given <i>M</i> data matrices (views) <i>Y</i><sup>(1)</sup>, ..., <i>Y</i><sup>(M)</sup>, where each matrix <i>Y</i><sup>(m)</sup> contains <i>D</i><sub>m</sub> features observed across <i>N</i> samples that are shared across views. For a given number of latent factors <i>L</i>, MOFTy represents each view through a low-rank factorization <i>W</i><sup>(m)</sup><i>Z</i>, where <i>Z</i> is the shared <i>L</i> &times; <i>N</i> latent factor matrix and <i>W</i><sup>(m)</sup> is the view-specific <i>D</i><sub>m</sub> &times; <i>L</i> weight matrix. For Gaussian likelihoods, the model includes an additive independent Gaussian noise term <i>&epsilon;</i><sup>(m)</sup> with feature-specific variance parameters learned during training:

<p align="center">
<i>Y</i><sup>(m)</sup> = <i>W</i><sup>(m)</sup><i>Z</i> + <i>&epsilon;</i><sup>(m)</sup>.
</p>

More generally, MOFTy models the signal in each view through a low-rank linear predictor which parameterizes the likelihood for feature <i>d</i> and sample <i>n</i> in view <i>m</i>. For Poisson likelihoods, the linear predictor is mapped to a positive rate parameter using an exponential link function. Alternatively, when non-negativity constraints are imposed on both factors and weights, an identity link can be used. For Bernoulli likelihoods, the linear predictor is mapped to a success probability using a logistic link function.

If samples are associated with continuous covariates, e.g. time or spatial coordinates, MOFTy decomposes the latent factors into

<p align="center">
<i>Z</i> = <i>Z</i><sub>CFM</sub> + <i>Z</i><sub>non-CFM</sub>.
</p>

Here, <i>Z</i><sub>CFM</sub> denotes the correlated field model (CFM) component, which captures variation along continuous covariates such as time or space. It is modeled using the correlated field model implemented in the numerical information field theory (NIFTy) library (Arras et al., 2021, 2022; Edenhofer et al., 2024; Steininger et al., 2019; Arras et al., 2019). The CFM assumes a homogeneous and isotropic correlation structure. Homogeneity implies, via the Wiener–Khinchin theorem, that the covariance operator is diagonal in Fourier space and can be represented by its power spectrum. Isotropy further restricts this spectrum to depend only on the magnitude of the Fourier mode. NIFTy models the power spectrum nonparametrically, allowing the correlation structure to be inferred from the data. This Fourier-space formulation also enables memory-efficient inference, with memory requirements scaling linearly with the number of grid points.

<i>Z</i><sub>non-CFM</sub> denotes the non-CFM component, which captures variation not explained by the CFM component, such as rapid fluctuations in one-dimensional time series or fine-scale variations on a smooth two-dimensional surface.

This explicit additive decomposition of each latent factor, together with the nonparametric correlated field model and uncertainty estimates for the individual components as well as the factor-specific correlation structures, distinguishes MOFTy from related frameworks such as MEFISTO (Velten et al., 2022) and MOFA-FLEX (Qoku et al., 2025).

Inference is performed using variational methods implemented in NIFTy:
- geoVI (geometric variational inference; Frank et al., 2021)
- MGVI (metric Gaussian variational inference; Knollmüller and Enßlin, 2020) — optional predecessor method

MOFTy reports total and factor-wise variance decompositions for each view, computed separately for the combined, CFM, and non-CFM components. For Gaussian likelihoods, this corresponds to the standard $R^2$. For Bernoulli likelihoods, we implemented Tjur's $R^2$ (Tjur, 2009).

### Features of MOFTy
- Multimodal factor analysis with Gaussian / Poisson / Bernoulli likelihoods
- Under Gaussian likelihoods, feature-specific noise levels in each view are inferred during training
- Configurable heavy-tailed shrinkage priors for weights and/or factors
- Modeling along continuous covariates (e.g., time or spatial coordinates) with the nonparametric correlated field model (CFM) in NIFTy
- Separation of each latent factor into a CFM and a non-CFM component
- Uncertainty estimates for the components, the factor-specific correlation structures, and the factor weights
- Optional non-negativity constraints
- Support of metric Gaussian variational inference (MGVI) and geometrical variational inference (geoVI)
- Parallelization via Message Passing Interface (MPI, https://www.mpi-forum.org/docs/)
- GPU acceleration via CuPy (Okuta et al., 2017)
- Compatibility with MOFA2 (R framework; mofax, Python) and integration with scverse ecosystem via muon for preprocessing and downstream analysis (Argelaguet et al., 2018; Argelaguet et al., 2020; Velten et al., 2022; Bredikhin et al., 2022; Virshup et al., 2023)

### Current limitations of MOFTy
- Only single-group configurations are supported.
- Component separation is unavailable when non-negativity constraints are enabled.
- Bernoulli likelihoods are incompatible with non-negative factorizations under the current logistic-link implementation.

---


## Setup

We recommend installing dependencies using micromamba:

https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```bash
git clone https://github.com/neumann-mn/mofty.git
cd mofty
```

To set up the Conda environment, run:
```bash
micromamba env create -f mofty.yaml --name mofty

# MOFTy with GPU support via CuPy (optional)
micromamba env create -f mofty_cupy.yaml --name mofty_cupy
```

Instructions for setting up optional preprocessing and downstream analysis packages are provided in [Integration with MOFA2, muon, and scverse](#integration-with-mofa2-muon-and-scverse).

### Running MOFTy

Basic usage:

```bash
python3 run_mofty.py --config <file_name>.yaml
```

Details for model configuration files and input formats are provided below. As a first example, we recommend starting with the synthetic datasets; see [Reproducing the MOFTy paper results](#reproducing-the-mofty-paper-results).


### Optional CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rp` | `int` | `1` | Number of run seeds executed in parallel |
| `--np` | `int` | `1` | Number of MPI workers per run |
|  |  |  | Total workers = `np × rp` |
| `--grid_check` | `store_true` | `false` | Check grid size and coverage for correlated field initialization |
| `--force_grid` | `store_true` | `false` | Force grid creation if estimated grid exceeds 1 megapixel |
| `--device` | `str` | `CPU` | Device type (`CPU` or `GPU`) |
| `--only_summarize` | `store_true` | `false` | Only summarize runs |
| `--profile_worker` | `store_true` | `false` | Enable worker profiling via `cProfile` |


## Model configuration

### Example configuration file

See `config_files/config_gbm_cfm_4.yaml`:
```yaml
INPUT_DIR: input/input_gbm
OUTPUT_BASE_DIR: output/output_gbm_cfm_4
HDF5_INPUT: mefisto_model_untrained.hdf5
DATA_KEYS:
- gene_exp
- protein
LIKELIHOOD_OPTIONS:
  gene_exp: gaussian
  protein: gaussian
USE_COVARIATES: true
INIT_CFM: true
GRID_SCALING: 260
FACTOR_INIT: PCA
N_FACTORS_INIT: 4
N_FACTORS_FINAL: 4
N_FACTORS_STEP: 1
N_SEEDS_PER_STEP: 4
N_ITERATIONS: 30
RESUME_TRAINING: true
FLIP_IMAGE: true
PLOT_DOT_SIZE: 1
```


### Input and output configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `INPUT_DIR` | `str` | - | Input data directory (required) |
| `OUTPUT_BASE_DIR` | `str` | - | Output directory (required) |
| `GROUP_IDENT` | `str` | `group1` | Group identifier |
| `HDF5_INPUT` | `str`| - | Untrained MOFA or MEFISTO model file (optional) |
| `DATA_KEYS` | `list[str]` | - | Keys for each view (required) |
| `LIKELIHOOD_OPTIONS` | `dict` | - | Likelihood options (`gaussian`, `bernoulli`, `poisson`) for each data key (required) |
| `FORCE` | `bool` | `false` | Remove previous output if `true`|

If an untrained MOFA or MEFISTO model file is used: typically, the MOFA2 R package uses `group1` as the group identifier, while MEFISTO-based workflows may use `single_group`.
When preprocessing with mofax or muon, the identifier is usually `group1`.
This is a technical remark in this context, since MOFTy currently supports only single-group models.

If `HDF5_INPUT` is not provided, MOFTy expects one CSV/parquet file per data key, named `<key>.csv` (or `<key>.parquet`). Format requirements:

- Rows correspond to features.
- Columns correspond to samples.
- The first column stores feature names.
- The header row stores sample IDs.
- We recommend to make sure that feature names are unique within and across data modalities/views.

Example:

```csv
,sample_0,sample_1,sample_2,...
feature_0,0.12,1.03,0.44,...
feature_1,2.31,0.00,4.20,...
feature_2,1.10,3.54,0.87,...
...
```


### Data preprocessing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SCALE_VIEWS` | `bool` | `true` | Scale views |
| `CENTER_DATA` | `bool` | `true` | Center input matrices |

If non-negativity constraints are used, set `CENTER_DATA` and `SCALE_VIEWS` to `false`.


### Likelihood transformations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `POISSON_EXP` | `bool` | `false` | Exponential link for Poisson likelihood |
| `POISSON_SOFTPLUS` | `bool` | `false` | Softplus link for Poisson likelihood |


### Heavy-tailed shrinkage priors

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IGAMMA_W` | `bool` | `true` | Use inverse-gamma-transformed scale variables in the weight matrix prior |
| `IGAMMA_Z` | `bool` | `false` | Use inverse-gamma-transformed scale variables in the factor matrix prior |
| `IGAMMA_PARAM` | `float` | `0.1` | Inverse-gamma rate parameter; shape is set to `IGAMMA_PARAM + 1` |

### Non-negativity and intercept constraints
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EXP_Z` | `bool` | `false` | Apply an exponential transform to factors |
| `SOFTPLUS_Z` | `bool` | `false` | Apply a softplus transform to factors |
| `EXP_W` | `bool` | `false` | Apply an exponential transform to weights |
| `SOFTPLUS_W` | `bool` | `false` | Apply a softplus transform to weights |
| `INCLUDE_INTERCEPT` | `bool` | `false` | Include a constant factor fixed to one; recommended for Bernoulli likelihoods |


### Continuous covariates

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_COVARIATES` | `bool` | `false` | Enable covariate modeling |
| `GRID_SCALING` | `int` | `1` | Spatial grid scaling |

If `HDF5_INPUT` is not provided, MOFTy expects a `covariates.csv` (or `covariates.parquet`) file.

Format requirements:

- Rows correspond to samples.
- Columns correspond to covariates.
- Row order must match the sample column order in each `<key>.csv` (or `<key>.parquet`) input file  
  (i.e., sample column 1 ↔ row 1, sample column 2 ↔ row 2, etc.).

Example:

```csv
covariate_1,covariate_2
0.01,2.30
1.20,0.30
1.10,3.54
...
```

The following correlated-field model (CFM) hyperparameters can be adjusted flexibly and are used to initialize a separate correlated field for each factor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FLUCTUATIONS` | `list[float]` | <code>[2,&nbsp;1]</code> | Prior mean and standard deviation for the overall fluctuation amplitude of the correlated field. |
| `LOGLOGAVGSLOPE` | `list[float]` | <code>[-2,&nbsp;1]</code> | Prior mean and standard deviation for the average slope of the log power spectrum on a log-log scale. |
| `FLEXIBILITY` | `list[float]` | <code>[2,&nbsp;1]</code> | Prior mean and standard deviation controlling deviations from a simple power-law spectrum. |
| `ASPERITY` | `list[float]` | <code>[2,&nbsp;1]</code> | Prior mean and standard deviation controlling small-scale roughness of the log power spectrum. |
| `OFFSET_MEAN` | `float` | `0` | Prior mean of the zero-mode, i.e. the field offset. |
| `OFFSET_STD` | `list[float]` | <code>[2,&nbsp;1]</code> | Prior mean and standard deviation for the zero-mode uncertainty. |


Additional CFM-related configuration options control grid construction, initialization, and mixing with the non-CFM component.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PADDING` | `int` | `2` | Padding factor used for the harmonic grid. |
| `INIT_NON_CFM` | `bool` | `true` | Whether to initialize the non-CFM factor component. |
| `INIT_CFM` | `bool` | `false` | Whether to initialize the CFM factor component. |
| `INIT_CFM_SCALING` | `bool` | `true` | Whether to initialize the factor-wise CFM scaling parameters. |
| `CFM_SCALING_INIT_VALUES` | `str` | `fixed` | Initialization strategy for the CFM scaling parameters; supported values include `fixed` (`BETA_PARAM`) and `random`. |
| `BETA_INIT` | `float` | `0` | Initial value for the CFM mixing parameter |
| `BETA_PARAM` | `float` | `1` | Shape parameter for the symmetric beta prior on the CFM mixing weight, giving `Beta(1, 1)` by default. |

A padding value of $2$ expands the computational pixel grid to twice its original extent along each axis, so that data-constrained grid points are centered within the padded domain.


### Noise model for Gaussian likelihoods

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NOISE_AXIS` | `str` | `row_wise` | Row-wise (feature-specific) or `simple` (only view-specific) |
| `NOISE_CLIP` | `float` | `1e-16` | Numerical stability clipping |
| `ICOV_GAMMA_PARAM` | `float` | `0.1` | Shape and rate hyperparameter for noise precision Gamma prior |


### Factor initialization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N_FACTORS_INIT` | `int` | `5` | Number of factors to start with |
| `N_FACTORS_FINAL` | `int` | `20` | Number of factors to end with, can be equal to `N_FACTORS_INIT` |
| `N_FACTORS_STEP` | `int` | `1` | Step-size to increase the number of factors used for training |
| `N_SEEDS_PER_STEP` | `int` | `5` | Number of random seeds used for each factor step |
| `FACTOR_INIT` | `str` | `random` | Initialization method (`random`, `PCA`, `NMF`) |

Alternatively, random seeds and reference seeds for posterior alignment can be provided explicitly, as shown in `config/config_*_master.yaml`.


### Inference and variational sampling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `INFERENCE_METHOD` | `str` | `geoVI` | `geoVI` or `MGVI` |
| `N_ITERATIONS` | `int` | `30` | Total global training iterations |
| `RESUME_TRAINING` | `bool` | `false` | Resume interrupted runs |
| `N_ITERATIONS_NOISE_CONSTANT` | `int` | `5` | Phase with fixed noise precision for Gaussian likelihoods |
| `NOISE_ICOV_INITIAL` | `float` | `1` | Initial noise precision for Gaussian likelihoods |
| `N_ITERATIONS_LOW_OPT` | `int` | `5` | Low optimization phase |


For each run, we draw $N$ mirror samples ($2N$ samples in total). Optionally, MOFTy provides an option to change the sampling scheme after a given number of global iterations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N_START_SAMPLES` | `int` | `4` | Initial samples |
| `N_FINAL_SAMPLES` | `int` | `4` | Final samples |
| `N_START_SAMPLES_ITERATIONS` | `int` | `5` | Warmup iterations |


### Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PLOT_DOT_SIZE` | `int` | `15` | Dot size in plots |
| `Z_SLICES` | `int` | `2` | Number of slices for visualization with 3D covariates |
| `FLIP_IMAGE` | `bool` | `false` | Flip plot images for visualization |


### Synthetic data


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SYN_DATA` | `bool`| `false` | Activate synthetic data generation mode. |

Set `SYN_DATA` flag to `true` in your config file to generate synthetic data. This corresponds to drawing one sample from the forward model according to the following parameters.


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N_FACTORS` | `int` | `5` | Number of factors |
| `DATA_SAMPLES`| `str`| `200`| Only used when `USE_COVARIATES` is `false`|
| `DATA_KEYS` | `list[str]`| `[view_1, view_2, view_3]`| Name of views |
| `LIKELIHOOD_OPTIONS` | `dict` | `{view_1: gaussian, view_2: gaussian, view_3: gaussian}` | Likelihood options per view |
| `N_FEATURES` | `list[int]`| `[300, 300, 300]`| Number of features per modality |
| `USE_COVARIATES`| `bool`| `false` | Use covariates |
| `COVARIATES_SHAPE` | `list[int]` | `[30, 30]` | Shape of covariates |
| `INIT_Z_CFM` | `bool` | `false` | Use CFM prior |
| `FLUCTUATIONS` | `list[float]` | <code>[3,&nbsp;0.5]</code> | Prior mean and standard deviation for the overall fluctuation amplitude of the correlated field. |
| `LOGLOGAVGSLOPE` | `list[float]` | <code>[-5,&nbsp;0.5]</code> | Prior mean and standard deviation for the average slope of the log power spectrum on a log-log scale. |
| `FLEXIBILITY` | `list[float]` | <code>[1,&nbsp;0.5]</code> | Prior mean and standard deviation controlling deviations from a simple power-law spectrum. |
| `ASPERITY` | `list[float]` | <code>[1,&nbsp;0.5]</code> | Prior mean and standard deviation controlling small-scale roughness of the log power spectrum. |
| `OFFSET_MEAN` | `float` | `0` | Prior mean of the zero-mode, i.e. the field offset. |
| `OFFSET_STD` | `list[float]` | <code>[1,&nbsp;0.5]</code> | Prior mean and standard deviation for the zero-mode uncertainty. |
| `PADDING` | `int` | `1` | Grid padding |
| `SEED`| `int`| `42` | Seed |
| `TARGET_SNR`| `float` | `1` | Target signal-to-noise ratio for views with Gaussian observation model |
| `CFM_SCALING_VALUES` | `list[float]` | `[0.99, 0.75, 0.5, 0.25, 0.01]` | Scaling of `Z_CFM` vs `Z_non-CFM`, has to be equal to `N_FACTORS`|
| `SYN_SCALING` | `float` | `0.6` | Scaling the standard deviation of all variables corresponding except inverse gamma prior on weights |
| `W_IGAMMA_SYN_SCALING` | `float` | `0.6` | Scaling the standard deviation of variables corresponding to inverse gamma prior on weights. |

To reconstruct synthetic data, set `SYN_DATA_REC` flag to `true` in your reconstruction configuration file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SYN_DATA_REC` | `bool` | `false` | Synthetic data reconstruction mode. |


Note that for `SYN_DATA_REC`, the reference seed is ignored during posterior alignment. Instead, the ground-truth factors are used to derive the orthogonal transformations of the latent factors. See the [MOFTy publication](#citing) for more details.

---


## Output format

Each run writes to `OUTPUT_BASE_DIR` with one directory per seed plus one combined directory.

Example for `output/output_gbm_cfm`:

```text
output/output_gbm_cfm
├── log_files/
│   └── task_seed_<seed>_factors_4_{stdout,stderr}.log
├── run_seed_<seed>_factors_4/   (for each seed)
│   ├── mofty_model_Z.hdf5 (MOFA2/muon compatible format containing posterior mean factor values for Z=Z_CFM + Z_non-CFM)
│   ├── mofty_model_Z_cfm.hdf5 (MOFA2/muon compatible format containing posterior mean factor values for Z_CFM)
│   ├── mofty_model_Z_non_cfm.hdf5 (MOFA2/muon compatible format containing posterior mean factor values for Z_non-CFM)
│   ├── W_*.parquet, Z*.parquet, noise_std_*.parquet, normalized_residuals_*.parquet
│   ├── W_*/, Z/, Z_cfm/, Z_non_cfm/, Z_cfm_op_*/, icov_*/, std_*/ (posterior samples or point estimates)
│   │   └── latest.hdf5
│   ├── energy_history/, minisanity_history/, pickle/
│   └── model_plots/
│       ├── <plot_family>_iteration_<0-29>.png
│       └── plot_family ∈ {
│           W_{rna,protein}_{mean_hist,mean,std},
│           Z_{mean_hist,mean,std},
│           Z_cfm_{mean_hist,mean,std},
│           Z_non_cfm_{mean_hist,mean,std},
│           Z_cfm_op_{mean_hist,mean,std,scaling_mean,pspec},
│           normalized_residuals_{rna,protein}_{hist,""},
│           std_{rna,protein}_mean{,_hist}
│         }
└── runs_combined_factors_4/
    ├── mofty_model_Z.hdf5 (MOFA2/muon compatible format containing factor values for Z=Z_CFM + Z_non-CFM)
    ├── mofty_model_Z_cfm.hdf5 (MOFA2/muon compatible format containing factor values for Z_CFM)
    ├── mofty_model_Z_non_cfm.hdf5 (MOFA2/muon compatible format containing factor values for Z_non-CFM)
    ├── [same core outputs as one run_seed directory]
    ├── model_plots/
    │   ├── <plot_family>_iteration_29.png
    │   └── plot_familiy ∈ {
    │       W_{rna,protein}_{mean_hist,mean,std},
    │       Z_{mean_hist,mean,std},
    │       Z_cfm_{mean_hist,mean,std},
    │       Z_non_cfm_{mean_hist,mean,std},
    │       Z_cfm_op_pspec, # diagonal approximation
    │       normalized_residuals_{rna,protein}_{hist,""},
    │       std_{rna,protein}_mean{,_hist}
    │     }
    └── ot_matrix_seed_<seed>.csv (orthogonal transformation matrix to align factors to reference seed)
```

## Integration with MOFA2, muon, and scverse

MOFTy outputs are compatible with:
- MOFA2 (R package; mofax, Python)
- muon, integrated in scverse (Python)

MOFTy can also load untrained MOFA or MEFISTO models as input.

To use muon together with additional packages from the scverse ecosystem, we recommend creating a separate environment for the required dependencies. The following guide describes how to set up all dependencies needed to reproduce the results of the [MOFTy publication](#citing), as described in [Reproducing the MOFTy paper results](#reproducing-the-mofty-paper-results).
```bash
micromamba env create -f muon_analysis.yaml --name muon_analysis --channel-priority flexible

# For GPU support via CuPy
micromamba env create -f muon_analysis_cupy.yaml --name muon_analysis_cupy --channel-priority flexible

```
Similarly, for the MOFA2 R package:
```bash
micromamba env create -f mofa2_analysis.yaml --name mofa2_analysis --channel-priority flexible
```

Next install the MOFA2 package in R. First activate the `mofa2_analysis` environment:
```bash
micromamba activate mofa2_analysis
```
Then, start R by entering the following command in the terminal:
```bash
R
```
Execute the following command in `R`. If you are asked for updates, center `n` for `none`.

```R
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.20")
```
Then execute the following command. If you are asked for updates, center `n` for `none`.
```bash
BiocManager::install("MOFA2")
```
Additionally, `Rfast2` is required for downstream analyses. If you are asked for updates, enter `3` for `none`.
```bash
install.packages("remotes")
remotes::install_version("Rfast2", version = "0.1.5.4", repos = "https://cloud.r-project.org")
```

Troubleshooting: A global system-wide `R` installation outside of conda environments can interfere with building the `mofa2_analysis` environment.
If you encounter related errors, a simple solution is to remove the system-wide `R` installation before building the environment.

---


## Reproducing the MOFTy paper results

For the [MOFTy publication](#citing), the following two systems were used: 
<a id="system-1"></a>
- **System 1:** Apple MacBook Pro (14 inch, November 2024), Apple M4 Max, 36 GB combined memory, macOS 15.1 (24B2083).
<a id="system-2"></a>
- **System 2:** Linux workstation (Ubuntu 24.04.4 LTS), two 64-core Intel Xeon Gold 6430 processors (two threads per core), 1 TB main memory, two NVIDIA A100 GPUs with 80 GB of memory each.

We remark that different hardware configurations may lead to slightly different results. On macOS, we ran all scripts with `nohup /usr/bin/time -l python3 <script + CLI args> &> script.log &`. On Linux, the command is `nohup /usr/bin/time --verbose python3 <script + CLI args> &> script.log &`. If the scripts below are run without `--rp 4` or `--rp 1` (default), the four random-seed models are trained sequentially. Alternatively, using `--rp 2` runs two models in parallel. The `<file_name>_master.yaml` configuration files used below contain the exact random seeds used in the MOFTy paper. To generate new random seeds instead, run MOFTy with `<file_name>.yaml`.

On [System 1](#system-1), we trained up to eight models in parallel using the model configurations described above: the glioblastoma and mouse gastrulation datasets with four random seeds each, and the 1D and 2D synthetic data reconstructions with four random seeds each.
Each model (seed) was initialized on a single CPU core.
For each dataset, total training time across the four parallel seed runs was measured as wall-clock time, and peak memory usage was measured as maximum resident set size.
- 1D synthetic data reconstruction (900 pixels): 2.6 GB total peak memory usage, corresponding to approximately 675 MB per model; 1.9 hours total training time. 
- 2D synthetic data reconstruction (30x30 pixels): 2.3 GB total peak memory usage, corresponding to approximately 625 MB per model; 1.8 hours total training time.
- Glioblastoma dataset (four factors): 5.3 GB total peak memory usage, corresponding to approximately 1.3 GB per model; 21.8 hours total training time.
- Mouse gastrulation dataset: 3.7 GB total peak memory usage, corresponding to approximately 925 MB per model; 8.1 hours total training time.

On [System 2](#system-2), we also trained up two eight models in parallel (four models/random seeds per GPU).
Each model (seed) was initialized on a single CPU core.
For each dataset, total training time across the four parallel seed runs was measured as wall-clock time, and peak memory usage was measured as maximum resident set size and maxmimum allocated GPU memory.
- Glioblastoma dataset (four factors): 2.8 GB (CPU) and 12.2 GB (GPU) memory, corresponding to approximately 700 MB (CPU) and 3.1 GB (GPU) per model; 11.7 hours total training time.
- Glioblastoma dataset (six factors): 3.2 GB (CPU) and 12.5 GB (GPU) memory, corresponding to approximately 800 MB (CPU) and 3.1 GB (GPU) per model; 15.3 hours total training time.
- Glioblastoma dataset (eight factors): 3.2 GB (CPU) and 13.3 GB (GPU) memory, corresponding to approximately 800 MB (CPU) and 3.3 GB (GPU) per model; 19.9 hours total training time.
- Glioblastoma dataset (ten factors): 3.2 GB (CPU) and 13.7 GB (GPU) memory, corresponding to approximately 800 MB (CPU) and 3.4 GB (GPU) per model; 24.3 hours total training time.
- 2D synthetic data reconstruction (100x100 pixels): 3.2 GB (CPU) and 29.7 GB (GPU) memory, corresponding to approximately 800 MB (CPU) and 7.4 GB (GPU) per model; 27.8 hours total training time.
For model comparison, we also trained MEFISTO models (fixed random seed 42) on the glioblastoma dataset with 4 factors and varying number of inducing points
- MEFISTO 1,000 inducing points: 4.6 GB (CPU) and 781 MB (GPU) memory; 0.4 hours total training time.
- MEFISTO 3,000 inducing points: 6.7 GB (CPU) and 781 MB (GPU) memory; 3.5 hours total training time.
- MEFISTO full GP: 13.8 GB (CPU) and 781 MB (GPU) memory; 9.3 hours total training time.

### Synthetic data

To generate the synthetic data used in the [MOFTy publication](#citing) with $900$ pixels (1D) and $30 \times 30$ pixels (2D), activate the `mofty` environment and run:
```bash
python3 run_mofty.py --config config_files/config_syn_cfm_dim1.yaml

python3 run_mofty.py --config config_files/config_syn_cfm_dim2_30.yaml
```
We observed that, for synthetic data, samples generated by the forward model can differ across system configurations. 
For the [MOFTy publication](#citing), we used [System 1](#system-1).

For reconstruction, run:
```bash
python3 run_mofty.py --config config_files/config_syn_rec_cfm_dim1_master.yaml --rp 4

python3 run_mofty.py --config config_files/config_syn_rec_cfm_dim2_30_master.yaml --rp 4
```

To reproduce the paper plots, run:
```bash
cd synthetic_data_plot

python3 syn_1D.py

python3 uncert_1D.py

# 30x30 grid
python3 syn_2D.py --pix_shape 30

python3 uncert_2D.py --pix_shape 30
```

We also generated a synthetic dataset with 100x100 pixels on [System 2](#system-2):
```bash
# To generate and reconstruct the data for the paper, we used the GPU environment. 
micromamba activate mofty_cupy 

python3 run_mofty.py --config config_files/config_syn_cfm_dim2_100.yaml
```

For reconstruction, run:
```bash
python3 run_mofty.py --config config_files/config_syn_rec_cfm_dim2_100_master.yaml --rp 4 --device GPU
```

To reproduce the paper plots, run:
```bash
cd synthetic_data_plot

# 100x100 grid
python3 syn_2D.py --pix_shape 100

python3 uncert_2D.py --pix_shape 100

# Weight correlation tables
python3 weight_corr_tables.py
```

### Glioblastoma dataset

First run `notebooks_py/prepprocessing_gbm.ipynb`, using the `muon_analysis` environment. Then activate the `mofty` environment and run:

```bash
python3 run_mofty.py --config config_files/config_gbm_cfm_4_master.yaml --rp 4
```

Then run `notebooks_py/downstream_gbm_mofty.ipynb` using the `muon_analysis` environment. 
For the MEFISTO, MOFTy, PCA comparison analysis, first train the models:
```bash
cd comparison_gbm

micromamba activate muon_analysis_cupy

# Full GP
python3 run_mefisto.py --gpu_mode --gpu_device 0 --mkl_threads 16

# 3000 inducing points
python3 run_mefisto.py --gpu_mode --gpu_device 0 --mkl_threads 16 --sparseGP  --n_inducing 3000

# 1000 inducing points
python3 run_mefisto.py --gpu_mode --gpu_device 0 --mkl_threads 16 --sparseGP  --n_inducing 1000
```
Then run:
```bash
python3 mofty_factors_comparison.py
```
You can also run the MOFTy downstream analysis for the full-GP MEFISTO model by first running `vignettes_R/mefisto_int_gbm.Rmd` using the `mofa2_analysis` environment and then running `notebooks_py/downstream_gbm_mefisto.ipynb` using the `muon_analysis` environment.

We also trained models 4, 6, 8 and 10 factors on [System 2](#system-2) with the `mofty_cupy` environment:
```bash
python3 run_mofty.py --config config_files/config_gbm_cfm_4_master.yaml --rp 4 --device GPU

python3 run_mofty.py --config config_files/config_gbm_cfm_6_master.yaml --rp 4 --device GPU

python3 run_mofty.py --config config_files/config_gbm_cfm_8_master.yaml --rp 4 --device GPU

python3 run_mofty.py --config config_files/config_gbm_cfm_10_master.yaml --rp 4 --device GPU
```
For the factor comparison analysis activate the `muon_analysis` environment and run:
```bash
cd comparison_gbm

python3 mofty_factors_comparison.py
```

### Mouse gastrulation dataset

First run `vignettes_R/mofty_scnmt_preprocessing.Rmd` using the `mofa2_analysis` environment. Then activate the `mofty` environment and run:

```bash
python3 run_mofty.py --config config_files/config_scnmt_cfm_master.yaml --rp 4
```

Then run `vignettes_R/mofty_scnmt_downstream.Rmd` using the `mofa2_analysis` environment.

---


## References

- Argelaguet, R. et al. Multi-Omics Factor Analysis—a framework for unsupervised integration of multi-omics data sets. Molecular Systems Biology 14 (2018). https://doi.org/10.15252/msb.20178124.
- Argelaguet, R. et al. MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. Genome Biology 21 (2020). https://doi.org/10.1186/s13059-020-02015-1.
- Arras, P. et al. NIFTy5: Numerical Information Field Theory v5. Astrophysics Source Code Library ascl:1903.008 (2019).
- Arras, P. et al. Comparison of classical and Bayesian imaging in radio interferometry. Cygnus A with CLEAN and resolve. Astronomy and Astrophysics 646, A84 (2021). https://doi.org/10.1051/0004-6361/202039258.
- Arras, P. et al. Variable structures in M87* from space, time and frequency resolved interferometry. Nature Astronomy 6, 259–269 (2022). https://doi.org/10.1038/s41550-021-01548-0.
- Bredikhin, D., Kats, I. & Stegle, O. MUON: multimodal omics analysis framework. Genome Biology 23, 42 (2022). https://doi.org/10.1186/s13059-021-02577-8.
- Edenhofer, G. et al. Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference. Journal of Open Source Software 9, 6593 (2024). https://doi.org/10.21105/joss.06593.
- Frank, P., Leike, R. & Enßlin, T. A. Geometric Variational Inference. Entropy 23 (2021). https://doi.org/10.3390/e23070853.
- Knollmüller, J. & Enßlin, T. A. Metric Gaussian Variational Inference. ArXiv e-prints (2020). https://doi.org/10.48550/arXiv.1901.11033.
- Qoku, A. et al. MOFA-FLEX: A Factor Model Framework for Integrating Omics Data with Prior Knowledge. bioRxiv (2025). https://doi.org/10.1101/2025.11.03.686250.
- Steininger, T. et al. NIFTy 3 - Numerical Information Field Theory: A Python Framework for Multicomponent Signal Inference on HPC Clusters. Annalen der Physik 531, 1800290 (2019). https://doi.org/10.1002/andp.201800290.
- Tjur, T. Coeﬃcients of Determination in Logistic Regression Models—A New Proposal: The Coeﬃcient of Discrimination. The American Statistician. 63, 366‐372 (2009,11), http://dx.doi.org/10.1198/tast.2009.08210.
- Townes, F. W. & Engelhardt, B. E. Nonnegative spatial factorization applied to spatial genomics. Nature
Methods 20, 229–238 (2023). https://doi.org/10.1038/s41592-022-01687-w.
- Velten, B. et al. Identifying temporal and spatial patterns of variation from multimodal data using MEFISTO. Nature Methods 19, 179–186 (2022). https://doi.org/10.1038/s41592-021-01343-9.
- Virshup, I., Bredikhin, D., Heumos, L., et al. 2023. The scverse project provides a computational ecosystem for single-cell omics data analysis. Nature Biotechnology 41 (5): 604–6. https://doi.org/10.1038/s41587-023-01733-8.
- Okuta, R., Unno, Y., Nishino, D., Hido, S. & Loomis, C. CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations (2017). LearningSys Workshop at NeurIPS 2017. Available at https://learningsys.org/nips17/assets/papers/paper_16.pdf.
- Message Passing Interface Forum. MPI: A Message-Passing Interface Standard Version 4.0 (2021). https://www.mpi-forum.org/docs/.

---
