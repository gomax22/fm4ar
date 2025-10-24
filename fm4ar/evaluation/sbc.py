# Taken from: https://github.com/sbi-dev/sbi/blob/main/sbi/diagnostics/sbc.py
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import torch
import pickle
from pathlib import Path
from pprint import pprint


from torch import ones, zeros
from torch.distributions import Uniform
from typing import Callable, Dict, List, Tuple, Union, Optional, Any
import numpy as np
from scipy.stats import kstest, uniform
import warnings
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from scipy.stats import binom

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    classifier: Union[str, Callable] = "rf",
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> torch.Tensor:
    """
    Return classifier based two-sample test accuracy between X and Y.

    For details on the method, see [1,2]. If the returned accuracy is 0.5, <X>
    and <Y> are considered to be from the same generating PDF, i.e. they can not
    be differentiated. If the returned accuracy is around 1., <X> and <Y> are
    considered to be from two different generating PDFs.

    Training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used
    (<classifier> = 'rf'). Alternatively, a multi-layer perceptron is available
    (<classifier> = 'mlp'). For a small study on the pros and cons for this
    choice see [4].

    Note: Both set of samples are normalized (z scored) using the mean and std
    of the samples in <X>. If <z_score> is set to False, no normalization is
    done. If features in <X> are close to constant with std close to zero, the
    std is set to 1 to avoud division by zero.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_` method in this module.

    Args:
        X: Samples from one distribution. Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use metric: sklearn compliant metric to use
        for the scoring parameter of
            cross_val_score
        classifier: classification architecture to use. Defaults to "rf" for a
            RandomForestClassifier. Should be a sklearn classifier, or a
            Callable that behaves like one.
        z_score: Z-scoring using X, i.e. mean and std deviation of X is
            used to normalize X and Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation
            <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of
        sklearn.model_selection.cross_val_score

    Return:
        torch.tensor containing the mean accuracy score over the test sets from
        cross-validation

    Example: ``` py > c2st(X,Y) [0.51904464] #X and Y likely come from the same
    PDF or ensemble > c2st(P,Q) [0.998456] #P and Q likely come from two
    different PDFs or ensembles ```

    References:
        [1]: http://arxiv.org/abs/1610.06545 [2]:
        https://www.osti.gov/biblio/826696/ [3]:
        https://scikit-learn.org/stable/modules/cross_validation.html [4]:
        https://github.com/psteinb/c2st/
    """

    # the default configuration
    if classifier == "rf":
        clf_class = RandomForestClassifier
        clf_kwargs = classifier_kwargs or {}  # use sklearn defaults
    elif classifier == "mlp":
        ndim = X.shape[-1]
        clf_class = MLPClassifier
        # set defaults for the MLP
        clf_kwargs = classifier_kwargs or {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        # Set std to 1 if it is close to zero.
        X_std[X_std < 1e-14] = 1
        assert not torch.any(torch.isnan(X_mean)), "X_mean contains NaNs"
        assert not torch.any(torch.isnan(X_std)), "X_std contains NaNs"
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data, convert to numpy
    data = np.concatenate((X.cpu().numpy(), Y.cpu().numpy()))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return torch.from_numpy(scores).mean()



# from sbi.analysis.plot import sbc_rank_plot
# from sbi.utils.metrics import c2st

# from sbi.analysis import pairplot
# from sbi.analysis.plot import sbc_rank_plot
# from sbi.analysis.plot import plot_tarp
# from sbi.utils.metrics import c2st

# from sbi.diagnostics.sbc import run_sbc, check_sbc
# from sbi.analysis.plot import sbc_rank_plot


# from sbi.analysis.sbc import check_sbc, run_sbc (in 0.23.3 sbi.analysis.sbc -> sbi.diagnostics.sbc)
# from sbi.analysis.sbc import check_sbc, run_sbc
# adapted version of run_sbc 
def run_sbc(
    thetas: torch.torch.Tensor,
    posterior_samples: torch.torch.Tensor,
    reduce_fns: Union[str, Callable, List[Callable]] = "marginals",
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_sbc_samples, num_posterior_samples, dim_theta = posterior_samples.shape

    if num_sbc_samples < 100:
        warnings.warn(
            """Number of SBC samples should be on the order of 100s to give realiable
            results. We recommend using 300."""
        )
    if num_posterior_samples < 100:
        warnings.warn(
            """Number of posterior samples for ranking should be on the order
            of 100s to give reliable SBC results. We recommend using at least 300."""
        )

    if isinstance(reduce_fns, str):
        assert reduce_fns == "marginals", (
            "`reduce_fn` must either be the string `marginals` or a Callable or a List "
            "of Callables."
        )
        reduce_fns = [
            eval(f"lambda theta, x: theta[:, {i}]") for i in range(dim_theta)
        ]


    dap_samples = torch.zeros((num_sbc_samples, dim_theta))
    ranks = torch.zeros((num_sbc_samples, len(reduce_fns)))

    pbar = tqdm(total=num_sbc_samples, desc="Running simulation-based calibration...")
    for idx, (tho, posterior_samples_batch) in enumerate(
        zip(thetas, posterior_samples)):
        
        # Save one random sample for data average posterior (dap).
        sample_idx = np.random.choice(range(num_posterior_samples))
        dap_samples[idx] = posterior_samples_batch[sample_idx]

        # rank for each posterior dimension as in Talts et al. section 4.1.
        for i, reduce_fn in enumerate(reduce_fns):
            ranks[idx, i] = (
                (reduce_fn(posterior_samples_batch, None) < reduce_fn(tho.unsqueeze(0), None)).sum().item()
            )
        pbar.update(1)

    return ranks, dap_samples


def check_sbc(
    ranks: torch.Tensor,
    prior_samples: torch.Tensor,
    dap_samples: torch.Tensor,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> Dict[str, torch.Tensor]:
    """Return uniformity checks and data averaged posterior checks for SBC.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        prior_samples: N samples from the prior
        dap_samples: N samples from the data averaged posterior
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_c2st_repetitions: number of times c2st is repeated to estimate robustness.

    Returns (all in a dictionary):
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
        c2st_dap: C2ST accuracy between prior and dap samples, single value.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            """You are computing SBC checks with less than 100 samples. These checks
            should be based on a large number of test samples theta_o, x_o. We
            recommend using at least 100."""
        )

    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )
    c2st_scores_dap = check_prior_vs_dap(prior_samples, dap_samples)

    return dict(
        ks_pvals=ks_pvals,
        c2st_ranks=c2st_ranks,
        c2st_dap=c2st_scores_dap,
    )


def check_prior_vs_dap(prior_samples: torch.Tensor, dap_samples: torch.Tensor) -> torch.Tensor:
    """Returns the c2st accuracy between prior and data avaraged posterior samples.

    c2st is calculated for each dimension separately.

    According to simulation-based calibration, the inference methods is well-calibrated
    if the data averaged posterior samples follow the same distribution as the prior,
    i.e., if the c2st score is close to 0.5. If it is not, then this suggests that the
    inference method is not well-calibrated (see Talts et al, "Simulation-based
    calibration" for details).
    """

    assert prior_samples.shape == dap_samples.shape, f"Shapes of prior ({prior_samples.shape}) and dap ({dap_samples.shape}) samples do not match."

    return torch.tensor(
        [
            c2st(s1.unsqueeze(1), s2.unsqueeze(1))
            for s1, s2 in zip(prior_samples.T, dap_samples.T)
        ]
    )


def check_uniformity_frequentist(ranks, num_posterior_samples) -> torch.Tensor:
    """Return p-values for uniformity of the ranks.

    Calculates Kolomogorov-Smirnov test using scipy.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.

    Returns:
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
    """
    kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
    )

    return kstest_pvals


def check_uniformity_c2st(
    ranks, num_posterior_samples, num_repetitions: int = 1
) -> torch.Tensor:
    """Return c2st scores for uniformity of the ranks.

    Run a c2st between ranks and uniform samples.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_repetitions: repetitions of C2ST tests estimate classifier variance.

    Returns:
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
    """

    c2st_scores = torch.tensor(
        [
            [
                c2st(
                    rks.unsqueeze(1),
                    Uniform(zeros(1), num_posterior_samples * ones(1)).sample(
                        torch.Size((ranks.shape[0],))
                    ),
                )
                for rks in ranks.T
            ]
            for _ in range(num_repetitions)
        ]
    )

    # Use variance over repetitions to estimate robustness of c2st.
    if (c2st_scores.std(0) > 0.05).any():
        warnings.warn(
            f"""C2ST score variability is larger than {0.05}: std={c2st_scores.std(0)},
            result may be unreliable. Consider increasing the number of samples.
            """
        )

    # Return the mean over repetitions as c2st score estimate.
    return c2st_scores.mean(0)



def sbc_rank_plot(
    ranks: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Additional options can be passed via the kwargs argument, see _sbc_rank_plot.

    Args:
        ranks: torch.Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.

    Returns:
        fig, ax: figure and axis objects.

    """

    return _sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins,
        plot_type,
        parameter_labels,
        ranks_labels,
        colors,
        fig=fig,
        ax=ax,
        figsize=figsize,
        **kwargs,
    )


def _sbc_rank_plot(
    ranks: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_repeats: int = 50,
    line_alpha: float = 0.8,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.3,
    xlim_offset_factor: float = 0.1,
    num_cols: int = 4,
    params_in_subplots: bool = False,
    show_ylabel: bool = False,
    sharey: bool = False,
    fig: Optional[FigureBase] = None,
    legend_kwargs: Optional[Dict] = None,
    ax=None,  # no type hint to avoid hassle with pyright. Should be `array(Axes).`
    figsize: Optional[tuple] = None,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Args:
        ranks: torch.Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.
        num_repeats: number of repeats for each empirical CDF step (resolution).
        line_alpha: alpha for cdf lines or histograms.
        show_uniform_region: whether to plot the region showing the cdfs expected under
            uniformity.
        uniform_region_alpha: alpha for region showing the cdfs expected under
            uniformity.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        num_cols: number of subplot columns, e.g., when plotting ranks of many
            parameters.
        params_in_subplots: whether to show each parameter in a separate subplot, or
            all in one.
        show_ylabel: whether to show ylabels and ticks.
        sharey: whether to share the y-labels, ticks, and limits across subplots.
        fig: figure object to plot in.
        ax: axis object, must contain as many sublpots as parameters or len(ranks).
        figsize: dimensions of figure object, default (8, 5) or (len(ranks) * 4, 5).

    Returns:
        fig, ax: figure and axis objects.

    """

    if isinstance(ranks, (torch.Tensor, np.ndarray)):
        ranks_list = [ranks]
    else:
        assert isinstance(ranks, List)
        ranks_list = ranks
    for idx, rank in enumerate(ranks_list):
        assert isinstance(rank, (torch.Tensor, np.ndarray))
        if isinstance(rank, torch.Tensor):
            ranks_list[idx]: np.ndarray = rank.numpy()  # type: ignore

    plot_types = ["hist", "cdf"]
    assert (
        plot_type in plot_types
    ), "plot type {plot_type} not implemented, use one in {plot_types}."

    if legend_kwargs is None:
        legend_kwargs = dict(loc="best", handlelength=0.8)

    num_sbc_runs, num_parameters = ranks_list[0].shape
    num_ranks = len(ranks_list)

    # For multiple methods, and for the hist plots plot each param in a separate subplot
    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for ranki in ranks_list:
        assert (
            ranki.shape == ranks_list[0].shape
        ), "all ranks in list must have the same shape."

    num_rows = int(np.ceil(num_parameters / num_cols))
    if figsize is None:
        figsize = (num_parameters * 4, num_rows * 5) if params_in_subplots else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i + 1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i + 1}" for i in range(num_ranks)]
    if num_bins is None:
        # Recommendation from Talts et al.
        num_bins = num_sbc_runs // 20

    # Plot one row subplot for each parameter, different "methods" on top of each other.
    if params_in_subplots:
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                num_rows,
                min(num_parameters, num_cols),
                figsize=figsize,
                sharey=sharey,
            )
            ax = np.atleast_1d(ax)  # type: ignore
        else:
            assert (
                ax.size >= num_parameters
            ), "There must be at least as many subplots as parameters."
            num_rows = ax.shape[0] if ax.ndim > 1 else 1
        assert ax is not None

        col_idx, row_idx = 0, 0
        for ii, ranki in enumerate(ranks_list):
            for jj in range(num_parameters):
                col_idx = jj if num_rows == 1 else jj % num_cols
                row_idx = jj // num_cols
                plt.sca(ax[col_idx] if num_rows == 1 else ax[row_idx, col_idx])

                if plot_type == "cdf":
                    _plot_ranks_as_cdf(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_repeats,
                        ranks_label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior ranks {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=jj == 0,
                        alpha=line_alpha,
                    )
                    if ii == 0 and show_uniform_region:
                        _plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs,
                            num_bins,
                            num_repeats,
                            alpha=uniform_region_alpha,
                        )
                elif plot_type == "hist":
                    _plot_ranks_as_hist(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_posterior_samples,
                        ranks_label=ranks_labels[ii],
                        color="firebrick" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=show_ylabel,
                        alpha=line_alpha,
                        xlim_offset_factor=xlim_offset_factor,
                    )
                    # Plot expected uniform band.
                    _plot_hist_region_expected_under_uniformity(
                        num_sbc_runs,
                        num_bins,
                        num_posterior_samples,
                        alpha=uniform_region_alpha,
                    )
                    # show legend only in first subplot.
                    if jj == 0 and ranks_labels[ii] is not None:
                        plt.legend(**legend_kwargs)

                else:
                    raise ValueError(
                        f"plot_type {plot_type} not defined, use one in {plot_types}"
                    )
                # Remove empty subplots.
        col_idx += 1
        while num_rows > 1 and col_idx < num_cols:
            ax[row_idx, col_idx].axis("off")
            col_idx += 1

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks_list[0]
        for jj in range(num_parameters):
            _plot_ranks_as_cdf(
                ranki[:, jj],  # type: ignore
                num_bins,
                num_repeats,
                ranks_label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel="posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            _plot_cdf_region_expected_under_uniformity(
                num_sbc_runs,
                num_bins,
                num_repeats,
                alpha=uniform_region_alpha,
            )
        # show legend on the last subplot.
        plt.legend(**legend_kwargs)

    return fig, ax  # pyright: ignore[reportReturnType]


def _plot_ranks_as_hist(
    ranks: np.ndarray,
    num_bins: int,
    num_posterior_samples: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: str = "firebrick",
    alpha: float = 0.8,
    show_ylabel: bool = False,
    num_ticks: int = 3,
    xlim_offset_factor: float = 0.1,
) -> None:
    """Plot ranks as histograms on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_posteriors_samples: number of posterior samples used for ranking.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter.
        color: histogram color, default from Talts et al.
        alpha: histogram transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        legend_kwargs: kwargs for the legend.
    """
    xlim_offset = int(num_posterior_samples * xlim_offset_factor)
    plt.hist(
        ranks,
        bins=num_bins,
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.ylabel("counts")
    else:
        plt.yticks([])

    plt.xlim(-xlim_offset, num_posterior_samples + xlim_offset)
    plt.xticks(np.linspace(0, num_posterior_samples, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_ranks_as_cdf(
    ranks: np.ndarray,
    num_bins: int,
    num_repeats: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_ylabel: bool = True,
    num_ticks: int = 3,
) -> None:
    """Plot ranks as empirical CDFs on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_repeats: number of repeats of each CDF step, i.e., resolution of the eCDF.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter
        color: line color for the cdf.
        alpha: line transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        legend_kwargs: kwargs for the legend.

    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_repeats: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity on the current axis."""

    # Construct uniform histogram.
    uni_bins = binom(num_sbc_runs, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    # Decrease value one in last entry by epsilon to find valid
    # confidence intervals.
    uni_bins_cdf[-1] -= 1e-9

    lower = [binom(num_sbc_runs, p=p).ppf(0.005) for p in uni_bins_cdf]
    upper = [binom(num_sbc_runs, p=p).ppf(0.995) for p in uni_bins_cdf]

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_repeats * num_bins),
        y1=np.repeat(lower / np.max(lower), num_repeats),
        y2=np.repeat(upper / np.max(upper), num_repeats),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )


def _plot_hist_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_posterior_samples: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity."""

    lower = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.005)
    upper = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.995)

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_posterior_samples, num_bins),
        y1=np.repeat(lower, num_bins),
        y2=np.repeat(upper, num_bins),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )


def run_simulation_based_calibration(
    posterior_samples: torch.Tensor,
    thetas: torch.Tensor,
    output_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Run Simulation-Based Calibration (SBC) evaluation.
    Args:
        posterior_samples: Tensor of shape (n_thetas, n_posterior_samples, n_dims)
            containing samples from the model posterior.
        thetas: Tensor of shape (n_thetas, n_dims) containing the true parameters.
        output_dir: Directory where to save the SBC results.
    Returns:
        A tuple containing:
        (1) ranks: Tensor of shape (n_thetas, n_dims) containing the SBC ranks.
        (2) dap_samples: Tensor of shape (n_thetas, n_dims) containing the DAP samples.
        (3) check_stats: Dictionary containing SBC check statistics.
    """


    
    # Run SBC
    num_samples, num_posterior_samples, dim_theta = posterior_samples.shape
    ranks, dap_samples = run_sbc(
        thetas,
        posterior_samples,
    )

    # Generate and save sbc rank histogram plot
    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="hist",
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    f.savefig(output_dir / "hist.png", dpi=400)

    # Generate and save sbc rank cdf plot
    f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf")
    f.savefig(output_dir / "cdf.png", dpi=400)

    # Compute SBC check statistics
    check_stats = check_sbc(ranks, thetas, dap_samples, num_posterior_samples)
    # pprint(check_stats)

    # Save SBC statistics
    with open(output_dir / "stats.pkl", "wb") as f:
        pickle.dump(check_stats, f)

    return ranks, dap_samples, check_stats
