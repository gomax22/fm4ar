"""
Utility functions to create corner plots for posterior distributions.
"""

import numpy as np
import corner
import matplotlib.pyplot as plt
from typing import List
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False 
mpl.rc('font',family='Times New Roman')


from fm4ar.evaluation.coverage import compute_confidence_interval

def corner_plot_single_distribution(
    posterior_samples: np.ndarray,
    theta: np.ndarray,
    labels: List,
    output_fname: str,
    label_kwargs: dict = {
        "fontsize": 12,
        "fontname": "Times New Roman",
    },
    title_kwargs: dict = {
        "fontsize": 10,
        "fontname": "Times New Roman",
    },
    legend_kwargs: dict = {
        "fontsize": 20,
    },
    offset: float = 0.2,
) -> None:
    """
    Create a corner plot for a single posterior distribution with true parameter values and confidence intervals.
    Args:
        posterior_samples: np.ndarray of shape (n_posterior_samples, n_dims)
            containing samples from the model posterior.
        theta: np.ndarray of shape (n_dims,) containing the true parameter values.
        labels: List of strings for parameter labels.
        output_fname: Filename to save the corner plot.
        label_kwargs: Dictionary of keyword arguments for label styling.
        title_kwargs: Dictionary of keyword arguments for title styling.
        offset: Float offset to adjust axis limits around true values.
    Returns:
        None
    """

    # compute confidence intervals
    ci_one_sigma = compute_confidence_interval(posterior_samples, 0.68)
    ci_two_sigma = compute_confidence_interval(posterior_samples, 0.95)

    n_targets = posterior_samples.shape[-1]

    try:
        fig = corner.corner(
            posterior_samples, 
            labels=labels, 
            label_kwargs=label_kwargs,
            show_titles=True, 
            color="black",
            title_kwargs=title_kwargs,
        )

        # Extract the axes
        axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
        # Loop over the diagonal
        for i in range(n_targets):
            ax = axes[i, i]
            ax.axvline(theta[i].item(), color="r")
            
            # plot 1-sigma confidence intervals for model
            ax.axvline(ci_one_sigma[0][i], color="k", linestyle="--")
            ax.axvline(ci_one_sigma[1][i], color="k", linestyle="--")

            # plot 2-sigma confidence intervals for model
            ax.axvline(ci_two_sigma[0][i], color="k", linestyle=":")
            ax.axvline(ci_two_sigma[1][i], color="k", linestyle=":")

            x_lim = ax.get_xlim()

            # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
            ax.set_xlim(min(x_lim[0], theta[i]) - offset, max(x_lim[1], theta[i]) + offset) 
        
        # Loop over the histograms
        for yi in range(n_targets): # yi 
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(theta[xi], color="r")
                ax.axhline(theta[yi], color="r")
                ax.plot(theta[xi], theta[yi], "sr")
                x_lim, y_lim = ax.get_xlim(), ax.get_ylim()

                # this offset makes sense in the standardized domain
                ax.set_xlim(min(x_lim[0], theta[xi]) - offset, max(x_lim[1], theta[xi]) + offset) 
                ax.set_ylim(min(y_lim[0], theta[yi]) - offset, max(y_lim[1], theta[yi]) + offset)
        
        # plot legend in upper right corner of the figure
        handles = [Line2D([0], [0], lw=4, color="red")]
        handles += [
            Line2D([0], [0], color=c, lw=4)
            for c in ['black']
        ]
        fig.legend(
            handles=handles,
            fontsize=legend_kwargs.get("fontsize", 20),
            labels=[r"$\theta_{\mathrm{in}}$", r"$p(\theta | \mathcal{E}(x))$"],
            ncols=1,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(0.95, 0.95),
        )

        # Save figure
        fig.savefig(f"{output_fname}.pdf", format='pdf', bbox_inches='tight', dpi=400)
        plt.close(fig)
    except Exception as e:
        print(e)
        print("Error in corner plot")


def corner_plot_prior_posterior(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    output_fname: str,
    labels: List[str],
    label_kwargs: dict = {
        "fontsize": 12,
        "fontname": "Times New Roman",
    },
    title_kwargs: dict = {
        "fontsize": 10,
        "fontname": "Times New Roman",
    },
    offset: float = 0.2,
) -> None:
    """
    Create a corner plot comparing prior samples (thetas) and posterior samples.
    Args:
        posterior_samples: np.ndarray of shape (n_thetas, n_posterior_samples, n_dims)
            containing samples from the model posterior.
        thetas: np.ndarray of shape (n_thetas, n_dims) containing the prior samples.
        output_fname: Filename to save the corner plot.
        labels: List of strings for parameter labels.
        label_kwargs: Dictionary of keyword arguments for label styling.
        title_kwargs: Dictionary of keyword arguments for title styling.
        offset: Float offset to adjust axis limits around mean values.
    Returns:
        None
    """
    # -------------------------------------------------
    # plot prior vs posteriors
    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])

    # take a random subset of size len(thetas)
    idx_model = np.random.choice(posterior_samples.shape[0], len(thetas), replace=False)
    posterior_samples = posterior_samples[idx_model]


    n_targets = thetas.shape[-1]
    fig = corner.corner(
        thetas, 
        labels=labels, 
        label_kwargs=label_kwargs,
        color="green", 
        show_titles=True, 
        title_kwargs=title_kwargs,
    )
    
    corner.corner(
        posterior_samples, 
        labels=labels, 
        show_titles=False, 
        color="blue", 
        fig=fig)   
    
    # Extract the axes
    axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
    # Loop over the diagonal
    for i in range(n_targets):
        ax = axes[i, i]
        ax.axvline(thetas[:, i].mean().item(), color="g", linestyle="--")
        x_lim = ax.get_xlim()

        # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
        ax.set_xlim(min(x_lim[0], thetas[:, i].mean().item()) - offset, max(x_lim[1], thetas[:, i].mean().item()) + offset) 
        ax.axvline(posterior_samples[:, i].mean().item(), color="blue", linestyle="--")
    
    # Loop over the histograms
    for yi in range(n_targets): # yi 
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(thetas[:, xi].mean().item(), color="g", linestyle="--")
            ax.axhline(thetas[:, yi].mean().item(), color="g", linestyle="--")
            ax.plot(thetas[:, xi].mean().item(), thetas[:, yi].mean().item(), "sg")
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()

            # # this offset makes sense in the standardized domain
            ax.set_xlim(min(x_lim[0], thetas[:, xi].mean().item()) - offset, max(x_lim[1], thetas[:, xi].mean().item()) + offset) 
            ax.set_ylim(min(y_lim[0], thetas[:, yi].mean().item()) - offset, max(y_lim[1], thetas[:, yi].mean().item()) + offset)
            
    # plot legend in upper right corner of the figure
    plt.figlegend(["PRIOR", "POSTERIOR"], fontsize='large')
    
    # both with theta
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)
    return None
