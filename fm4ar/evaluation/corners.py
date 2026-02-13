"""
Utility functions to create corner plots for posterior distributions.
"""

import numpy as np
import corner
import matplotlib.pyplot as plt
from typing import List
from matplotlib.lines import Line2D
import matplotlib as mpl
from pathlib import Path
import os
from tqdm import tqdm

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



def corner_plot_prior_posteriors(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: List[str],
    output_dir: str
):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    # -------------------------------------------------
    # plot prior vs posteriors
    thetas = thetas.reshape(-1, thetas.shape[-1])
    posteriors = np.reshape(posteriors, (len(posteriors), -1, posteriors[0].shape[-1]))

    # take a random subset of size len(thetas)
    random_indices = np.random.choice(thetas.shape[0], len(thetas), replace=False)
    thetas = thetas[random_indices]
    posteriors = posteriors[:, random_indices]

    # make a corner plot of prior and posterior samples
    n_targets = thetas.shape[-1]
    # fig = corner.corner(
    #     thetas,
    #     labels=labels,
    #     label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
    #     color="green",
    #     show_titles=True, 
    #     title_kwargs={"fontsize": 12, "fontname": "Times New Roman"}
    # )
    fig = corner.corner(
        data=thetas,
        labels=labels, 
        color="green", 
        bins=100,
        smooth=3.0,
        smooth1d=3.0,
        show_titles=True,
        plot_datapoints=False,
        plot_contours=True,
        plot_density=False,
        levels=(0.68, 0.95),
        contour_kwargs=dict(linewidths=2.0),
        label_kwargs={"fontsize": 12, "fontname": "Times New Roman"}
    )
    
    for posterior, color in zip(posteriors, colors):
        # corner.corner(
        #     posterior, 
        #     labels=labels, 
        #     label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        #     color=color, 
        #     show_titles=False, 
        #     fig=fig
        # )   
        corner.corner(
            posterior, 
            labels=labels,
            color=color, 
            bins=100,
            smooth=3.0,
            smooth1d=3.0,
            show_titles=False,
            plot_datapoints=False,
            plot_contours=True,
            plot_density=False,
            levels=(0.68, 0.95),
            contour_kwargs=dict(linewidths=2.0),
            label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
            fig=fig
        )
    
    # Extract the axes
    axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
    # Loop over the diagonal
    # for i in range(n_targets):
    #     ax = axes[i, i]
    #     ax.axvline(thetas[:, i].mean().item(), color="g", linestyle="--")
    #     x_lim = ax.get_xlim()
    #     ax.set_xlim(min(x_lim[0], thetas[:, i].mean().item()) - 0.2, max(x_lim[1], thetas[:, i].mean().item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
    #     for posterior, color in zip(posteriors, colors):
    #         ax.axvline(posterior[:, i].mean().item(), color=color, linestyle="--")
    
    # Loop over the histograms
    # for yi in range(n_targets): # yi 
    #     for xi in range(yi):
    #         ax = axes[yi, xi]
    #         # ax.set_title(f"yi: {yi}, xi: {xi}")
    #         ax.axvline(thetas[:, xi].mean().item(), color="g", linestyle="--")
            
    #         # ax.axvline(value2[xi], color="r")
    #         ax.axhline(thetas[:, yi].mean().item(), color="g", linestyle="--")
            
    #         for posterior, color in zip(posteriors, colors):
    #             ax.axvline(posterior[:, xi].mean().item(), color=color, linestyle="--")
    #             ax.axhline(posterior[:, yi].mean().item(), color=color, linestyle="--")
            
    #         # ax.axhline(value2[yi], color="r")
    #         ax.plot(thetas[:, xi].mean().item(), thetas[:, yi].mean().item(), "sg")
    #         # ax.plot(value2[xi], value2[yi], "sr")
    #         x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    #         ax.set_xlim(min(x_lim[0], thetas[:, xi].mean().item()) - 0.2, max(x_lim[1], thetas[:, xi].mean().item()) + 0.2)
    #         ax.set_ylim(min(y_lim[0], thetas[:, yi].mean().item()) - 0.2, max(y_lim[1], thetas[:, yi].mean().item()) + 0.2)

    # plot legend in upper right corner of the figure
    handles = [Line2D([0], [0], lw=4, color="green")]
    handles += [
        Line2D([0], [0], color=c, lw=4)
        for c in colors
    ]
    fig.legend(
        handles=handles,
        labels=[r"$\theta_{\mathrm{in}}$"] + [label for label in model_labels],
        ncols=1,
        frameon=False,
        loc="upper right",
        fontsize=20,
        bbox_to_anchor=(0.95, 0.95),
    )
    # plt.figlegend(model_labels, fontsize='large')
    fig.savefig(
        os.path.join(
            output_dir, "corner_plot_prior_posterior.png"
        ), 
        format='png', 
        bbox_inches='tight', 
        dpi=300
    )
    fig.savefig(
        os.path.join(
            output_dir,
            "corner_plot_prior_posterior.pdf"
        ),
        format='pdf', 
        bbox_inches='tight', 
        dpi=300
    )
    
    plt.close(fig)


def corner_plot_multiple_distributions(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: List[str],
    output_dir: str
    ):  

    num_samples, num_targets = thetas.shape

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx in tqdm(range(num_samples), desc="Plotting corners..."):
        theta = thetas[idx]
        posterior_samples = posteriors[:, idx]

        sample_output_dir = os.path.join(output_dir, f"sample_{idx}")
        # sample_output_dir = os.path.join(output_dir, f"sample_{1723}")
        if not Path(sample_output_dir).exists():
            Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

        # fig = corner.corner(
        #     posterior_samples[0], 
        #     labels=labels, 
        #     label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},

        #     hist_kwargs={
        #         "histtype": "stepfilled", 
        #         "alpha": 0.5, 
        #         "color": colors[0],
                
        #     },
        #     hist2d_kwargs={
        #         "levels": (0.68, 0.95),
        #         # "no_fill_contours": True,
        #         "plot_datapoints": False,
        #         "plot_density": False,
        #         "plot_contours": True,
        #         "pcolor_kwargs": {"alpha": 0.5},
        #         "contour_kwargs": {"alpha": 0.5, "lw": 0.5},
        #         "contourf_kwargs": {"alpha": 0.5, "lw": 0.5},
        #         "alpha": 0.5, 
        #         "smooth": 3.0,
        #         "smooth1d": 3.0,},
        #     bins=100,
        #     show_titles=True, 
        #     color=colors[0], 
        #     title_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        # )
        fig = corner.corner(
            data=posterior_samples[0],
            labels=labels, 
            color=colors[0], 
            bins=100,
            smooth=3.0,
            smooth1d=3.0,
            plot_datapoints=False,
            plot_contours=True,
            plot_density=False,
            levels=(0.68, 0.95),
            contour_kwargs=dict(linewidths=2.0),
            label_kwargs={"fontsize": 12, "fontname": "Times New Roman"}
        )
        # fig = corner.corner(posteriors[0], labels=labels, show_titles=False, title_kwargs={"fontsize": 10})
        # Extract the axes
        axes = np.array(fig.axes).reshape((num_targets, num_targets))
        
        for posterior, color in zip(posterior_samples[1:], colors[1:]):
            # corner.corner(
            #     posterior, 
            #     labels=labels,

            #     hist_kwargs={
            #         "histtype": "stepfilled", 
            #         "alpha": 0.5, 
            #         "color": color,
            #     },
            #     hist2d_kwargs={
            #         "levels": (0.68, 0.95),
            #         # "no_fill_contours": True,
            #         "plot_datapoints": False,
            #         "plot_density": False,
            #         "plot_contours": True,
            #         "pcolor_kwargs": {"alpha": 0.5},
            #         "contour_kwargs": {"alpha": 0.5, "lw": 0.5},
            #         "contourf_kwargs": {"alpha": 0.5, "lw": 0.5},
            #         "alpha": 0.5, 
            #         "smooth": 3.0,
            #         "smooth1d": 3.0,},
            #     color=color, 
            #     bins=100,
            #     levels=(0.68, 0.95),
            #     show_titles=False, 
            #     fig=fig
            # )
            corner.corner(
                posterior, 
                labels=labels,
                color=color, 
                bins=100,
                smooth=3.0,
                smooth1d=3.0,
                plot_datapoints=False,
                plot_contours=True,
                plot_density=False,
                levels=(0.68, 0.95),
                contour_kwargs=dict(linewidths=2.0),
                label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
                show_titles=True, 
                fig=fig
            )
            


            # compute confidence intervals
            # ci_one_sigma = compute_confidence_interval(posterior, 0.68)
            # ci_two_sigma = compute_confidence_interval(posterior, 0.95)

            # # Loop over the diagonal
            # for i in range(n_targets):
            #     ax = axes[i, i]
            #     # plot 1-sigma confidence intervals for model
            #     ax.axvline(ci_one_sigma[0][i], color=color, linestyle="--")
            #     ax.axvline(ci_one_sigma[1][i], color=color, linestyle="--")

            #     # plot 2-sigma confidence intervals for model
            #     ax.axvline(ci_two_sigma[0][i], color=color, linestyle=":")
            #     ax.axvline(ci_two_sigma[1][i], color=color, linestyle=":")
        
        # Loop over the diagonal
        for i in range(num_targets):
            ax = axes[i, i]
            ax.axvline(theta[i].item(), linestyle="--", color="g", lw=2)
            x_lim = ax.get_xlim()
            ax.set_xlim(min(x_lim[0], theta[i].item()) - 0.2, max(x_lim[1], theta[i].item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)

        # Loop over the histograms
        for yi in range(num_targets): # yi 
            for xi in range(yi):
                ax = axes[yi, xi]
                # ax.set_title(f"yi: {yi}, xi: {xi}")
                ax.axvline(theta[xi].item(), linestyle="--", color="g", lw=2)
                
                # ax.axvline(value2[xi], color="r")
                ax.axhline(theta[yi].item(), linestyle="--", color="g", lw=2)

                # ax.axhline(value2[yi], color="r")
                ax.plot(theta[xi].item(), theta[yi].item(), "sg", markersize=4)
                # ax.plot(value2[xi], value2[yi], "sr")
                x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
                ax.set_xlim(min(x_lim[0], theta[xi].item()) - 0.2, max(x_lim[1], theta[xi].item()) + 0.2) # this offset makes sense in the standardized domain
                ax.set_ylim(min(y_lim[0], theta[yi].item()) - 0.2, max(y_lim[1], theta[yi].item()) + 0.2)



        # plot legend in upper right corner of the figure
        handles = [Line2D([0], [0], lw=4.0, ls="--", color="green")]
        handles += [
            Line2D([0], [0], color=c, lw=4)
            for c in colors
        ]
        fig.legend(
            handles=handles,
            labels=[rf'Ground truth ($\theta^{{\mathrm{{in}}}}_{{{idx}}}$)'] + [label for label in model_labels],
            ncols=1,
            frameon=False,
            loc="upper right",
            fontsize=20,
            bbox_to_anchor=(0.95, 0.95),
        )
        
        # plt.figlegend(model_labels, fontsize='large')
        fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{idx}.png"), dpi=400)
        fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{idx}.pdf"), format='pdf', bbox_inches='tight', dpi=400)
        # fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{1723}.png"), dpi=400)
        # fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{1723}.pdf"), format='pdf', bbox_inches='tight', dpi=400)
        
        plt.close(fig)