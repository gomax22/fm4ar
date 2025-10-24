import os
import torch
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

from fm4ar.datasets import load_dataset
from fm4ar.evaluation.config import (
    CalibrationConfig, 
    RegressionConfig, 
    CoverageConfig,
    LogProbsConfig,
    DrawCornerPlotsConfig
)
from fm4ar.evaluation.sbc import run_simulation_based_calibration
from fm4ar.evaluation.tarp import run_tarp_evaluation
from fm4ar.evaluation.calibration import (
    compute_calibration_metrics, 
    plot_calibration_diagrams
)
from fm4ar.evaluation.corners import (
    corner_plot_prior_posterior,
    corner_plot_single_distribution
)

from fm4ar.evaluation.coverage import perform_coverage_analysis
from fm4ar.evaluation.regression import (
    calculate_errors,
    compute_regression_metrics_from_errors,
    save_regression_metrics_to_csv,
    save_regression_errors_to_csv,
    make_violin_plots_of_errors,
)
from fm4ar.evaluation.log_probs import save_log_probs_to_csv
from fm4ar.evaluation.args import get_cli_arguments
from fm4ar.evaluation.config import load_config
from fm4ar.utils.config import load_config as load_experiment_config


def run_regression_metrics(
    args: argparse.Namespace,
    config: RegressionConfig,
    output_dir: Path
) -> dict:
    """
    Run regression metrics evaluation.
    Args:
        args: Command line arguments.
        config: Regression configuration.
        output_dir: Output directory for saving results.
    Returns:
        A dictionary containing regression metrics.
    """

    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    print("Loading experiment config...", end=' ', flush=True)
    experiment_config = load_experiment_config(
        experiment_dir=args.experiment_dir
    )
    print("Done!", flush=True)

    # Load posterior samples
    print("Loading posterior samples...", end=' ', flush=True)
    posterior_samples = np.load(
        args.experiment_dir / 'posterior_distribution.npy'
    )
    print("Done!", flush=True)

    # Load top samples
    print("Loading top posterior samples...", end=' ', flush=True)
    top_samples = np.load(
        args.experiment_dir / 'posterior_top_samples.npy'
    )
    print("Done!", flush=True)


    # Load thetas
    print("Loading test dataset...", end=' ', flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    thetas = test_dataset.get_parameters()
    print("Done!", flush=True)

    # compute squared and absolute errors
    print("Calculating errors...", end=' ', flush=True)
    errors = calculate_errors(
        posterior_samples,
        thetas,
        top_samples,
    )
    print("Done!", flush=True)

    # compute metrics
    print("Computing regression metrics...", end=' ', flush=True)
    metrics = compute_regression_metrics_from_errors(**errors)
    print("Done!", flush=True)

    # make violin plots
    print("Making violin plots of errors...", end=' ', flush=True)
    make_violin_plots_of_errors(
        errors=errors,
        output_dir=output_dir,
        figsize=config.figsize,
        fontsize=config.fontsize,
        y_scale=config.y_scale,
        labels=test_dataset.get_parameters_labels(),
    )
    print("Done!", flush=True)

    print("Saving regression errors to CSV...", end=' ', flush=True)
    save_regression_errors_to_csv(
        **errors,
        output_dir=output_dir,
        labels=test_dataset.get_parameters_labels(),
    )
    print("Done!", flush=True)

    # save metrics to csv
    print("Saving regression metrics to CSV...", end=' ', flush=True)
    save_regression_metrics_to_csv(
        metrics=metrics,
        output_dir=output_dir,
        labels=test_dataset.get_parameters_labels(),
    )
    print("Done!", flush=True)

    del posterior_samples, thetas, test_dataset, experiment_config

    return metrics



def run_calibration_metrics(
    args: argparse.Namespace,
    config: CalibrationConfig,
    output_dir: Path
) -> dict:
    
    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    print("Loading experiment config...", end=' ', flush=True)
    experiment_config = load_experiment_config(
        experiment_dir=args.experiment_dir
    )
    print("Done!", flush=True)

    # Load posterior samples
    print("Loading posterior samples...", end=' ', flush=True)
    posterior_samples = np.load(
        args.experiment_dir / 'posterior_distribution.npy'
    )
    print("Done!", flush=True)


    # Load thetas
    print("Loading test dataset...", end=' ', flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    thetas = test_dataset.get_parameters()
    print("Done!", flush=True)
    
    # perform one-dimensional sbc ranking
    # multi-dimensional sbc requires log probs of reference samples
    print("Running simulation-based calibration...", end=' ', flush=True)
    ranks, dap_samples, sbc_stats = run_simulation_based_calibration(
        posterior_samples=torch.from_numpy(posterior_samples), 
        thetas=torch.from_numpy(thetas),
        output_dir=output_dir
    )
    print("Done!", flush=True)

    # Posterior calibration with TARP (Lemos et al. 2023)
    # the tarp method returns the ECP values for a given set of alpha coverage levels.
    # TODO: handle references properly (default is None)
    print("Running TARP evaluation...", flush=True)
    tarp_results = run_tarp_evaluation(
        posterior_samples=torch.from_numpy(posterior_samples),
        thetas=torch.from_numpy(thetas),
        output_dir=output_dir,
        references=None,
    )
    print("Done!", flush=True)

    ## Regression Calibration Metrics
    # In regression calibration, the most common metric is the Negative Log Likelihood (NLL) to measure the quality of a predicted probability distribution w.r.t. the ground-truth:
    # Negative Log Likelihood (NLL) (netcal.metrics.NLL)
    # The metrics Pinball Loss, Prediction Interval Coverage Probability (PICP), 
    # and Quantile Calibration Error (QCE) evaluate the estimated distributions by means of the predicted quantiles. 
    # For example, if a forecaster makes 100 predictions using a probability distribution for each estimate targeting the true ground-truth, 
    # we can measure the coverage of the ground-truth samples for a certain quantile level (e.g., 95% quantile). 
    # If the relative amount of ground-truth samples falling into a certain predicted quantile is above or below the specified quantile level, 
    # a forecaster is told to be miscalibrated in terms of quantile calibration. Appropriate metrics in this context are

    # Pinball Loss (netcal.metrics.PinballLoss)
    # Prediction Interval Coverage Probability (PICP) [14] (netcal.metrics.PICP)
    # Quantile Calibration Error (QCE) [15] (netcal.metrics.QCE)
    # Finally, if we work with normal distributions, we can measure the quality of the predicted variance/stddev estimates. 
    # For variance calibration, it is required that the predicted variance matches the observed error variance which is equivalent to then Mean Squared Error (MSE). 
    # Metrics for variance calibration are

    # Expected Normalized Calibration Error (ENCE) [17] (netcal.metrics.ENCE)
    # Uncertainty Calibration Error (UCE) [18] (netcal.metrics.UCE)

    # Calibration plot
    # 1. Compute histogram of observed thetas
    # 2. Compute histogram of posterior thetas
    # 3. For each confidence level, compute the fraction of posterior samples that fall within the confidence interval.

    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
    # https://github.com/EFS-OpenSource/calibration-framework/blob/main/examples/regression/multivariate/main.py

    print("Computing calibration metrics...", end=' ', flush=True)
    results = compute_calibration_metrics(
        posterior_samples, 
        thetas, 
        bins=config.bins,
        quantiles=np.linspace(
            config.start_quantiles,
            config.end_quantiles,
            config.num_quantiles
        )
    )
    print("Done!", flush=True)
    results["sbc"] = sbc_stats
    results["tarp"] = tarp_results

    print("Plotting calibration diagrams...", end=' ', flush=True)
    plot_calibration_diagrams(
        posterior_samples=posterior_samples,
        thetas=thetas,
        output_dir=output_dir,
        bins=config.bins,
        quantiles=np.linspace(
            config.start_quantiles,
            config.end_quantiles,
            config.num_quantiles
        )
    )
    print("Done!", flush=True)

    # save results to a pickle file
    print("Saving calibration results to pickle...", end=' ', flush=True)
    with open(output_dir / "calibration_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Done!", flush=True)

    del posterior_samples, thetas, test_dataset, experiment_config

    return results

def run_coverage_analysis(
    args: argparse.Namespace,
    config: CoverageConfig,
    output_dir: Path
) -> dict:
    """
    Perform coverage analysis for posterior samples and ground-truth thetas.
    Args:
        args: Command line arguments.
        config: Coverage configuration.
    Returns:
        A dictionary containing coverage analysis results.
    """

    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    print("Loading experiment config...", end=' ', flush=True)
    experiment_config = load_experiment_config(
        experiment_dir=args.experiment_dir
    )
    print("Done!", flush=True)

    # Load posterior samples
    print("Loading posterior samples...", end=' ', flush=True)
    posterior_samples = np.load(
        args.experiment_dir / 'posterior_distribution.npy'
    )
    print("Done!", flush=True)

    # Load thetas
    print("Loading test dataset...", end=' ', flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    thetas = test_dataset.get_parameters()
    print("Done!", flush=True)

    # perform coverage analysis
    print("Performing coverage analysis...", end=' ', flush=True)
    results = perform_coverage_analysis(
        posterior_samples=posterior_samples,
        thetas=thetas,
        output_dir=output_dir,
        labels=test_dataset.get_parameters_labels(),
        statistics=config.statistics,
        confidence_levels=config.confidence_levels,
        support=config.support,
    )
    print("Done!", flush=True)

    del posterior_samples, thetas, test_dataset, experiment_config

    return results


def run_log_probs_analysis(
    args: argparse.Namespace,
    config: LogProbsConfig,
    output_dir: Path
) -> dict:
    
    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load posterior samples
    print("Loading posterior log-probabilities...", end=' ', flush=True)
    posterior_log_probs = np.load(
        args.experiment_dir / 'posterior_log_probs.npy'
    )
    print("Done!", flush=True)

    # Load top log probs
    print("Loading top posterior log-probabilities...", end=' ', flush=True)
    top_log_probs = np.load(
        args.experiment_dir / 'posterior_top_log_probs.npy'
    )
    print("Done!", flush=True)

    # Load true theta log probs    
    print("Loading posterior log-probabilities of true thetas...", end=' ', flush=True)
    posterior_log_probs_true_theta = np.load(
        args.experiment_dir / 'posterior_log_probs_true_theta.npy'
    )
    print("Done!", flush=True)

    print("Saving log-probabilities to CSV...", end=' ', flush=True)
    df = save_log_probs_to_csv(
        log_probs=posterior_log_probs,
        top_log_probs=top_log_probs,
        log_probs_true_theta=posterior_log_probs_true_theta,
        output_dir=output_dir,
    )
    print("Done!", flush=True)

    return df


def draw_corner_plots(
    args: argparse.Namespace,
    config: DrawCornerPlotsConfig,
    output_dir: Path, 
) -> None:
    """
    Draw corner plots comparing prior and posterior distributions.
    Args:
        args: Command line arguments.
        config: Corner plots configuration.
        output_dir: Output directory for saving corner plots.
    Returns:
        None
    """

    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    print("Loading experiment config...", end=' ', flush=True)
    experiment_config = load_experiment_config(
        experiment_dir=args.experiment_dir
    )
    print("Done!", flush=True)
    
    # Load posterior samples
    print("Loading posterior samples...", end=' ', flush=True)
    posterior_samples = np.load(
        args.experiment_dir / 'posterior_distribution.npy'
    )
    print("Done!", flush=True)


    # Load thetas
    print("Loading test dataset...", end=' ', flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    thetas = test_dataset.get_parameters()
    print("Done!", flush=True)

    # Plot corner plots for individual posterior distributions
    num_corners = (
        config.max_plots
        if isinstance(config.max_plots, int)
        else len(posterior_samples)
    )
    with tqdm(total=num_corners, desc="Making corner plots...") as pbar:
        for idx, (p_samples, theta) in enumerate(zip(posterior_samples, thetas)):
            if idx >= num_corners:
                break
            corner_plot_single_distribution(
                posterior_samples=p_samples,
                theta=theta,
                labels=test_dataset.get_parameters_labels(),
                output_fname=output_dir / f"corner_{idx}",
                label_kwargs=config.label_kwargs,
                title_kwargs=config.title_kwargs,
                legend_kwargs=config.legend_kwargs,
                offset=config.offset,
            )
            pbar.update(1)
    
    # plot corner plot
    corner_plot_prior_posterior(
        posterior_samples, 
        thetas, 
        output_dir / "prior_vs_posteriors.png",
        labels=test_dataset.get_parameters_labels(),
        label_kwargs=config.label_kwargs,
        title_kwargs=config.title_kwargs,
        offset=config.offset,
    )



if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN EVALUATION\n")

    # Get the command line arguments and define shortcuts
    args = get_cli_arguments()

    # Make sure the working directory exists before we proceed
    if not args.experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.experiment_dir}")
    
    # Load the importance sampling config
    config = load_config(experiment_dir=args.experiment_dir)


    # -------------------------------------------------------------------------
    # Stage 1: Evaluate regression errors on test set
    # -------------------------------------------------------------------------

    if args.stage == "evaluate_regression_metrics" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Evaluate regression metrics on test set", flush=True)
        print(80 * "-" + "\n", flush=True)
        
        # Check if the output file already exists
        regression_output_dir = Path(
            args.experiment_dir / "evaluation" / "regression"
        )

        # Within this function, we compute:
        # - RMSE
        # - MAE
        # - Median AE
        # - MSE
        # We save the results to a file in the experiment directory (maybe into a 
        # dedicated directory), and we also print them to the console.
        results = run_regression_metrics(
            args=args,
            config=config.evaluate_regression_metrics,
            output_dir=regression_output_dir,
        )
        print("Done!\n\n")

    # -------------------------------------------------------------------------
    # Stage 2: Evaluate calibration errors on test set
    # -------------------------------------------------------------------------

    if args.stage == "evaluate_calibration_metrics" or args.stage is None:

        print(80 * "-", flush=True)
        print("(2) Evaluate calibration metrics on test set", flush=True)
        print(80 * "-" + "\n", flush=True)

        calibration_output_dir = Path(
            args.experiment_dir / "evaluation" / "calibration"
        )


        # Within this function, we compute:
        # - run_calibration_metrics
        #     - NLL
        #     - ENCE
        #     - UCE
        #     - QCE
        #     - Pinball Loss
        #     - Sharpness ?
        #     - ...
        # - plot_calibration_diagrams
        #     - reliability diagrams
        #     - uce plots
        # We save the results to a file in the experiment directory (maybe into a
        # dedicated directory), and we also print them to the console.
        results = run_calibration_metrics(
            args=args, 
            config=config.evaluate_calibration_metrics,
            output_dir=calibration_output_dir
        )

        print("Done!\n\n")
        
    # -------------------------------------------------------------------------
    # Stage 3: Evaluate coverage on test set
    # -------------------------------------------------------------------------

    if args.stage == "evaluate_coverage_metrics" or args.stage is None:

        print(80 * "-", flush=True)
        print("(3) Evaluate coverage metrics on test set", flush=True)
        print(80 * "-" + "\n", flush=True)

        coverage_output_dir = Path(
            args.experiment_dir / "evaluation" / "coverage"
        )

        # Within this function, we compute:
        # - run_coverage_evaluation
        #     - empirical coverage probabilities at different levels
        # We save the results to a file in the experiment directory (maybe into a
        # dedicated directory), and we also print them to the console.
        results = run_coverage_analysis(
            args=args, 
            config=config.evaluate_coverage_metrics,
            output_dir=coverage_output_dir
        )

        print("Done!\n\n")


    # -------------------------------------------------------------------------
    # Stage 4: Draw corner plots on test set
    # -------------------------------------------------------------------------
    if args.stage == "evaluate_log_probabilities" or args.stage is None:

        print(80 * "-", flush=True)
        print("(4) Evaluate log probabilities on test set", flush=True)
        print(80 * "-" + "\n", flush=True)

        log_probs_output_dir = Path(
            args.experiment_dir / "evaluation" / "log_probs"
        )

        # Within this function, we compute:
        results = run_log_probs_analysis(
            args=args, 
            config=config.evaluate_log_probs,
            output_dir=log_probs_output_dir
        )

        print("Done!\n\n")

    # -------------------------------------------------------------------------
    # Stage 5: Draw corner plots on test set
    # -------------------------------------------------------------------------
    if args.stage == "draw_corner_plots" or args.stage is None:

        print(80 * "-", flush=True)
        print("(5) Draw corner plots on test set", flush=True)
        print(80 * "-" + "\n", flush=True)

        corner_output_dir = Path(
            args.experiment_dir / "evaluation" / "corners"
        )

        # Within this function, we compute:
        # - draw_corner_plots
        draw_corner_plots(
            args=args, 
            config=config.draw_corner_plots,
            output_dir=corner_output_dir
        )
    
        print("Done!\n\n")

    


    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")

