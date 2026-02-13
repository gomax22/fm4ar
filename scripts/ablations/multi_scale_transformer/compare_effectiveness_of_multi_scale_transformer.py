import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

from time import time
from typing import Optional
from pathlib import Path
from yaml import safe_load
from tqdm import tqdm
from typing import List

import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from pprint import pprint
from fm4ar.utils.paths import expand_env_variables_in_path
mpl.rcParams['text.usetex'] = False 
mpl.rc('font',family='Times New Roman')


filter_columns = {
    'regression': {
        'mse_mean': 'MSE', 
        'mae_mean': 'MAE',
        'rmse_mean': 'RMSE',
        'median_ae_mean': 'MedAE'
    },
    'calibration': {
        'NLL_joint_mean': 'NLL',
        'QCE_joint_mean': 'QCE',
        'Pinball_mean': 'PL',
        'ENCE_joint_mean': 'ENCE',
        'UCE_joint_mean': 'UCE',
        'Sharpness_mean': 'Sharp.',
    },
    'coverage': {
        'accuracy_68_ratio': 'MCR\n(1s)',
        'accuracy_95_ratio': 'MCR\n(2s)', 
        'accuracy_99_ratio': 'MCR\n(3s)',
        'accuracy_support_ratio': 'MCR\n(S)',
        'accuracy_68_all': 'JCR\n(1s)',
        'accuracy_95_all': 'JCR\n(2s)',
        'accuracy_99_all': 'JCR\n(3s)',
        'accuracy_support_all': 'JCR\n(S)',
        
    },
    'log_probs': {
        'log_prob_true_thetas_mean': 'True LP',
        'top_log_prob_mean': 'Top LP',
        'mean_log_prob_samples_mean': 'Avg. Pred. LP',
    },
    'distribution_metrics': {
        'jsd_aggregate': 'JSD',
        'mmd_per_observation_mean': 'MMD',
    }
}



# ============================
#  Color system
# ============================
def make_estimator_palette(estimator_names):
    """
    Assign maximally separated, color-blind safe colors to estimators.
    """
    cmap = plt.get_cmap("tab20")
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(estimator_names))]
    return dict(zip(estimator_names, colors))


def fade_color(color, amount=0.45):
    """
    Lighten and desaturate a color (for performance drops).
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)

def measure_relative_performance(
    dfs: List[pd.DataFrame],
    filter_columns: Optional[dict[str, str]] = None
):
    # Assumes both dataframes have the same structure and indices
    baseline_df, comparison_df = dfs

    # Drop 'parameters' column (first column)
    if baseline_df.columns[0] in ['parameters', 'parameter']:
        baseline_df = baseline_df.drop(columns=baseline_df.columns[0])
        comparison_df = comparison_df.drop(columns=comparison_df.columns[0])
    
    relative_performance = (comparison_df - baseline_df) / baseline_df * 100

    if filter_columns is not None:
        relative_performance = relative_performance[list(filter_columns.keys())]

    # Average over all rows
    relative_performance = relative_performance.mean().to_frame().T
    return relative_performance


def load_experimental_results(
    config: dict,
    relative_file_path: str,
    description: str,
    warning_label: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Generic function to load CSV files from multiple experiments.

    Parameters
    ----------
    config : dict
        Configuration containing the list of experiments.
    relative_file_path : str
        Relative path to the CSV file inside each experiment directory.
    description : str
        Description used for tqdm and print messages.
    warning_label : str, optional
        Label used in warning messages to specify which file type is missing.
    
    Returns
    -------
    pd.DataFrame or None
        The merged DataFrame if successful, otherwise None.
    """
    all_dfs = []
    pbar = tqdm(config['estimators'], desc=f"Loading {description}")

    for estimator, experiment in config['estimators'].items():
        estimator_dfs = []
        for key, exp in experiment.items():
            exp_name = estimator.upper()
            results_path = expand_env_variables_in_path(exp['experiment-dir'])
            file_path = results_path / relative_file_path

            if not file_path.exists():
                label = warning_label or description
                print(f"\nWARNING: {label} file not found for {exp_name} at {file_path}")
                pbar.update(1)
                continue

            df = pd.read_csv(file_path)
            estimator_dfs.append(df)
        all_dfs.append(estimator_dfs)
        pbar.update(1)

    pbar.close()

    if not all_dfs:
        print(f"No valid files found for {description}. Skipping.")
        return None
    print("Done!", flush=True)

    return all_dfs


# ---- Specific helper wrappers ---- #
def load_regression_metrics_dap(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/regression/regression_metrics_dap.csv",
        description="regression metrics DAP"
    )


def load_regression_metrics_all_samples(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/regression/regression_metrics_all_samples.csv",
        description="regression metrics all samples"
    )

def load_sklearn_regression_metrics_dap(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/regression/sklearn_regression_metrics_dap.csv",
        description="sklearn regression metrics DAP"
    )

def load_sklearn_regression_metrics_all_samples(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/regression/sklearn_regression_metrics_all_samples.csv",
        description="sklearn regression metrics all samples"
    )

def load_calibration_metrics_summary(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/calibration/calibration_metrics_summary.csv",
        description="calibration metrics summary"
    )


def load_coverage_per_target(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/coverage/coverage_per_target.csv",
        description="coverage per target"
    )


def load_distribution_metrics(config: dict) -> Optional[pd.DataFrame]:
    jsd_aggregate_df = load_experimental_results(
        config=config,
        relative_file_path="evaluation/distribution/jsd_aggregate_summary.csv",
        description="jsd aggregate"
    )
    jsd_per_observation_df = load_experimental_results(
        config=config,
        relative_file_path="evaluation/distribution/jsd_per_observation_summary.csv",
        description="jsd per observation"
    )
    mmd_aggregate_df = load_experimental_results(
        config=config,
        relative_file_path="evaluation/distribution/mmd_summary.csv",
        description="mmd summary"
    )
    return jsd_aggregate_df, jsd_per_observation_df, mmd_aggregate_df

def load_log_probs_true_thetas(config: dict) -> Optional[pd.DataFrame]:
    return load_experimental_results(
        config=config,
        relative_file_path="evaluation/log-probs/log_probs_very_short_summary.csv",
        description="log probs true thetas"
    )


def plot_multiple_relative_performances(
    relative_performances: dict,
    output_dir: Path,
    tasks: List[str] = [
        'regression', 
        'calibration', 
        'coverage', 
        # 'log_probs', 
        'distribution_metrics'
    ],
    titles: dict = {
        'regression': 'Regression',
        'calibration': 'Calibration',
        'coverage': 'Coverage',
        # 'log_probs': 'Log Probabilities',
        'distribution_metrics': 'Distribution'
    }
):
    estimators = list(relative_performances.keys())
    n_estimators = len(estimators)

    # Get metric keys from first experiment & task
    first_est = estimators[0]
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, len(tasks), figsize=(28, 10))
    # fig, axs = plt.subplots(2, 2, figsize=(14, 14))

    width = 0.8 / n_estimators   # total bar width per metric
   
    # Build estimator color palette
    colors = make_estimator_palette(estimators)

    for task in tasks:
        ax = axs.flatten()[tasks.index(task)]
        species = [filter_columns[task][key] for key in list(relative_performances[first_est][task].keys())]
        x = np.arange(len(species))

        vs = []
        for i, estimator in enumerate(estimators):
            values = list(relative_performances[estimator][task].values())
            vs.extend(values)

            offsets = x - 0.4 + width/2 + i * width
            
            base_color = colors[estimator]
            base_colors, hatches = [], []
            
            # for j, value in enumerate(values):
            #     if value < 0:
            #         base_colors.append(base_color)
            #         hatches.append(None)
            #     else:
            #         base_colors.append(fade_color(base_color))
            #         hatches.append("///")

            if task in ['coverage']:
                for j, value in enumerate(values):
                    if value > 0:
                        base_colors.append(base_color)
                        hatches.append(None)
                    else:
                        base_colors.append(fade_color(base_color))
                        hatches.append("///")
            else:
                for j, value in enumerate(values):
                    if value < 0:
                        base_colors.append(base_color)
                        hatches.append(None)
                    else:
                        base_colors.append(fade_color(base_color))
                        hatches.append("///")

            ax.bar(
                offsets,
                values,
                width=width,
                color=base_colors,
                hatch=hatches,
                edgecolor="black",
                alpha=0.95
            )


        # Add grid lines
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.set_tick_params(labelsize=18)
        if max(vs) > 100:
            ax.set_yscale('log')

        # rects1 = ax.bar(species, values, width, label='w/o auxiliary data', color=colors)
        # ax.set_ylabel('Relative Performance (%)', fontsize=24)
        # ax.set_title(titles[task], fontsize=24)
        # ax.set_xticks(x, species, fontsize=20)
        # ax.set_yticks(fontsize=20)
        # ax.set_ylabel('Relative Performance (%)', fontsize=24, fontweight='bold')
        ax.set_title(titles[task], fontsize=24, fontweight='bold')
        ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=22, fontweight='bold')
        ax.set_xticks(x, species, fontsize=20, fontweight='bold')


        # ax.set_ylim(-np.abs(values).max() - 10, np.abs(values).max() + 10)
    axs[0].set_ylabel('Relative Performance (%)', fontsize=24, fontweight='bold')

    # ============================
    #  Legends
    # ============================

    # Estimator legend
    estimator_handles = [
        Patch(facecolor=colors[estimator], edgecolor="black", label=estimator)
        for estimator in estimators
    ]

    # Gain / drop legend
    gain_handle = Patch(
        facecolor="gray", 
        edgecolor="black", 
        label="Performance gain"
    )
    drop_handle = Patch(
        facecolor=fade_color("gray"),
        hatch="///",
        edgecolor="black",
        label="Performance drop"
    )

    fig.legend(
        handles=estimator_handles,
        ncols=len(estimator_handles),
        frameon=False,
        # fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        prop={"weight": "bold", "size" : 22}
    )

    fig.legend(
        handles=[gain_handle, drop_handle],
        ncols=2,
        frameon=False,
        # fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        prop={"weight": "bold", "size" : 22}
    )

    fig.tight_layout()
    # fig.subplots_adjust(top=0.9, bottom=0.1)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    # ============================
    #  Save
    # ============================

    fig.savefig(
        os.path.join(output_dir, "relative_performances_multi_scale_approach.pdf"),
        format="pdf", bbox_inches="tight", dpi=400
    )
    fig.savefig(
        os.path.join(output_dir, "relative_performances_multi_scale_approach.png"),
        format="png", bbox_inches="tight", dpi=400
    )

    plt.close(fig)





if __name__ == "__main__":
    script_start = time()
    print("\nRUN ABLATION STUDY ON THE EFFECTIVENESS OF MULTI SCALE APPROACH\n")

    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to the configuration file specifying the estimators to \
            compare."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        choices=[
            "regression",
            "calibration",
            "coverage",
            "log_probs",
            "distribution_metrics",
        ],
        default=None,
        help="Stage of the evaluation workflow that should be run.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Ensure output directory exists
    output_dir = expand_env_variables_in_path(args.config.parent / args.config.stem)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


    # Init a dict with an entry for each estimator
    model_labels = [k.upper() for k in list(config['estimators'].keys())]
    relative_performances = {estimator: {} for estimator in model_labels}
    # load colors from config
    # colors = [v['color'] for k,v in config['estimators'].items()]

    # -------------------------------------------------
    # Stage 1: Measure relative performance for regression results
    # -------------------------------------------------
    if args.tasks is None or args.tasks == "regression":

        print(80 * "-", flush=True)
        print("(1) Measure relative performance for regression results", flush=True)
        print(80 * "-" + "\n", flush=True)
        
        # regression_dap_dfs = load_regression_metrics_dap(config)
        regression_all_samples_dfs = load_regression_metrics_all_samples(config)
        # sklearn_regression_metrics_dap_dfs = load_sklearn_regression_metrics_dap(config)
        # sklearn_regression_metrics_all_samples_dfs = load_sklearn_regression_metrics_all_samples(config)

        #### HERE WE COMPUTE PERFORMANCE GAINS FOR REGRESSION
        for estimator, reg_dfs in zip(model_labels, regression_all_samples_dfs):
            regression_relative_perf = measure_relative_performance(
                dfs=reg_dfs,
                filter_columns=filter_columns['regression']
            )
            relative_performances[estimator]['regression'] = regression_relative_perf.iloc[0].to_dict()
        print("Done!\n\n")

    # -------------------------------------------------
    # Stage 2: Measure relative performance for calibration results
    # -------------------------------------------------
    if args.tasks is None or args.tasks == "calibration":
        print(80 * "-", flush=True)
        print("(2) Measure relative performance for calibration results", flush=True)
        print(80 * "-" + "\n", flush=True)
        
        calibration_dfs = load_calibration_metrics_summary(config)

        #### HERE WE COMPUTE PERFORMANCE GAINS FOR CALIBRATION
        for estimator, cal_dfs in zip(model_labels, calibration_dfs):
                
            calibration_relative_perf = measure_relative_performance(
                dfs=cal_dfs,
                filter_columns=filter_columns['calibration']
            )

            relative_performances[estimator]['calibration'] = calibration_relative_perf.iloc[0].to_dict()


        print("Done!\n\n")

    # -------------------------------------------------
    # Stage 3: Measure relative performance for coverage results
    # -------------------------------------------------
    if args.tasks is None or args.tasks == "coverage":
        print(80 * "-", flush=True)
        print("(3) Measure relative performance for coverage results", flush=True)
        print(80 * "-" + "\n", flush=True)

        coverage_dfs = load_coverage_per_target(config)


        # Extract 'ratio' row and 'all' row from each dataframe
        
        coverage_concat_dfs = []
        for dfs in coverage_dfs:
            concat_dfs = []
            for df in dfs:
                df = df.set_index('parameter')

                ratio_df = df.loc['ratio'].to_frame().T
                all_df = df.loc['all'].to_frame().T
                concat_df = pd.concat([ratio_df, all_df])

                concat_df = (
                    concat_df
                    .stack()                                      # move values into rows
                    .rename_axis(["suffix", "metric"])            # name the two index levels
                    .reset_index(name="value")
                    .assign(new_col=lambda x: x["metric"] + "_" + x["suffix"])
                    .pivot_table(values="value", columns="new_col", aggfunc="first")
                )

                concat_dfs.append(concat_df)
            coverage_concat_dfs.append(concat_dfs)


        #### HERE WE COMPUTE PERFORMANCE GAINS FOR COVERAGE
        for estimator, cc_dfs in zip(model_labels, coverage_concat_dfs):
            coverage_relative_perf = measure_relative_performance(
                dfs=cc_dfs,
                filter_columns=filter_columns['coverage']
            )
            relative_performances[estimator]['coverage'] = coverage_relative_perf.iloc[0].to_dict()

        print("Done!\n\n")


    # -------------------------------------------------
    # Stage 4: Measure relative performance for distribution metrics results
    # -------------------------------------------------
    if args.tasks is None or args.tasks == "distribution_metrics":
        print(80 * "-", flush=True)
        print("(4) Measure relative performance for distribution metrics results", flush=True)
        print(80 * "-" + "\n", flush=True)
        jsd_aggregate_dfs, _, mmd_aggregate_dfs = load_distribution_metrics(config)

        mmd_expanded_dfs = []
        for jsd_dfs, mmd_dfs in zip(jsd_aggregate_dfs, mmd_aggregate_dfs):
            mmd_expanded_df = []    
            for jsd_df, mmd_df in zip(jsd_dfs, mmd_dfs):
                # Duplicate rows in MMD dataframe to match the number of rows in JSD dataframe
                if jsd_df is not None and mmd_df is not None:
                    if len(jsd_df) != len(mmd_df):
                        repeat_factor = len(jsd_df) // len(mmd_df)
                        mmd_expanded = pd.DataFrame(
                            np.repeat(mmd_df.values, repeat_factor, axis=0),
                            columns=mmd_df.columns
                        )
                    else:
                        mmd_expanded = mmd_df


                    # Add 'parameters' column to MMD dataframe if missing
                    if 'parameters' not in mmd_expanded.columns and 'parameters' in jsd_df.columns:
                        mmd_expanded.insert(0, 'parameters', jsd_df['parameters'])
                    mmd_expanded_df.append(mmd_expanded)
            mmd_expanded_dfs.append(mmd_expanded_df)
        
        # Merge the two dataframes on 'parameters' column
        distribution_metrics_dfs = []
        for jsd_dfs, mmd_dfs in zip(jsd_aggregate_dfs, mmd_expanded_dfs):
            merged_dfs = []
            for jsd_df, mmd_df in zip(jsd_dfs, mmd_dfs):
                if jsd_df is not None and mmd_df is not None:
                    merged_df = pd.merge(
                        jsd_df, 
                        mmd_df, 
                        on='parameters', 
                        # suffixes=('_jsd', '_mmd')
                    )
                    merged_dfs.append(merged_df)
            distribution_metrics_dfs.append(merged_dfs)

        #### HERE WE COMPUTE PERFORMANCE GAINS FOR DISTRIBUTION METRICS
        for estimator, dm_dfs in zip(model_labels, distribution_metrics_dfs):
                
            distribution_metrics_relative_perf = measure_relative_performance(
                dfs=dm_dfs,
                filter_columns=filter_columns['distribution_metrics']
            )
            relative_performances[estimator]['distribution_metrics'] = distribution_metrics_relative_perf.iloc[0].to_dict()


        print("Done!\n\n")

    # -------------------------------------------------
    # Stage 5: Measure relative performance for log probs results
    # -------------------------------------------------
    if args.tasks is None or args.tasks == "log_probs":
        print(80 * "-", flush=True)
        print("(5) Measure relative performance for log probs results", flush=True)
        print(80 * "-" + "\n", flush=True)
        log_probs_dfs = load_log_probs_true_thetas(config)

        #### HERE WE COMPUTE PERFORMANCE GAINS FOR LOG PROBS
        for estimator, lp_dfs in zip(model_labels, log_probs_dfs):
            log_probs_relative_perf = measure_relative_performance(
                dfs=lp_dfs,
                filter_columns=filter_columns['log_probs']
            )
            relative_performances[estimator]['log_probs'] = log_probs_relative_perf.iloc[0].to_dict()

        print("Done!\n\n")


    pprint(relative_performances)
    # -------------------------------------------------
    # Stage 6: Plot relative performance gains
    # -------------------------------------------------
        
    print(80 * "-", flush=True)
    print("(6) Plot relative performance gains", flush=True)
    print(80 * "-" + "\n", flush=True)
    plot_multiple_relative_performances(
        relative_performances=relative_performances,
        output_dir=output_dir
    )

    # Print the total runtime
    print(f"This took {time() - script_start:.1f} seconds!\n")
