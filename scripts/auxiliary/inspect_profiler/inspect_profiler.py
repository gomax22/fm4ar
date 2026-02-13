import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pprint import pprint
from collections import defaultdict
from tqdm import tqdm

from fm4ar.utils.nfe import NFEProfiler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Inspect profiler output files generated during sampling."
    )
    ap.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the directory with the trained model.",
    )
    ap.add_argument(
        "--n-bins",
        type=int,
        default=11,
        help="Number of bins for binning time step distributions.",
    )
    args = ap.parse_args()
    experiment_dir = args.experiment_dir
    n_bins = args.n_bins
    output_dir = experiment_dir / "plots"

    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load profiler output
    with open(experiment_dir / "nfe-profile.pkl", "rb") as f:
        profiler_output = pickle.load(f)

    # TODO: add ODESolver information to NFEProfiler class
    

    # Create NFEProfiler from loaded history
    profiler = NFEProfiler.from_history(profiler_output)
    
    # Print the following information:
    # 1. Total number of function calls (NFEs) per each method
    pprint(profiler.summary()) # effective time_sec should be divided by the number of jobs 

    # 2.1 Average number of NFEs per call
    summary_dict = defaultdict(lambda: [])
    for batch in profiler.history:
        key = batch["profile"][0]["key"]
        summary_dict[key].append(len(batch["profile"]))

    for key in summary_dict:
        summary_dict[key] = np.mean(summary_dict[key])

    summary_dict = dict(summary_dict)
    print("Average number of NFEs per call:")
    pprint(summary_dict)

    # 3. Average number of seconds per call
    summary_dict = defaultdict(lambda: [])
    for batch in profiler.history:
        for entry in batch["profile"]:
            key = entry["key"]
            duration = entry["duration"]
            summary_dict[key].append(duration)
    for key in summary_dict:
        summary_dict[key] = np.mean(summary_dict[key])
    summary_dict = dict(summary_dict)
    print("Average number of seconds per call:")
    pprint(summary_dict)


    # 4. Distribution of time steps 
    summary_dict = defaultdict(lambda: defaultdict(lambda: 0))
    pbar = tqdm(total=len(profiler.history), desc="Collecting time step distributions")
    for batch in profiler.history:
        for entry in batch["profile"]:
            key = entry["key"]
            t = entry["t"]
            summary_dict[key][t] += 1
        pbar.update(1)
    pbar.close()

    summary_dict = dict(summary_dict)
    
    # Normalize 
    pbar = tqdm(total=len(summary_dict), desc="Normalizing time step distributions")
    for key in summary_dict:
        total_counts = sum([summary_dict[key][t] for t in summary_dict[key].keys()])   # should be equal to number of NFEs for this key
        for t in summary_dict[key]:
            summary_dict[key][t] /= total_counts
        pbar.update(1)
    pbar.close()

    for key in summary_dict:
        ts = sorted(summary_dict[key].keys())
        freqs = [summary_dict[key][t] for t in ts]
        fig, ax = plt.subplots()
        grid = np.linspace(0, 1, len(ts))
        ax.plot(grid, grid, color='black', linestyle='dashed', lw=1.0, label='Uniform Distribution')
        ax.plot(grid, ts, color='red', linestyle='solid', lw=1.0,  label='Observed Distribution')
        ax.set_title(f"Time Step Distribution for Key: {key}")
        ax.set_xlabel("Expected Time Frequency")
        ax.set_ylabel("Observed Time Frequency")
        ax.grid(alpha=0.5)
        ax.legend()
        fig.savefig(
            output_dir / f"time_step_distribution_{key.replace('/', '_')}.png"
        )   
        plt.close(fig)

    # 5. More plots
    summary_dict = defaultdict(lambda: [])
    pbar = tqdm(total=len(profiler.history), desc="Collecting trajectories")
    for batch in profiler.history:
        key = batch["profile"][0]["key"]
        summary_dict[key].append([batch['profile'][idx]['t'] for idx in range(len(batch['profile']))])
        pbar.update(1)
    pbar.close()


    Path(output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    for key in summary_dict:
        pbar = tqdm(total=len(summary_dict[key][:10]), desc=f"Plotting trajectories for {key}")
        for idx, traj in enumerate(summary_dict[key][:10]):
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            # print(traj)

            ax.plot(traj, color='purple', lw=1.0, label='Adaptive-step Trajectories')

            ax.set_title(f"Trajectories for Key: {key}")
            ax.set_xlabel("#Timesteps")
            ax.set_ylabel("Trajectory")
            ax.grid(alpha=0.5)
            plt.show()

            # grid = np.linspace(0, 1, 10)
            # min_traj_length = min([len(traj) for traj in summary_dict[key]])

            # plot the uniform distribution for reference
            # grid = np.linspace(0, , len(traj))
            ax.plot(
                np.linspace(0, len(traj), 10), 
                np.linspace(0, 1, 10), color='black', linestyle='dashed', lw=1.0, label='Fixed-step Trajectory')
            # ax.axvline(x=min_traj_length, color='black', linestyle='dashed', lw=1.0, label='Min Trajectory Length')



            handles = [
                plt.Line2D([0], [0], color='black', lw=1.0, linestyle='dashed', label='Fixed-step Trajectory'),
                plt.Line2D([0], [0], color='purple', lw=1.0, linestyle='solid', label='Adaptive-step Trajectories'),
            ]
            ax.legend(handles=handles)
            fig.savefig(
                output_dir / "trajectories" / f"trajectories_{key.replace('/', '_')}_{idx}.png"
            )   
            plt.close(fig)



            pbar.update(1)
        pbar.close()
 



    # For each key, collect all time steps
    summary_dict = defaultdict(lambda: defaultdict(lambda: 0))
    time_steps_dict = defaultdict(list)
    pbar = tqdm(total=len(profiler.history), desc="Collecting time steps")
    for batch in profiler.history:
        for entry in batch["profile"]:
            key = entry["key"]
            t = entry["t"]
            time_steps_dict[key].append(t)
            summary_dict[key][t] += 1
        pbar.update(1)
    pbar.close()

    # Now assign each time step to bins, don't use np.histogram 
    # so that we can get std deviation across different bins
    binned_time_steps_dict = defaultdict(lambda: defaultdict(list))
    pbar = tqdm(total=len(time_steps_dict), desc="Binning time steps")
    for key in time_steps_dict:

        # Define bins for histograms
        # bins = np.linspace(0, 1, len(summary_dict[key]))  # 100 bins between 0 and 1
        bins = np.linspace(
            min(list(summary_dict[key].keys())),
            max(list(summary_dict[key].keys())),
            n_bins
        )
        for t in time_steps_dict[key]:
            bin_idx = np.digitize(t, bins) - 1  # digitize returns 1-based index
            binned_time_steps_dict[key][bin_idx].append(t)
        pbar.update(1)
    pbar.close()

    # Plot time step curves with error bars
    for key in binned_time_steps_dict:
        means = []
        stds = []
        for bin_idx in sorted(binned_time_steps_dict[key].keys()):
            mean = np.mean(binned_time_steps_dict[key][bin_idx])
            std = np.std(binned_time_steps_dict[key][bin_idx])
            means.append(mean)
            stds.append(std)

        fig, ax = plt.subplots()

        grid = np.linspace(0, 1, len(means))
        means = np.array(means)
        stds = np.array(stds)

        ax.fill_between(
            grid, 
            means - 3 * stds, 
            means + 3 * stds, 
            color='red', 
            label='3σ (99.7%)',
            alpha=0.1, 
            zorder=3
        )
        ax.fill_between(
            grid, 
            means - 2 * stds, 
            means + 2 * stds, 
            color='red', 
            label='2σ (95.4%)',
            alpha=0.3, 
            zorder=3
        )
        ax.fill_between(
            grid, 
            means - stds, 
            means + stds, 
            color='red', 
            label='1σ (68.2%)',
            alpha=0.5, 
            zorder=3
        )
        ax.plot(
            grid, 
            grid, 
            color='black', 
            linestyle='dashed', 
            lw=1.0, 
            label='Uniform Distribution',
            zorder=1)
        ax.plot(
            grid, 
            means, 
            color='red',
            linestyle='solid',
            lw=1.0,
            label='Mean Time Step',
            zorder=2
        )
        ax.set_title(f"Time Steps with Error Bars for Key: {key}")
        ax.set_xlabel("Expected Time Frequency")
        ax.set_ylabel("Observed Time Frequency")
        ax.grid(alpha=0.5)

        handles = [
            plt.Line2D([0], [0], color='black', lw=1.0, linestyle='dashed', label='Uniform Distribution'),
            plt.Line2D([0], [0], color='red', lw=1.0, linestyle='solid', label='Observed Distribution'),
            plt.Rectangle((0,0),1,1, color='red', alpha=0.5, label='1σ (68.2%)'),
            plt.Rectangle((0,0),1,1, color='red', alpha=0.3, label='2σ (95.4%)'),
            plt.Rectangle((0,0),1,1, color='red', alpha=0.1, label='3σ (99.7%)'),            
        ]
        ax.legend(handles=handles)
        fig.savefig(
            output_dir / f"time_steps_error_bars_{key.replace('/', '_')}.png"
        )   
        plt.close(fig)