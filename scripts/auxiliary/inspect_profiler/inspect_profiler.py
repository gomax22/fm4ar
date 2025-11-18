import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pprint import pprint
from collections import defaultdict, Counter

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
    args = ap.parse_args()
    experiment_dir = args.experiment_dir
    output_dir = experiment_dir / "plots"

    # Ensure output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load profiler output
    with open(experiment_dir / "nfe-profile.pkl", "rb") as f:
        profiler_output = pickle.load(f)

    # Create NFEProfiler from loaded history
    profiler = NFEProfiler.from_history(profiler_output)
    
    # Print the following information:
    # 1. Total number of function calls (NFEs) per each method
    pprint(profiler.summary()) # effective time_sec should be divided by the number of jobs 

    # 2. Average number of NFEs per call
    summary_dict = defaultdict(lambda: [])
    for batch in profiler.history:
        for entry in batch["profile"]:
            key = entry["key"]
            summary_dict[key].append(1)

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
    for batch in profiler.history:
        for entry in batch["profile"]:
            key = entry["key"]
            t = entry["t"]
            summary_dict[key][t] += 1

    summary_dict = dict(summary_dict)
    
    # Normalize 
    for key in summary_dict:
        total_counts = sum([summary_dict[key][t] for t in summary_dict[key].keys()])   # should be equal to number of NFEs for this key
        for t in summary_dict[key]:
            summary_dict[key][t] /= total_counts

    for key in summary_dict:
        ts = sorted(summary_dict[key].keys())
        freqs = [summary_dict[key][t] for t in ts]
        fig, ax = plt.subplots()
        grid = np.linspace(0, 1, len(ts))
        ax.plot(grid, grid, color='black', linestyle='dashed', marker='.', label='Uniform Distribution')
        ax.plot(grid, freqs, color='red', linestyle='solid', marker='.', label='Observed Distribution')
        ax.set_title(f"Time Step Distribution (RK4) for Key: {key}")
        ax.set_xlabel("Expected Time Frequency")
        ax.set_ylabel("Observed Time Frequency")
        ax.grid(alpha=0.5)
        ax.legend()
        fig.savefig(
            output_dir / f"time_step_distribution_{key.replace('/', '_')}.png"
        )   
        plt.close(fig)
