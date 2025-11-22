import os
from pathlib import Path
from typing import Iterable


# -----------------------------------------------------------------------------
# Function evaluation counter utilities
# -----------------------------------------------------------------------------

class NFECounter:
    """Utility class to track number of function evaluations (NFE)."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def tick(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self.counts.clear()
        else:
            self.counts[name] = 0

    def __getitem__(self, name: str) -> int:
        return self.counts.get(name, 0)

    def __repr__(self) -> str:
        return f"NFE({self.counts})"


def count_nfe(counter_name: str):
    """
    Decorator that increments the given NFE counter before executing the function.
    Assumes 'self.nfe' exists.
    """
    def decorator(fn):
        def wrapped(self, *args, **kwargs):
            self.nfe.tick(counter_name)
            return fn(self, *args, **kwargs)
        return wrapped
    return decorator


class NFEProfiler:
    """
    Tracks neural function evaluations (NFEs) during ODE solving.
    """

    def __init__(self):
        
        self.nfe = NFECounter()
        self.history = []  
        self.current_batch_idx = 0
        self._current_batch_history = []

    @classmethod
    def from_history(cls, history: list[dict]):
        """
        Create a new NFEProfiler instance from an existing history.

        Args:
            history: List of batch histories to initialize the profiler with.
        
        Returns:
            profiler: NFEProfiler instance with the given history.
        """
        profiler = cls()
        profiler.history = history
        profiler.current_batch_idx = len(history)
        profiler._current_batch_history = []
        profiler.nfe.reset()
        return profiler

    def reset(self):
        """
        Reset profiler for a new batch/chunk.
        """
        if self._current_batch_history:
            self.history.append({
                "batch_idx": self.current_batch_idx,
                "profile": self._current_batch_history
            })
            self.current_batch_idx += 1

        self._current_batch_history = []
        self.nfe.reset()

    def log_event(self, step: int, t: float, key: str, duration: float):
        """Log one RHS evaluation within the current batch."""
        self._current_batch_history.append({
            "step": step,
            "t": t,
            "key": key,
            "duration": duration
        })

    def finalize(self):
        """Store the last batch if it hasn't been stored yet."""
        if self._current_batch_history:
            self.history.append({
                "batch_idx": self.current_batch_idx,
                "profile": self._current_batch_history
            })
            self._current_batch_history = []

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Return NFEs and compute time summarized per key.
        Output format:
            {
                key1: {"NFEs": ..., "time_sec": ...},
                key2: {"NFEs": ..., "time_sec": ...},
                ...
            }
        """
        from collections import defaultdict

        self.finalize()

        summary_dict = defaultdict(lambda: {
            "NFEs": 0, 
            "time_sec": 0.0,
        })

        for batch in self.history:
            for entry in batch["profile"]:
                key = entry["key"]
                duration = entry["duration"]
                summary_dict[key]["NFEs"] += 1

                summary_dict[key]["time_sec"] += duration

        return dict(summary_dict)


    def export(self, file_path: str, format: str = "pickle"):
        """Export profiler data per GPU/process."""
        self.finalize()
        if format == "pickle":
            import pickle
            with open(file_path, "wb") as f:
                pickle.dump(self.history, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")


    def __repr__(self):
        return f"NFEProfiler(current_batch={self.current_batch_idx}, events={len(self._current_batch_history)})"


def merge_histories(
    target_dir: Path,
    name_pattern: str,
    output_file_path: Path,
    reindex_batches: bool = True,
    delete_after_merge: bool = False,
    show_progressbar: bool = False,
):
    """
    Merge NFE profiler histories from multiple files into a single file.

    Args:
        target_dir: Directory containing the profiler files.
        name_pattern: Pattern to match profiler files (e.g., "profiler-*.pkl").
        output_file_path: Path to save the merged profiler history.
        reindex_batches: Whether to reindex batch indices in the merged history.
        delete_after_merge: Whether to delete individual files after merging.
        show_progressbar: Whether to show a progress bar during merging.
    """
    import pickle
    from tqdm import tqdm

    all_histories = []
    all_paths = sorted(target_dir.glob(name_pattern))
    if len(all_paths) == 0:
        print("No files to merge.")
        return
    
    paths_iter: Iterable[Path] = tqdm(all_paths, unit=" files", ncols=80, disable=not show_progressbar)


    for fp in paths_iter:
        with open(fp, "rb") as f:
            history = pickle.load(f)
            all_histories.extend(history)

    if reindex_batches:
        for new_idx, batch in enumerate(all_histories):
            batch["batch_idx"] = new_idx

    # Save merged history
    with open(output_file_path, "wb") as f:
        pickle.dump(all_histories, f)

    # Optionally delete source files (do after closing handles)
    if delete_after_merge:
        for fp in all_paths:
            try:
                fp.unlink()
            except Exception as e:
                print(f"Warning: Could not delete file {fp}: {e}")

    return all_histories
