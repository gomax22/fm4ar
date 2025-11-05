"""
Utility functions for working with HDF5 files.
"""

from pathlib import Path
from typing import Sequence, Iterable

import h5py
import numpy as np
from tqdm import tqdm


def save_to_hdf(
    file_path: Path,
    **kwargs: np.ndarray,
) -> None:
    """
    Save the given arrays to an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.
        kwargs: Arrays to save.
    """

    # Ensure that the file exists and is empty
    with h5py.File(file_path, "w") as _:
        pass

    # Save the arrays to the HDF5 file (one by one)
    for key, value in kwargs.items():
        with h5py.File(file_path, "a") as f:
            f.create_dataset(name=key, data=value, dtype=value.dtype)


def load_from_hdf(
    file_path: Path,
    keys: list[str] | None = None,
    idx: int | slice | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Load the given keys from an HDF file.

    Args:
        file_path: Path to the HDF file.
        keys: Keys of the arrays to load. If None, load all arrays.
        idx: Indices of the arrays to load. If None, load all indices.

    Returns:
        data: Loaded arrays.
    """

    data = {}
    with h5py.File(file_path, "r") as f:
        if keys is None:
            keys = sorted(list(f.keys()))
        for key in keys:
            if key not in list(f.keys()):
                print(f"Warning: Key '{key}' not found in HDF file!")
                data[key] = np.empty(shape=())
                continue
            if idx is None:
                data[key] = np.array(f[key], dtype=f[key].dtype)
            else:
                data[key] = np.array(f[key][idx], dtype=f[key].dtype)

    return data



def merge_hdf_files(
    target_dir: Path,
    name_pattern: str,
    output_file_path: Path,
    keys: list[str] | None = None,
    singleton_keys: Sequence[str] = ("wlen",),
    delete_after_merge: bool = False,
    show_progressbar: bool = False,
    axis: int = 0,
) -> None:
    """
    Merge the HDF files in `target_dir` matching `name_pattern` into `output_file_path`.

    Arrays are concatenated along `axis`. Singleton keys (same across files)
    are copied from the first file.

    Args:
        target_dir: Directory containing the HDF files.
        name_pattern: Glob pattern for files to merge (e.g. "seed-*.hdf").
        output_file_path: Path to write merged HDF file.
        keys: Optional list of keys to merge. If None, discover non-singleton keys
              from the first file.
        singleton_keys: Keys copied from the first file (not concatenated).
        delete_after_merge: If True, delete input files after successful merge.
        show_progressbar: Whether to show a tqdm progress bar over files.
        axis: Axis along which to concatenate (0-based).
    """

    # Collect source HDF files
    all_paths = sorted(target_dir.glob(name_pattern))
    if len(all_paths) == 0:
        print("No files to merge.")
        return

    # Ensure singleton_keys is a set for quick membership tests
    singleton_set = set(singleton_keys)

    # Inspect first file to determine keys/shapes/dtypes if keys is None
    first_path = all_paths[0]
    with h5py.File(first_path, "r") as f0:
        available_keys = list(f0.keys())

        # If keys provided, validate they exist in the first file (warn if not)
        if keys is None:
            # All keys except singleton ones are candidates for concatenation
            keys_to_merge = [k for k in available_keys if k not in singleton_set]
        else:
            # Use only keys that are in the first file (warn about missing)
            missing = [k for k in keys if k not in available_keys and k not in singleton_set]
            if missing:
                print(f"Warning: The following requested keys are not present in the first file and will be skipped: {missing}")
            keys_to_merge = [k for k in keys if k in available_keys and k not in singleton_set]

        # Record reference shapes and dtypes for keys_to_merge
        keys_shapes_dtypes = {}
        for k in keys_to_merge:
            ds = f0[k]
            keys_shapes_dtypes[k] = (tuple(ds.shape), ds.dtype)

        # For singleton keys, ensure they exist before attempting to copy later
        singleton_present = [k for k in singleton_keys if k in f0]

    # Prepare (empty) output file and create datasets
    # We'll create datasets with axis-size 0 and maxshape allowing growth along axis
    with h5py.File(output_file_path, "w") as out_f:
        # Copy singleton datasets from the first file (if present)
        with h5py.File(first_path, "r") as f0:
            for k in singleton_present:
                out_f.create_dataset(name=k, data=f0[k][...], dtype=f0[k].dtype)

        # Create empty expandable datasets for merge keys
        for key, (shape, dtype) in keys_shapes_dtypes.items():
            # compute initial shape with axis dim zero
            init_shape = list(shape)
            if axis < 0 or axis >= len(shape):
                raise ValueError(f"Axis {axis} is out of bounds for dataset '{key}' with shape {shape}")
            init_shape[axis] = 0
            maxshape = list(shape)
            maxshape[axis] = None  # None indicates unlimited in h5py
            out_f.create_dataset(
                name=key,
                shape=tuple(init_shape),
                maxshape=tuple(maxshape),
                dtype=dtype,
            )

    # Iterate all files and append their data into the output datasets
    paths_iter: Iterable[Path] = tqdm(all_paths, unit=" files", ncols=80, disable=not show_progressbar)

    for p in paths_iter:
        with h5py.File(p, "r") as src, h5py.File(output_file_path, "a") as dst:
            for key, (ref_shape, ref_dtype) in keys_shapes_dtypes.items():
                if key not in src:
                    print(f"Warning: Key '{key}' not found in {p}; skipping.")
                    continue

                src_ds = src[key]
                # Load the entire array for this key from the source file.
                # If arrays are huge you can replace this with chunked reads.
                value = src_ds[...]
                if value.size == 0:
                    # skip empty arrays
                    continue

                # Check dimensionality compatibility
                if value.ndim != len(ref_shape):
                    raise RuntimeError(
                        f"Dimensionality mismatch for key '{key}' in file {p}: "
                        f"expected {len(ref_shape)} dims (reference {ref_shape}), got {value.shape}"
                    )

                # Check that all axes except the concatenation axis match the reference
                for ax in range(len(ref_shape)):
                    if ax == axis:
                        continue
                    if value.shape[ax] != ref_shape[ax]:
                        raise RuntimeError(
                            f"Shape mismatch on non-concatenation axis for key '{key}' in file {p}: "
                            f"axis {ax} expected size {ref_shape[ax]} but got {value.shape[ax]}"
                        )

                # Now append: resize and write into the newly allocated slice
                dst_ds = dst[key]
                current_size = dst_ds.shape[axis]
                add_size = value.shape[axis]
                new_size = current_size + add_size
                # Resize dataset along the concatenation axis
                new_shape = list(dst_ds.shape)
                new_shape[axis] = new_size
                dst_ds.resize(tuple(new_shape))

                # Build a tuple of slices selecting the region to write
                write_slice = [slice(None)] * value.ndim
                write_slice[axis] = slice(current_size, new_size)
                dst_ds[tuple(write_slice)] = value

    # Optionally delete source files (do after closing handles)
    if delete_after_merge:
        for p in all_paths:
            try:
                p.unlink()
            except Exception as e:
                print(f"Warning: could not delete {p}: {e}")
