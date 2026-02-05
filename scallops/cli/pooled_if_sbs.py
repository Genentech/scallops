"""
pooled_if_sbs: Pooled In-Situ Sequencing (pooled SBS) Module

This module defines a Python script or module for running a pooled in-situ sequencing (SBS) pipeline
in the context of single-cell analysis. The pipeline involves various stages such as spot detection,
reads calling, and merging of single-cell sequencing (SCS) and phenotype data.

"""

import argparse
import json
import logging
import os
import re
from typing import Literal

import anndata
import dask
import dask.array as da
import dask.dataframe as dd
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from dask.delayed import delayed
from matplotlib import pyplot as plt
from skimage.segmentation import expand_labels

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _write_image,
    cli_metadata,
    load_json,
)
from scallops.envir import SCALLOPS_BASE_ORDER
from scallops.features.constants import _metadata_columns_whitelist_str
from scallops.io import (
    _add_suffix,
    _create_subset_function,
    _get_fs_protocol,
    _images2fov,
    _set_up_experiment,
    _to_parquet,
    is_parquet_file,
    is_scallops_zarr,
)
from scallops.reads import (
    apply_channel_crosstalk_matrix,
    assign_barcodes_to_labels,
    barcode_to_prefix,
    channel_crosstalk_matrix,
    correct_mismatches,
    decode_max,
    merge_sbs_phenotype,
    peaks_to_bases,
    read_statistics,
)
from scallops.spots import (
    _find_peaks_deep,
    find_peaks,
    max_filter,
    normalize_base_intensities,
    peak_thresholds_from_bases,
    peak_thresholds_from_reads,
    std,
    transform_log,
)
from scallops.utils import _fix_json
from scallops.visualize.crosstalk import pairwise_channel_scatter_plot
from scallops.xr import _z_projection
from scallops.zarr_io import (
    _get_fs,
    _get_sep,
    _get_store_path,
    _write_zarr_image,
    open_ome_zarr,
    read_ome_zarr_array,
)

logger = logging.getLogger("scallops")


@dask.delayed
def _crosstalk_plots(bases_dataset, dest):
    """Generate and save pairwise channel scatter plots for uncorrected and corrected intensities.

    :param bases_dataset: Dataset with intensity information.
    :param dest: Destination path to save the plots.
    """
    f, _ = pairwise_channel_scatter_plot(bases_dataset["uncorrected_intensity"])
    f.savefig(f"{dest}-uncorrected.png")
    plt.close(f)

    if "corrected_intensity" in bases_dataset:
        f, _ = pairwise_channel_scatter_plot(bases_dataset["corrected_intensity"])
        f.savefig(f"{dest}-corrected.png")
        plt.close(f)


def _read_qc_stats(df_reads: pd.DataFrame, df_labels: pd.DataFrame) -> dict:
    """Calculate and return quality control (QC) statistics for read and label recovery.

    This function computes QC metrics to assess the recovery of labeled regions based on
    the barcode counts or intensities. It helps evaluate the effectiveness of the sequencing
    and label assignment process.

    :param df_reads: DataFrame containing read data with barcode assignments.
                     The DataFrame should include a 'label' column.
    :param df_labels: DataFrame containing label data with barcode statistics.
                      This DataFrame should contain either `barcode_count` or
                      `barcode_intensity` columns for evaluation.

    :return: A dictionary with the following QC statistics:
        - `labels_recovered`: Total number of labels with significant barcode counts/intensities.
        - `fraction_labels_recovered`: Fraction of total labels that are recovered.
        - `fraction_total_labels_recovered`: Fraction of recovered labels out of all unique labels.
    """
    # Compute general read statistics
    qc_stats = read_statistics(df_reads)

    # Determine the number of recovered labels based on barcode data
    n_recovered = (
        df_labels.query("barcode_count_0 / barcode_count > 0.5").shape[0]
        if "barcode_count_0" in df_labels.columns
        else df_labels.query("barcode_intensity_0 / barcode_intensity > 0.5").shape[0]
    )

    # Populate QC statistics dictionary with recovery metrics
    qc_stats["labels_recovered"] = n_recovered
    qc_stats["fraction_labels_recovered"] = n_recovered / df_labels.shape[0]

    # Calculate fraction of total labels recovered from the reads data
    qc_stats["fraction_total_labels_recovered"] = (
        n_recovered / df_reads.query("label > 0")["label"].nunique()
    )

    return qc_stats


def _peaks_to_bases(
    maxed: xr.DataArray,
    peaks: pd.DataFrame,
    labels: np.ndarray,
    labels_only: bool,
    bases: list[str],
) -> xr.DataArray:
    """Convert detected peaks into base calls and assign them to labels.

    This function converts peaks identified from spot detection into corresponding bases (e.g., 'A',
    'T', 'C', 'G'), using the provided labels. Optionally, it can return only the labeled regions if
    specified.

    :param maxed: Max-filtered image as an xarray DataArray.
    :param peaks: Dask DataFrame containing the detected peaks.
    :param labels: Numpy array containing segmentation labels for the image.
    :param labels_only: If True, only the labeled regions will be processed.
    :param bases: List of base names (e.g., ["A", "T", "C", "G"]) used in sequencing.
    :return: DataArray containing the maxed values.
    """
    bases_array = peaks_to_bases(
        maxed=maxed, peaks=peaks, labels=labels, labels_only=labels_only, bases=bases
    )

    reorder_bases = os.environ.get(SCALLOPS_BASE_ORDER)
    if reorder_bases is not None:
        bases = reorder_bases.split(",")
        bases_array = bases_array.sel(c=bases)  # match order for median correction ties
        bases_array = bases_array.assign_coords(
            t=np.arange(1, 1 + len(bases_array.t))
        )  # match do not preserve t values
    return bases_array


def spot_detection_pipeline(
    image_tuple: tuple[tuple[str, ...], list[str], dict],
    iss_channels: list[int],
    output: str,
    max_filter_width: int,
    sigma_log: float | list[float],
    z_index: int | str,
    save_keys: tuple[str] | list[str] = ("max", "log", "std", "peaks"),
    output_image_format: str = "zarr",
    cycles: None | list[int] = None,
    qmin: float | None = None,
    qmax: float | None = None,
    eps: float = 1e-20,
    force: bool = False,
    peak_neighborhood_size: int = 5,
    chunks: tuple[int, int] | None = None,
    no_version: bool = False,
    spot_detection_method: Literal["log", "spotiflow", "u-fish", "piscis"] = "log",
    spot_detection_n_cycles: int | None = None,
    expected_cycles: int | None = None,
):
    """Run the spot detection pipeline.

    This function processes a set of images, performs spot detection, and saves the
    results to disk.

    :param image_tuple: A tuple containing information about the images.
    :param iss_channels: List of channel indices used for ISS sequencing.
    :param output: Root path to where the results will be stored.
    :param max_filter_width: Maximum filter width used in spot detection.
    :param z_index: Either 'max' or z-index
    :param sigma_log: Sigma parameter for log transformation in spot detection.
    :param save_keys: List of keys specifying which results to save.
    :param output_image_format: Output format for saved images.
    :param cycles: Optional list of cycle indices to process.
    :param qmin: Minimum quantile for normalization
    :param qmax: Maximum quantile for normalization
    :param eps: Small value added to the denominator for normalization
    :param force: Whether to overwrite existing output
    :param chunks: Tuple specifying chunking size for ISS image.
    :param peak_neighborhood_size: Peak neighborhood size
    :param no_version: Whether to skip version/CLI information in output.
    :param spot_detection_method: Spot detection method to use.
    :param spot_detection_n_cycles: Number of cycles to use for spot detection.
    :param expected_cycles: Expected number of cycles present.
    """
    _, file_list, metadata = image_tuple
    image_key = metadata["id"]
    output_fs = fsspec.url_to_fs(output)[0]
    output_sep = output_fs.sep
    output = output.rstrip(output_sep)
    points_path = f"{output}{output_sep}points"

    points_protocol = _get_fs_protocol(output_fs)
    if points_protocol != "file":
        points_path = f"{points_protocol}://{points_path}"
    peaks_path = f"{points_path}{output_sep}{image_key}-peaks.parquet"
    if not force:
        if is_parquet_file(peaks_path):
            logger.info(f"Skipping spot detection for {image_key}")
            return
    image = _images2fov(file_list, metadata, dask=True)
    image = _z_projection(image, z_index)
    if expected_cycles is not None:
        assert expected_cycles == image.sizes["t"], (
            f"Expected {expected_cycles} cycles, got {image.sizes['t']}"
        )
    if cycles is not None:
        image = image.isel(t=cycles)
    image_metadata = metadata["group_metadata"]["group"]
    match_ops_image_scale = os.environ.get("SCALLOPS_IMAGE_SCALE") == "1"
    if not match_ops_image_scale:
        image = image.isel(c=iss_channels)
    image = (
        image.chunk({"y": chunks[0], "x": chunks[1], "t": "auto", "c": "auto"})
        if chunks is not None
        else image.chunk({"t": "auto", "c": "auto"})
    )
    logger.info(f"Running spot detection for {image_key}.")
    if qmin is not None or qmax is not None:
        image.data = normalize_base_intensities(
            image.data, qmin=qmin, qmax=qmax, eps=eps
        )

    if match_ops_image_scale:
        # include DAPI channel to match ops rescaling
        loged = transform_log(image, sigma=sigma_log)
        loged = loged.isel(c=iss_channels)
    else:
        loged = transform_log(image, sigma=sigma_log)

    maxed = max_filter(loged, max_filter_width)
    std_arr = None
    cycles_spot_detection = (
        None if spot_detection_n_cycles is None else np.arange(spot_detection_n_cycles)
    )
    if spot_detection_method != "log":
        peaks = _find_peaks_deep(
            image.isel(t=cycles_spot_detection)
            if cycles_spot_detection is not None
            else image,
            peak_neighborhood_size,
            method=spot_detection_method,
        )
    else:
        std_arr = std(
            loged.isel(t=cycles_spot_detection)
            if cycles_spot_detection is not None
            else loged
        )
        peaks = find_peaks(std_arr, peak_neighborhood_size)
    dask_delayed = []
    compute = True
    metadata = cli_metadata() if not no_version else dict()
    metadata["image_metadata"] = image_metadata
    if "log" in save_keys:
        loged.attrs.update(metadata)
        dask_delayed.append(
            _write_image(
                name=f"{image_key}-log",
                root=open_ome_zarr(output, mode="a"),
                image=loged,
                output_format=output_image_format,
                zarr_format="zarr",
                compute=compute,
            )
        )
    else:
        del loged
    if "std" in save_keys and std_arr is not None:
        std_arr.attrs.update(metadata)
        dask_delayed.append(
            _write_image(
                name=f"{image_key}-std",
                root=open_ome_zarr(output, mode="a"),
                image=std_arr,
                output_format=output_image_format,
                metadata=dict(parent=image_key),
                compute=compute,
            )
        )
    else:
        del std_arr
    if "max" in save_keys:
        maxed.attrs.update(metadata)
        dask_delayed.append(
            _write_image(
                name=f"{image_key}-max",
                root=open_ome_zarr(output, mode="a"),
                image=maxed,
                output_format=output_image_format,
                zarr_format="zarr",
                compute=compute,
            )
        )
    else:
        del maxed
    if "peaks" in save_keys:
        output_fs.makedirs(points_path, exist_ok=True)

        if output_fs.exists(peaks_path):
            output_fs.rm(peaks_path, recursive=True)

        dask_delayed.append(
            _to_parquet(
                peaks,
                peaks_path,
                compute=compute,
                custom_metadata=dict(scallops=json.dumps(metadata)),
            )
        )
    if not compute and len(dask_delayed) > 0:
        dask.compute(*dask_delayed)


def _fix_cycles(sbs_cycles):
    """Fix or adjust SBS cycle numbering to ensure proper indexing.

    :param sbs_cycles: List of SBS cycles.
    :return: Adjusted list of SBS cycles.
    """
    if isinstance(sbs_cycles[0], str):
        sbs_cycles = np.arange(1, len(sbs_cycles) + 1)
    else:
        if sbs_cycles[0] == 0:
            # assume cycle numbering starts at 0 instead of 1
            for i in range(len(sbs_cycles)):
                sbs_cycles[i] = sbs_cycles[i] + 1
    logger.info(
        f"Timepoint indices (0-based): {', '.join([str(t - 1) for t in sbs_cycles])}"
    )
    return sbs_cycles


def _merged_to_matrix(
    merged_df: pd.DataFrame | dd.DataFrame,
    phenotype_paths: list[str],
    feature_names: list[str],
    feature_columns: list[list[str]],
    name: str,
    format: Literal["anndata", "xarray"] = "anndata",
) -> xr.DataArray | anndata.AnnData:
    assert format in {"anndata", "xarray"}
    if isinstance(merged_df, dd.DataFrame):
        merged_df = merged_df.compute()
    logger.info(f"Merging {len(merged_df):,} labels.")

    @delayed
    def read_values(url, index, columns):
        df = pd.read_parquet(url, columns=columns)
        values = df.values
        if index is not None:
            indices = index.get_indexer_for(df.index)
            keep = indices != -1
            values = values[keep]
            indices = indices[keep]
            aligned_values = np.full((len(index), values.shape[1]), np.nan)
            aligned_values[indices] = values
            return aligned_values
        return values.astype(np.float64, copy=False)

    arrays = []
    index_delayed = delayed(merged_df.index)
    for i in range(len(phenotype_paths)):
        url = phenotype_paths[i]
        columns = feature_columns[i]
        if len(columns) > 0:
            array = read_values(url, index_delayed, columns)
            arrays.append(
                da.from_delayed(
                    array,
                    shape=(len(merged_df.index), len(columns)),
                    dtype=np.float64,
                )
            )
    data = da.concatenate(arrays, axis=1).rechunk(("auto", 10))
    rename = dict()
    _replace_chars = " |-"
    for c in merged_df.columns:
        new_name = re.sub(_replace_chars, "_", c)
        if c != new_name:
            rename[c] = new_name
    if len(rename) > 0:
        merged_df = merged_df.rename(columns=rename)
    if format == "xarray":
        coords = dict(label=merged_df.index, feature=feature_names)
        for c in merged_df.columns:
            if c not in (
                "barcode_Q_0",
                "barcode_Q_1",
            ):  # FIXME need to specify encoding for zarr
                coords[c] = ("label", merged_df[c].values)
        return xr.DataArray(
            name=name, data=data, dims=["label", "feature"], coords=coords
        )
    else:
        obs = merged_df.reset_index(names="label")
        obs.index = obs.index.astype(str)
        for c in obs.columns:
            if pd.api.types.is_object_dtype(obs[c]):
                obs[c] = obs[c].astype(str)  # to save with anndata
        return anndata.AnnData(obs=obs, var=pd.DataFrame(index=feature_names), X=data)


def _rename_unique(columns, unique_values, prefix):
    rename = dict()
    replace_chars = " |-"
    for value in columns:
        new_value = value
        if value in unique_values:
            new_value = f"{value}_{prefix}"
            if new_value in unique_values:
                counter = 0
                new_value = f"{value}_{prefix}_{counter}"
                while new_value in unique_values:
                    counter += 1
                    new_value = f"{value}_{prefix}_{counter}"
        new_value = re.sub(replace_chars, "_", new_value)
        if value != new_value:
            rename[value] = new_value

    return rename


def merge_sbs_phenotype_pipeline(
    image_key: str,
    sbs_path: str,
    phenotype_paths: list[str],
    phenotype_suffix: list[str],
    df_barcode: pd.DataFrame,
    output_dir: str,
    join_sbs: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    join_phenotype: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    force: bool = False,
    no_version: bool = False,
    output_format: Literal["parquet", "zarr"] = "parquet",
):
    """Merge pooled SBS (labels output from reads) and phenotype data.

    :param image_key: Image identifier.
    :param sbs_path: SBS label assignments path in Parquet format.
    :param phenotype_paths: List of parquet paths containing phenotype data.
    :param phenotype_suffix: List of suffixes for phenotype columns.
    :param df_barcode: DataFrame containing barcode information.
    :param output_dir: Directory to save the merged results.
    :param join_sbs: Type of join to perform for SBS data.
    :param join_phenotype: Type of join to perform for phenotype data.
    :param force: Force overwriting of existing results.
    :param no_version: Whether to skip version/CLI information in output.
    :param output_format: Output file format.
    """

    output_file = f"{output_dir}{image_key}.{output_format}"
    if not force and (
        (output_format == "parquet" and is_parquet_file(output_file))
        or (output_format == "zarr" and is_scallops_zarr(output_file))
    ):
        logger.info(f"Skipping merge for {image_key}")
        return []
    paths_and_suffixes = []
    for i in range(len(phenotype_paths)):
        path = phenotype_paths[i]
        if phenotype_suffix is not None:
            path += f" ({phenotype_suffix[i]})"
        paths_and_suffixes.append(path)
    logger.info(f"Running merge for {image_key} with {', '.join(paths_and_suffixes)}.")
    fs, sbs_path_ = fsspec.core.url_to_fs(sbs_path)
    iss_dataset = pq.ParquetDataset(sbs_path_, filesystem=fs)
    image_metadata = None
    sbs_cycles = None
    if b"scallops" in iss_dataset.schema.metadata:
        iss_meta = json.loads(iss_dataset.schema.metadata[b"scallops"])
        sbs_cycles = iss_meta["sbs_cycles"]
        sbs_cycles = _fix_cycles(sbs_cycles)
        image_metadata = iss_meta.get("image_metadata")

    df_phenotypes = []
    # can have duplicate columns if features is called in multiple batches

    feature_names = []
    metadata_columns = []
    feature_columns = []  # used to read in subset of columns when merging to zarr
    # df_labels has 'mismatch', 'barcode_Q_mean', 'barcode_Q_min', 'barcode_peak', 'barcode_count', 'barcode_0', ...
    df_labels = dd.read_parquet(sbs_path)
    if "label" in df_labels.columns:
        df_labels = df_labels.set_index("label")
    if sbs_cycles is None:
        sbs_cycles = df_labels[["barcode_0"]].head()["barcode_0"].str.len().max()
        logger.info(f"ISS cycle metadata not found. Assuming {sbs_cycles} cycles.")
        sbs_cycles = np.arange(1, sbs_cycles + 1)
    prefixes = []
    unique_columns = set()
    unique_columns.update(df_barcode.columns.tolist())
    unique_columns.update(df_labels.columns.tolist())

    for i in range(len(phenotype_paths)):
        df = dd.read_parquet(phenotype_paths[i])
        _metadata_cols = df.columns[
            df.columns.str.contains(_metadata_columns_whitelist_str)
        ].tolist()
        _feature_cols = df.columns[
            ~df.columns.str.contains(_metadata_columns_whitelist_str)
        ].tolist()
        metadata_columns.append(_metadata_cols)
        feature_columns.append(_feature_cols)
        if phenotype_suffix is not None:
            df.columns = df.columns + phenotype_suffix[i]

        prefixes.append(phenotype_paths[i].split("/")[-3])

        if output_format == "zarr":  # read index and metadata
            if len(_metadata_cols) > 0:
                df = df.drop(_metadata_cols, axis=1)
            feature_names_i = df.columns.tolist()
            rename_features = _rename_unique(
                feature_names_i, unique_columns, prefixes[i]
            )
            for key in rename_features:
                feature_names_i[feature_names_i.index(key)] = rename_features[key]
            df = dd.read_parquet(phenotype_paths[i], columns=_metadata_cols)
            feature_names += feature_names_i
            unique_columns.update(feature_names_i)

        rename_cols = _rename_unique(df.columns, unique_columns, prefixes[i])
        if len(rename_cols) > 0:
            df = df.rename(columns=rename_cols)
        df_phenotypes.append(df)
        unique_columns.update(df.columns.tolist())

    df_labels, df_phenotypes = dask.compute(df_labels, df_phenotypes)

    if len(df_phenotypes) > 1:
        if isinstance(df_phenotypes[0], dd.DataFrame):
            df_phenotype = dd.concat(df_phenotypes, axis=1, join=join_phenotype)
        else:
            df_phenotype = pd.concat(df_phenotypes, axis=1, join=join_phenotype)
    else:
        df_phenotype = df_phenotypes[0]

    merged_df = merge_sbs_phenotype(
        df_labels=df_labels,
        df_phenotype=df_phenotype,
        df_barcode=df_barcode,
        sbs_cycles=sbs_cycles,
        how=join_sbs,
    )

    if image_metadata is not None:
        for col in image_metadata.keys():
            value = image_metadata[col]
            if col in unique_columns:
                counter = 0
                renamed_col = col + "_" + str(counter)
                while renamed_col in unique_columns:
                    counter += 1
                    renamed_col = col + "_" + str(counter)
                col = renamed_col
            unique_columns.add(col)
            merged_df[col] = value

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    if output_format == "zarr":
        data = _merged_to_matrix(
            merged_df=merged_df,
            phenotype_paths=phenotype_paths,
            feature_names=feature_names,
            feature_columns=feature_columns,
            name=image_key,
        )

        data.write_zarr(output_file, convert_strings_to_categoricals=False)
        store = zarr.open(output_file, mode="r+")
        store.attrs["scallops"] = _fix_json(metadata)

    elif isinstance(merged_df, pd.DataFrame):
        table = pa.Table.from_pandas(merged_df, preserve_index=True)
        table = table.replace_schema_metadata(
            {
                "scallops".encode(): json.dumps(metadata).encode(),
                **table.schema.metadata,
            }
        )
        fs, output_file = fsspec.url_to_fs(output_file)
        pq.write_table(
            table,
            output_file,
            filesystem=fs,
        )


def merge_main(arguments: argparse.Namespace):
    """Merge single-cell sequencing (SCS) and phenotype data.

    This function reads SCS and phenotype data, performs a merge, and saves the results.

    :param arguments: argparse namespace containing command-line arguments.
    """
    sbs = arguments.sbs
    dask_scheduler_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster)
        if arguments.dask_cluster is not None
        else _dask_workers_threads()
    )
    phenotype_paths = arguments.phenotype
    output_dir = arguments.output
    output_format = arguments.format
    join_phenotype = arguments.join_phenotype
    phenotype_suffix = arguments.phenotype_suffix
    if phenotype_suffix is not None:
        assert len(phenotype_paths) == len(phenotype_suffix), (
            "Length of phenotype and suffix must match"
        )
    join_sbs = arguments.join_sbs
    subset = arguments.subset
    force = arguments.force
    no_version = arguments.no_version
    sbs_fs, _ = fsspec.core.url_to_fs(sbs)

    sbs = sbs.rstrip(sbs_fs.sep)
    df_barcode = pd.read_csv(arguments.barcodes)
    barcode_column = arguments.barcode_col
    if barcode_column != "barcode":
        rename = dict()
        rename[barcode_column] = "barcode"
        df_barcode = df_barcode.rename(rename, axis=1)
    assert "barcode" in df_barcode.columns, (
        f"`barcode` column not found in {arguments.barcodes}"
    )
    phenotype_filesystems = []
    if len(set(phenotype_paths)) != len(phenotype_paths):
        raise ValueError("Duplicate phenotype paths")
    for i in range(len(phenotype_paths)):
        phenotype_fs, _ = fsspec.core.url_to_fs(phenotype_paths[i])
        phenotype_paths[i] = phenotype_paths[i].rstrip(phenotype_fs.sep)
        phenotype_filesystems.append(phenotype_fs)
    paths = []
    sbs_fs_protocol = _get_fs_protocol(sbs_fs)
    sbs_matches = sbs_fs.glob(sbs + sbs_fs.sep + "*.parquet")

    if sbs_fs_protocol != "file":
        sbs_matches = [f"{sbs_fs_protocol}://{m}" for m in sbs_matches]
    if subset is not None:
        subset = _create_subset_function(subset)
    for sbs_path in sbs_matches:
        name = os.path.splitext(os.path.basename(sbs_path))[0]
        if not name.startswith("."):  # ignore hidden files
            image_key, _ = os.path.splitext(name)
            if subset is None or subset(image_key):
                _phenotype_paths = []
                _phenotype_suffix = []
                for i in range(len(phenotype_paths)):
                    # match */A1-*.parquet and */A1.parquet
                    matches = phenotype_filesystems[i].glob(
                        f"{phenotype_paths[i]}{phenotype_filesystems[i].sep}*{phenotype_filesystems[i].sep}{name}-*.parquet"
                    ) + phenotype_filesystems[i].glob(
                        f"{phenotype_paths[i]}{phenotype_filesystems[i].sep}*{phenotype_filesystems[i].sep}{name}.parquet"
                    )

                    if len(matches) == 0:
                        # match A1-*.parquet and A1.parquet
                        matches = phenotype_filesystems[i].glob(
                            f"{phenotype_paths[i]}{phenotype_filesystems[i].sep}{name}-*.parquet"
                        ) + phenotype_filesystems[i].glob(
                            f"{phenotype_paths[i]}{phenotype_filesystems[i].sep}{name}.parquet"
                        )
                    pheno_fs_protocol = _get_fs_protocol(phenotype_filesystems[i])
                    if pheno_fs_protocol != "file":
                        matches = [f"{pheno_fs_protocol}://{m}" for m in matches]
                    for x in matches:
                        _phenotype_paths.append(x)
                        if phenotype_suffix is not None:
                            _phenotype_suffix.append(phenotype_suffix[i])

                if len(_phenotype_paths) > 0:
                    paths.append(
                        (
                            image_key,
                            sbs_path,
                            _phenotype_paths,
                            _phenotype_suffix if phenotype_suffix is not None else None,
                        )
                    )

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)
    output_fs.makedirs(output_dir, exist_ok=True)
    if len(paths) == 0:
        logger.warning("No files found to merge")
    else:
        with (
            _create_default_dask_config(),
            _create_dask_client(dask_scheduler_url, **dask_cluster_parameters),
        ):
            for path in paths:
                image_key, sbs_path, phenotype_paths, phenotype_suffix = path
                merge_sbs_phenotype_pipeline(
                    image_key=image_key,
                    sbs_path=sbs_path,
                    phenotype_paths=phenotype_paths,
                    phenotype_suffix=phenotype_suffix,
                    df_barcode=df_barcode,
                    output_dir=output_dir + output_fs.sep,
                    join_sbs=join_sbs,
                    join_phenotype=join_phenotype,
                    force=force,
                    output_format=output_format,
                    no_version=no_version,
                )


def spot_detect_main(arguments: argparse.Namespace):
    """Run spot detection pipeline.

    This function reads images, performs spot detection, and saves the results to disk.

    :param arguments: argparse namespace containing command-line arguments.
    """

    images = arguments.images
    z_index = arguments.z_index
    image_pattern = arguments.image_pattern
    max_filter_width = arguments.max_filter_width
    sigma_log = arguments.sigma_log
    group_by = arguments.groupby
    channels = arguments.channels
    peak_neighborhood_size = arguments.peak_neighborhood_size
    output = arguments.output
    cycles = arguments.cycles
    spot_detection_method = arguments.spot_detection_method
    spot_detection_n_cycles = arguments.spot_detection_n_cycles
    eps = None
    qmin = None
    qmax = None
    optional_save = arguments.save
    force = arguments.force
    subset = arguments.subset
    expected_cycles = arguments.expected_cycles
    dask_scheduler_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    save_keys = ["peaks", "max"]
    if optional_save is not None:
        save_keys += list(optional_save)
    chunks = arguments.chunks
    no_version = arguments.no_version
    if qmin is not None:
        if qmin < 0 or qmin > 1:
            if qmin != -1:
                logger.info("Disabling qmin as it is out of range.")
            qmin = None
    if qmax is not None:
        if qmax < 0 or qmax > 1:
            if qmax != -1:
                logger.info("Disabling qmax as it is out of range.")
            qmax = None

    if chunks is not None:
        chunks = (chunks, chunks)

    output = _add_suffix(output, ".zarr")

    exp_gen = _set_up_experiment(images, image_pattern, group_by, subset=subset)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_scheduler_url, **dask_cluster_parameters),
    ):
        for img in exp_gen:
            spot_detection_pipeline(
                img,
                iss_channels=channels,
                output=output,
                z_index=z_index,
                output_image_format="zarr",
                max_filter_width=max_filter_width,
                sigma_log=sigma_log,
                save_keys=save_keys,
                cycles=cycles,
                eps=eps,
                qmin=qmin,
                qmax=qmax,
                chunks=chunks,
                peak_neighborhood_size=peak_neighborhood_size,
                force=force,
                no_version=no_version,
                spot_detection_method=spot_detection_method,
                spot_detection_n_cycles=spot_detection_n_cycles,
                expected_cycles=expected_cycles,
            )


def reads_pipeline(
    image_key: str,
    spots_root: zarr.Group,
    labels_root: zarr.Group,
    barcodes_file: str,
    file_separator: str,
    threshold_peaks: float | str | None,
    threshold_peaks_crosstalk_correction: float | str | None,
    output_dir: str,
    save_keys: list[str],
    crosstalk_correction_method: Literal["median", "li_and_speed", "none"],
    crosstalk_correction_by_t: bool,
    crosstalk_correction_method_args: dict,
    bases: list[str],
    label_name: str,
    n_mismatches: int | None = None,
    labels_only: bool = False,
    force: bool = False,
    expand_labels_distance: int | None = None,
    no_version: bool = False,
    barcode_column: str = "barcode",
    read_filter: float | None = None,
    crosstalk_n_reads: int = 500000,
):
    """Run the reads pipeline.

    This function processes spot detection results and generates reads data.

    :param image_key: Unique identifier for the image.
    :param spots_root: Root group containing spot detection results.
    :param labels_root: Root group containing labels data.
    :param barcodes_file: Path to the barcodes file.
    :param file_separator: Separator used in file paths.
    :param threshold_peaks: Single threshold or list of thresholds for spot detection. Alternatively
        a valid query ("peak >= peak.quantile(0.1)")
    :param threshold_peaks_crosstalk_correction: Threshold for crosstalk correction. Alternatively a
        valid query ("peak >= peak.quantile(0.1)")
    :param crosstalk_correction_method_args: Additional arguments for crosstalk correction
    :param crosstalk_correction_by_t: Do crosstalk correction on every cycle instead of together
    :param output_dir: Directory where the output will be saved.
    :param save_keys: List of keys specifying which data to save.
    :param crosstalk_correction_method: Method for crosstalk correction.
    :param bases: List of bases.
    :param label_name: Name of the label.
    :param n_mismatches: Correct reads with less than or equal to number of mismatches in whitelist
    :param labels_only: Call reads in labels only
    :param force: Whether to overwrite existing output
    :param expand_labels_distance: Expand labels in label image by distance pixels
        without overlapping
    :param no_version: Whether to skip version/CLI information in output.
    :param barcode_column: Column name of barcode
    :param read_filter: Filter reads by quality score before assigning reads to labels.
    """

    if not force:
        labels_file = (
            f"{output_dir}{file_separator}labels{file_separator}{image_key}.parquet"
        )
        if is_parquet_file(labels_file):
            logger.info(f"Skipping reads for {image_key}")
            return

    df_barcode = pd.read_csv(barcodes_file)

    if barcode_column != "barcode":
        rename = dict()
        rename[barcode_column] = "barcode"
        df_barcode = df_barcode.rename(rename, axis=1)
    assert "barcode" in df_barcode.columns, (
        f"`barcode` column not found in {barcodes_file}"
    )

    logger.info(f"Running reads for {image_key}")
    spots_sep = _get_sep(spots_root)
    points_path = f"{_get_store_path(spots_root).rstrip(spots_sep)}{spots_sep}points"
    spots_protocol = _get_fs_protocol(_get_fs(spots_root))
    if spots_protocol != "file":
        points_path = f"{spots_protocol}://{points_path}"
    points_path = f"{points_path}/{image_key}-peaks.parquet"
    peaks = dd.read_parquet(points_path)
    maxed = read_ome_zarr_array(spots_root["images"][image_key + "-max"], dask=True)
    labels = read_ome_zarr_array(
        labels_root[image_key + "-" + label_name], dask=True
    ).data.compute()
    if expand_labels_distance is not None and expand_labels_distance > 0:
        labels = expand_labels(labels, distance=expand_labels_distance)
    iss_cycles = maxed.coords["t"].values
    iss_cycles = _fix_cycles(iss_cycles)
    df_barcode["barcode"] = df_barcode["barcode"].apply(
        barcode_to_prefix, args=(iss_cycles,)
    )
    output_dir = output_dir + file_separator
    crosstalk_correction_method = crosstalk_correction_method.lower()
    bases_peaks = peaks.compute()  # load all peaks
    bases_array_reads = _peaks_to_bases(
        maxed=maxed,
        peaks=bases_peaks,
        labels=labels,
        labels_only=labels_only,
        bases=bases,
    )
    logger.info(
        f"{bases_array_reads.sizes['read']:,} spots{' in labels' if labels_only else ''}."
    )
    bases_dataset = xr.Dataset()
    bases_dataset["intensity"] = bases_array_reads
    crosstalk_bases_array = bases_array_reads

    iss_cycles = maxed.coords["t"].values
    custom_metadata = dict(scallops=dict(sbs_cycles=iss_cycles.tolist()))
    if not labels_only:
        crosstalk_bases_array = crosstalk_bases_array.query(dict(read="label>0"))
    if (
        crosstalk_correction_method != "none"
        and threshold_peaks_crosstalk_correction is not None
    ):
        if isinstance(threshold_peaks_crosstalk_correction, float):
            crosstalk_bases_array = crosstalk_bases_array.query(
                dict(read=f"peak>{threshold_peaks_crosstalk_correction}")
            )
        elif isinstance(threshold_peaks_crosstalk_correction, str):
            if threshold_peaks_crosstalk_correction == "auto":
                threshold_peaks_crosstalk_correction_df = peak_thresholds_from_bases(
                    bases_array=crosstalk_bases_array, n_reads=crosstalk_n_reads
                )
                threshold_peaks_crosstalk_correction = (
                    threshold_peaks_crosstalk_correction_df.iloc[0]["threshold"]
                )
                logger.info(
                    f"Threshold peaks crosstalk correction: {threshold_peaks_crosstalk_correction:.2f}."
                )
                crosstalk_bases_array = crosstalk_bases_array.query(
                    dict(read=f"peak>{threshold_peaks_crosstalk_correction}")
                )

                threshold_peaks_crosstalk_correction_df.to_parquet(
                    f"{output_dir}spots{file_separator}{image_key}-peak-crosstalk.parquet",
                    index=False,
                )
            elif threshold_peaks_crosstalk_correction != "none":
                crosstalk_bases_array = crosstalk_bases_array.query(
                    dict(read=threshold_peaks_crosstalk_correction)
                )

        logger.info(
            f"{crosstalk_bases_array.sizes['read']:,} spots in labels for crosstalk"
            f" correction."
        )
        if threshold_peaks_crosstalk_correction is not None:
            custom_metadata["scallops"]["threshold_peaks_crosstalk_correction"] = (
                threshold_peaks_crosstalk_correction
            )
        w = channel_crosstalk_matrix(
            crosstalk_bases_array,
            method=crosstalk_correction_method,
            by_t=crosstalk_correction_by_t,
            **crosstalk_correction_method_args,
        )
        if isinstance(w, da.Array):
            w = w.compute()
            logger.info("Computed crosstalk matrix.")

        bases_array_reads = apply_channel_crosstalk_matrix(bases_array_reads, w)

        if "bases" in save_keys:
            bases_dataset["corrected_intensity"] = bases_array_reads
    if os.environ.get("SCALLOPS_IMAGE_SCALE") == "1":
        bases_array_reads = bases_array_reads.astype(int)

    df_reads = decode_max(bases_array_reads, barcodes=df_barcode)
    if n_mismatches is not None and n_mismatches > 0:
        df_reads = correct_mismatches(
            reads=df_reads, barcodes=df_barcode, n_mismatches=n_mismatches
        )
    reads_to_labels_query = ["(label > 0)"]
    if read_filter is not None:
        reads_to_labels_query.append(f"(Q_mean >= {read_filter})")
    barcode_matches_reads_to_labels_filter = (
        os.environ.get("SCALLOPS_BARCODES_TO_LABELS_NO_FILTER", "") != "1"
    )
    if barcode_matches_reads_to_labels_filter:
        if "barcode_match" in df_reads.columns:
            reads_to_labels_query.append("barcode_match")
        else:
            logger.warning(
                "`barcode_match` column not found in reads - including both matches "
                "and mismatches"
            )
    if threshold_peaks == "auto":
        peak_thresholds_lower_df = peak_thresholds_from_reads(
            df_reads.query(" & ".join(reads_to_labels_query)).compute()
        )
        peak_thresholds_lower_df.to_parquet(
            f"{output_dir}spots{file_separator}{image_key}-peak-labels.parquet",
            index=False,
        )
        threshold_peaks = peak_thresholds_lower_df.iloc[0]["threshold"]
        logger.info(
            f"Threshold peaks for assigning reads to labels: {threshold_peaks:.2f}."
        )
        custom_metadata["scallops"]["threshold_peaks"] = threshold_peaks
    if isinstance(threshold_peaks, float):
        reads_to_labels_query.append(f"peak>{threshold_peaks}")
    elif isinstance(threshold_peaks, str) and threshold_peaks != "none":
        reads_to_labels_query.append(threshold_peaks)

    df_labels = assign_barcodes_to_labels(
        df_reads.query(" & ".join(reads_to_labels_query))
    )

    delayed_results = []

    points_fs, points_path = fsspec.core.url_to_fs(points_path)
    points_ds = pq.ParquetDataset(points_path, filesystem=points_fs)
    image_metadata = None
    if b"scallops" in points_ds.schema.metadata:
        points_metadata = json.loads(points_ds.schema.metadata[b"scallops"])
        image_metadata = points_metadata.get("image_metadata")
    if image_metadata is not None:
        custom_metadata["scallops"]["image_metadata"] = image_metadata

    if not no_version:
        custom_metadata["scallops"].update(cli_metadata())
    custom_metadata["scallops"] = json.dumps(custom_metadata["scallops"])
    if w is not None and "crosstalk" in save_keys:
        dest = f"{output_dir}crosstalk{file_separator}{image_key}-w.zarr"
        delayed_results.append(
            _write_zarr_image(
                name=None,
                group=None,
                root=open_ome_zarr(dest, mode="w"),
                zarr_format="zarr",
                image=w,
                compute=False,
            )
        )

    if "bases" in save_keys:
        bases_dataset = bases_dataset.unify_chunks()
        df_bases = bases_dataset.to_dask_dataframe()

        delayed_results.append(
            _to_parquet(
                df_bases,
                f"{output_dir}bases{file_separator}{image_key}.parquet",
                compute=False,
                write_index=False,
                custom_metadata=custom_metadata,
            )
        )
        # delayed_results.append(
        #     bases_dataset.to_zarr(
        #         f"{output_dir}bases{file_separator}{image_key}.xzarr", compute=False
        #     )
        # )

    if "reads" in save_keys:
        delayed_results.append(
            _to_parquet(
                df_reads,
                f"{output_dir}reads{file_separator}{image_key}.parquet",
                compute=False,
                write_index=False,
                schema={"Q": pa.list_(pa.float64(), len(iss_cycles))},
                custom_metadata=custom_metadata,
            )
        )

    if "labels" in save_keys:
        delayed_results.append(
            _to_parquet(
                df_labels,
                f"{output_dir}labels{file_separator}{image_key}.parquet",
                compute=False,
                write_index=False,
                schema={
                    "barcode_Q_0": pa.list_(pa.float64(), len(iss_cycles)),
                    "barcode_Q_1": pa.list_(pa.float64(), -1),
                },
                custom_metadata=custom_metadata,
            )
        )

    dask.compute(*delayed_results)
    return delayed_results


def reads_main(arguments: argparse.Namespace):
    """Main function for processing reads in the scallops CLI. This function orchestrates the
    processing of reads, handling various parameters and saving the results to the specified output
    directory.

    :param arguments: The parsed command-line arguments.
    :raises`ValueError`: If no input spots are found.


    Parameters extracted from `arguments`:
    - `threshold_peaks`: Threshold for peak detection.
    - `spots`: Path to the spots file.
    - `labels`: Path to the labels file.
    - `barcodes_file`: Path to the barcodes file.
    - `output_dir`: Output directory for saving results.
    - `subset`: Subset of images to process.
    - `save_keys`: List of keys specifying what results to save.
    - `label_name`: Name of the labels.

    Additional parameters derived from `arguments`:
    - `bases`: List of bases.
    - `output_image_format`: Output image format (default is "zarr").
    - `crosstalk_correction_method`: Method for crosstalk correction.
    - `threshold_peaks_crosstalk_correction`: Threshold for peaks in crosstalk correction.

    .. note::
    - This function iterates through the available spots, applies the `reads_pipeline` function to each spot, and computes the results.

    See Also:
    - `reads_pipeline`: The function applied to each spot during the processing.
    """
    no_version = arguments.no_version
    threshold_peaks = arguments.threshold_peaks
    threshold_peaks_crosstalk_correction = arguments.threshold_peaks_crosstalk
    if threshold_peaks is not None:
        try:
            threshold_peaks = float(threshold_peaks)
        except ValueError:
            pass
    if threshold_peaks_crosstalk_correction is not None:
        try:
            threshold_peaks_crosstalk_correction = float(
                threshold_peaks_crosstalk_correction
            )
        except ValueError:
            pass
    spots = arguments.spots
    dask_scheduler_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    labels = arguments.labels
    read_filter = arguments.read_quality_filter
    barcode_column = arguments.barcode_col
    barcodes_file = arguments.barcodes
    output_dir = arguments.output
    subset = arguments.subset
    expand_labels_distance = arguments.expand_labels_distance
    crosstalk_correction_by_t = arguments.crosstalk_correction_by_t
    save_keys = ["reads", "labels", "crosstalk"]
    if arguments.save_bases:
        save_keys.append("bases")
    if threshold_peaks == "auto" or threshold_peaks_crosstalk_correction == "auto":
        save_keys.append("spots")
    label_name = arguments.label_name
    labels_only = not arguments.all_labels
    n_mismatches = arguments.n_mismatches
    crosstalk_n_reads = arguments.crosstalk_nreads

    bases = list(arguments.bases)
    crosstalk_correction_method = arguments.crosstalk_correction_method
    force = arguments.force
    crosstalk_correction_method_args = dict()

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)
    for key in save_keys:
        output_fs.makedirs(output_dir + output_fs.sep + key, exist_ok=True)
    labels_fs, _ = fsspec.core.url_to_fs(labels)
    labels = labels.rstrip(labels_fs.sep)

    spots_fs, _ = fsspec.core.url_to_fs(spots)
    spots = spots.rstrip(spots_fs.sep)
    image_keys = []
    if subset is not None:
        subset = _create_subset_function(subset)
    for key in spots_fs.glob(
        f"{spots}{spots_fs.sep}points{spots_fs.sep}*-peaks.parquet"
    ):
        name = os.path.splitext(os.path.basename(key))[0]
        if not name.startswith("."):  # ignore hidden files
            name = name[: -len("-peaks")]
            if subset is None or subset(name):
                image_keys.append(name)
    if len(image_keys) == 0:
        raise ValueError("No input spots found")
    with (
        _create_default_dask_config(
            {"distributed.admin.large-graph-warning-threshold": "250MB"}
        ),
        _create_dask_client(dask_scheduler_url, **dask_cluster_parameters),
    ):
        for key in image_keys:
            reads_pipeline(
                key,
                spots_root=zarr.open(spots, mode="r"),
                labels_root=zarr.open(labels + labels_fs.sep + "labels", mode="r"),
                barcodes_file=barcodes_file,
                file_separator=output_fs.sep,
                threshold_peaks=threshold_peaks,
                threshold_peaks_crosstalk_correction=threshold_peaks_crosstalk_correction,
                output_dir=output_dir,
                save_keys=save_keys,
                crosstalk_correction_method=crosstalk_correction_method,
                crosstalk_correction_by_t=crosstalk_correction_by_t,
                crosstalk_correction_method_args=crosstalk_correction_method_args,
                bases=bases,
                label_name=label_name,
                expand_labels_distance=expand_labels_distance,
                n_mismatches=n_mismatches,
                labels_only=labels_only,
                force=force,
                no_version=no_version,
                barcode_column=barcode_column,
                read_filter=read_filter,
                crosstalk_n_reads=crosstalk_n_reads,
            )
