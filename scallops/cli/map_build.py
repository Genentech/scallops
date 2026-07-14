"""Module for the Command-Line Interface (CLI) related to building perturbation maps.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import os

import anndata
import dask.array as da
import numpy as np
import pandas as pd

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features.agg import agg_features
from scallops.features.map_cluster import cluster_similarity
from scallops.features.decomposition import (
    pca as pca_embed,
    select_pca_components,
    sphere,
)
from scallops.features.map_eval import (
    fetch_string,
    gmt_to_gene_sets,
    pairwise_benchmark,
    pairwise_similarities,
    read_corum,
    read_gmt,
    read_reactome_fi,
    read_string,
    set_benchmark,
)
from scallops.features.normalize import normalize_features, typical_variation_normalization
from scallops.features.preprocessing import (
    filter_batch_correlated,
    filter_data,
    filter_low_cardinality,
    filter_zero_inflated,
    remove_correlated_features,
    transform_features_yj,
)
from scallops.features.util import (
    _query_anndata, _read_data, _read_parquet_for_map, _slice_anndata,
)
from scallops.io import is_parquet_file
from scallops.utils import _fix_json
from scallops.zarr_io import is_anndata_zarr

logger = _get_cli_logger()

_SCALLOPS_UNS_KEY = "scallops"



_INTERNAL_UNS_KEYS = ("_parquet_sources", "_zarr_is_remote")


def _merge_uns(source: anndata.AnnData, result: anndata.AnnData) -> None:
    """Forward ``uns`` and ``varm`` from *source* into *result*.

    Keys already present in *result* are never overwritten, so transformation-
    specific entries (e.g. ``uns["pca"]`` from TVN) take precedence while
    upstream metadata (e.g. backprojection parameters) is preserved.

    ``varm`` is only propagated when the ``var`` index is unchanged; this
    handles the ``map-filter`` step correctly because
    :func:`~scallops.features.preprocessing.filter_data` calls
    ``_slice_anndata`` internally, which already slices ``varm`` along the
    feature axis before control reaches this function.

    :param source: AnnData whose ``uns`` / ``varm`` should be forwarded.
    :param result: AnnData that receives the missing keys.
    """
    for key, value in source.uns.items():
        if key not in result.uns and key not in _INTERNAL_UNS_KEYS:
            result.uns[key] = value
    if source.var.index.equals(result.var.index):
        for key, value in source.varm.items():
            if key not in result.varm:
                result.varm[key] = value
    # Propagate obsm when both obs are the same population (same index, same length).
    # This ensures that intermediate embeddings (X_pca, X_tvn) persist through all
    # subsequent transformation steps.
    if (len(source.obs) == len(result.obs)
            and source.obs.index.equals(result.obs.index)):
        for key, value in source.obsm.items():
            if key not in result.obsm:
                result.obsm[key] = value


def _skip_if_exists(output: str, force: bool) -> bool:
    """Return *True* when *output* already exists and *force* is *False*.

    Logs an informational message when skipping.

    :param output: Destination path whose existence is checked (zarr or parquet).
    :param force: When *True* the function always returns *False* (overwrite).
    :return: *True* if the step should be skipped, *False* otherwise.
    """
    if force:
        return False
    suffix = output.rstrip("/").lower()
    if suffix.endswith(".zarr") and is_anndata_zarr(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return True
    if (suffix.endswith(".parquet") or suffix.endswith(".pq")) and is_parquet_file(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return True
    return False


def _save_zarr(data: anndata.AnnData, output: str, metadata: dict) -> None:
    """Write *data* to a Zarr store, appending *metadata* to the provenance chain.

    The ``uns["scallops"]`` entry is maintained as a JSON-encoded list of
    per-step metadata dicts.  anndata's zarr writer cannot store a list-of-dicts
    natively, so the chain is serialised to a JSON string.  Callers can
    deserialise it with ``json.loads(data.uns["scallops"])``.

    :param data: AnnData to save.  ``uns["scallops"]`` is updated in-place
        before writing.
    :param output: Destination path.  The ``".zarr"`` suffix is added when
        absent.
    :param metadata: Current step's provenance dict (e.g. from
        :func:`~scallops.cli.util.cli_metadata`).
    """
    if not output.lower().endswith(".zarr"):
        output = output + ".zarr"
    if isinstance(data.X, da.Array) and not da.core._check_regular_chunks(data.X.chunks):
        chunks = list(data.X.chunksize)
        chunks[0] = "auto"
        data.X = data.X.rechunk(tuple(chunks))
    # Additive provenance: accumulate every step's metadata as a JSON string.
    # anndata/zarr cannot store a list-of-dicts in uns, so we serialize to str.
    prev_raw = data.uns.get(_SCALLOPS_UNS_KEY)
    if isinstance(prev_raw, str):
        try:
            prev_chain = json.loads(prev_raw)
            if not isinstance(prev_chain, list):
                prev_chain = [prev_chain]
        except Exception:
            prev_chain = []
    elif isinstance(prev_raw, dict):
        prev_chain = [prev_raw]
    else:
        prev_chain = []
    data.uns[_SCALLOPS_UNS_KEY] = json.dumps(
        prev_chain + [_fix_json(metadata)], default=str
    )
    for _k in _INTERNAL_UNS_KEYS:
        data.uns.pop(_k, None)
    data.write_zarr(output, convert_strings_to_categoricals=False)


# ---------------------------------------------------------------------------
# map-filter
# ---------------------------------------------------------------------------

def run_pipeline_map_filter(arguments: argparse.Namespace) -> None:
    """Filter cells and features as the first step of the map-build pipeline.

    Removes cells exceeding the non-finite-value budget and features whose
    variance falls outside the requested bounds.  All upstream ``uns`` keys
    (including backprojection parameters from prior steps) and correctly-sliced
    ``varm`` entries are forwarded to the output Zarr.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path (``".zarr"`` appended if absent).
        * ``features`` (*list[str] | None*) — feature subset to load; all
          features used when *None*.
        * ``label_filter`` (*str | None*) — pandas query expression applied to
          ``obs`` before variance filtering (e.g. ``"plate=='p1'"``).
        * ``min_variance`` (*float | None*) — minimum variance threshold; values
          < 0 are treated as *None* (disabled).
        * ``max_variance`` (*float | None*) — maximum variance threshold; values
          < 0 are treated as *None* (disabled).
        * ``max_fraction_not_finite`` (*float | None*) — maximum fraction of
          non-finite values allowed per cell.
        * ``by`` (*list[str] | None*) — columns in ``obs`` used to stratify the
          variance computation (e.g. ``["plate", "well"]``).
        * ``client`` (*str | None*) — Dask scheduler URL; ``"none"`` disables
          distributed execution; *None* starts a local cluster.
        * ``dask_cluster`` (*str | None*) — JSON URL or inline JSON with cluster
          parameters.
        * ``force`` (*bool*) — overwrite existing output.
        * ``no_version`` (*bool*) — omit scallops version from provenance.

    Optional additional filter attributes (all default to disabled / safe
    values when absent so the function remains backward-compatible):

        * ``max_correlation`` (*float | None*) — maximum allowed absolute
          Pearson correlation between any two retained features.
        * ``correlation_reference`` (*str | None*) — query expression
          restricting the correlation estimate to reference cells (e.g. NTC).
        * ``correlation_chunk_size`` (*int*) — column-block size for the
          blocked correlation computation (default 512).
        * ``max_zero_fraction`` (*float | None*) — remove features where this
          fraction of values is at or near zero.
        * ``near_zero_threshold`` (*float*) — absolute value below which a
          measurement is counted as zero (default 0.0).
        * ``min_unique`` (*int | None*) — remove features with fewer distinct
          finite values than this (catches binary/categorical columns).
        * ``batch_column`` (*str | list[str] | None*) — obs column(s) that
          define batch identity; enables the batch-correlation filter.
        * ``batch_reference`` (*str | None*) — query restricting the
          batch-correlation test to reference cells.
        * ``batch_pvalue`` (*float*) — significance threshold for the
          batch-correlation test (default 0.05).
        * ``batch_method`` (*str*) — ``"kruskal"`` (default) or ``"anova"``.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    min_variance = arguments.min_variance
    max_variance = arguments.max_variance
    max_fraction_not_finite = arguments.max_fraction_not_finite
    by = arguments.by
    label_filter = arguments.label_filter
    features = arguments.features

    if min_variance is not None and min_variance < 0:
        min_variance = None
    if max_variance is not None and max_variance < 0:
        max_variance = None
    if max_fraction_not_finite is not None and max_fraction_not_finite < 0:
        max_fraction_not_finite = None

    # Optional filters — use getattr for backward compatibility
    max_correlation = getattr(arguments, "max_correlation", None)
    corr_reference = getattr(arguments, "correlation_reference", None)
    corr_chunk = int(getattr(arguments, "correlation_chunk_size", 512))
    max_zero_fraction = getattr(arguments, "max_zero_fraction", None)
    near_zero_threshold = float(getattr(arguments, "near_zero_threshold", 0.0))
    min_unique = getattr(arguments, "min_unique", None)
    batch_column = getattr(arguments, "batch_column", None)
    batch_reference = getattr(arguments, "batch_reference", None)
    batch_pvalue = float(getattr(arguments, "batch_pvalue", 0.05))
    batch_method = getattr(arguments, "batch_method", "kruskal")

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths, features)
        if label_filter is not None:
            data = _slice_anndata(data, _query_anndata(data, label_filter).index)

        logger.info(
            f"Before filter: {data.shape[0]:,} cells, {data.shape[1]:,} features"
        )

        # Variance + finite-value filter
        result = filter_data(
            data,
            max_fraction_not_finite=max_fraction_not_finite,
            min_variance=min_variance,
            max_variance=max_variance,
            by=by,
        )
        _merge_uns(data, result)
        data = result

        # Zero-inflation filter
        if max_zero_fraction is not None:
            data = filter_zero_inflated(
                data,
                max_zero_fraction=max_zero_fraction,
                near_zero_threshold=near_zero_threshold,
                by=by,
            )

        # Low-cardinality (categorical) filter
        if min_unique is not None:
            data = filter_low_cardinality(data, min_unique=int(min_unique))

        # Batch-correlation filter
        if batch_column is not None:
            data = filter_batch_correlated(
                data,
                batch_column=batch_column,
                reference_query=batch_reference,
                pvalue_threshold=batch_pvalue,
                method=batch_method,
            )

        # Correlated-feature filter (done last so p is already minimal)
        if max_correlation is not None:
            data = remove_correlated_features(
                data,
                threshold=float(max_correlation),
                reference_query=corr_reference,
                chunk_size=corr_chunk,
            )

        logger.info(
            f"After filter: {data.shape[0]:,} cells, {data.shape[1]:,} features"
        )
        _save_zarr(data, output, metadata)


# ---------------------------------------------------------------------------
# map-transform-yj
# ---------------------------------------------------------------------------

def run_pipeline_map_transform_yj(arguments: argparse.Namespace) -> None:
    """Apply a Yeo-Johnson power transform to make feature distributions more Gaussian.

    The transform is fitted and applied independently per feature (and
    optionally per group when ``by`` is specified).  Upstream ``uns`` and
    ``varm`` are forwarded unchanged to the output.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``by`` (*list[str] | None*) — columns in ``obs`` to stratify the
          transform (e.g. ``["plate", "well"]``); when *None* the transform is
          fitted on all data jointly.
        * ``client``, ``dask_cluster``, ``force``, ``no_version`` — see
          :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths)
        logger.info(f"Shape: {data.shape[0]:,} x {data.shape[1]:,}")
        result = transform_features_yj(data, by=by)
        _merge_uns(data, result)
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-pca
# ---------------------------------------------------------------------------

def run_pipeline_map_pca(arguments: argparse.Namespace) -> None:
    """Embed data with PCA, optionally fitting the model on a reference subset.

    PCA is fitted on the reference subset (or all observations when no
    ``reference`` is provided) and then applied to *all* observations.  The
    fitted model is stored in ``uns["pca"]`` for downstream use.  Output
    ``var`` is relabelled ``PC1, PC2, …, PCn``.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``reference`` (*str | None*) — pandas query expression selecting
          observations used to *fit* PCA; all cells are *projected*.  When
          *None* PCA is fitted on the full dataset.
        * ``components`` (*int*) — number of PCA components to retain (default
          128).
        * ``batch_size`` (*int*) — batch size for incremental PCA; ≤ 0
          disables incremental fitting.
        * ``whiten`` (*bool*) — divide each component by the square root of
          its explained variance.
        * ``client``, ``dask_cluster``, ``force``, ``no_version`` — see
          :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    n_components = arguments.components
    batch_size = arguments.batch_size
    if batch_size is not None and batch_size <= 0:
        batch_size = None
    whiten = arguments.whiten
    reference_query = getattr(arguments, "reference_query", None)

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            logger.info("Computing dask array for PCA")
            data.X = data.X.compute()
        logger.info(f"Shape: {data.shape[0]:,} x {data.shape[1]:,}")

        if reference_query is not None:
            ref_data = _slice_anndata(
                data, _query_anndata(data, reference_query).index
            )
            logger.info(f"Fitting PCA on {ref_data.shape[0]:,} reference cells")
        else:
            ref_data = data  # fit on all; no copy needed

        # Fit PCA on reference (or all) cells.
        # When ref_data is the full dataset, pca_embed fits and transforms in one
        # pass so we can reuse X_transformed directly instead of recomputing.
        pca_result = pca_embed(
            ref_data,
            n_components=n_components,
            batch_size=batch_size,
            whiten=whiten,
            standardize=False,
        )
        pca_info = pca_result.uns["pca"]

        if reference_query is not None:
            # Project ALL cells in chunks to avoid materialising the full
            # N × n_features matrix at once.
            # Peak memory per chunk: chunk_size × n_features × 4 bytes.
            # Accumulated result:    N × n_pcs × 4 bytes.
            PCs_f32 = pca_info["PCs"].T.astype(np.float32)   # (n_features, n_pcs)
            mean_f32 = (pca_info["mean"].astype(np.float32)
                        if pca_info["mean"] is not None else None)
            var_f32 = (pca_info["variance"].astype(np.float32) if whiten else None)
            n_total = data.shape[0]
            n_pcs = PCs_f32.shape[1]
            chunk_sz = batch_size if batch_size is not None else 50_000
            X_transformed = np.empty((n_total, n_pcs), dtype=np.float32)
            for start in range(0, n_total, chunk_sz):
                end = min(start + chunk_sz, n_total)
                X_chunk = np.asarray(
                    data.X[start:end].compute()
                    if isinstance(data.X, da.Array)
                    else data.X[start:end],
                    dtype=np.float32,
                )
                if mean_f32 is not None:
                    X_chunk -= mean_f32
                X_transformed[start:end] = X_chunk @ PCs_f32
                if var_f32 is not None:
                    X_transformed[start:end] /= np.sqrt(var_f32)
        else:
            # pca_result.X already contains the transformed full dataset
            X_transformed = np.asarray(pca_result.X, dtype=np.float32)

        n_out = X_transformed.shape[1]
        result = anndata.AnnData(
            X=X_transformed,
            obs=data.obs.copy(),
            var=pd.DataFrame(index=[f"PC{i + 1}" for i in range(n_out)]),
            uns=pca_result.uns,
        )
        # Store the PCA embedding in obsm["X_pca"] (scanpy convention) in addition
        # to X so downstream Python analysis can access it via the standard key.
        result.obsm["X_pca"] = X_transformed.copy()
        # Propagate upstream uns; pca_result.uns["pca"] already has priority
        _merge_uns(data, result)
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-tvn
# ---------------------------------------------------------------------------

def run_pipeline_map_tvn(arguments: argparse.Namespace) -> None:
    """Apply Typical Variation Normalization (TVN) and store backprojection parameters.

    TVN performs (1) z-scoring against the reference controls, (2) PCA, and
    optionally (3) per-group covariance alignment.  The following parameters
    required for downstream backprojection are stored in the output ``uns``:

    * ``uns["pca"]`` — PCA components, mean, and explained variance.
    * ``uns["tvn_pre_scale_mean"]`` / ``uns["tvn_pre_scale_std"]`` — statistics
      used for the initial z-scoring step.
    * ``uns["covariance_alignment_inv"]`` — inverse covariance-alignment
      matrices keyed by group (only when ``by`` is set).
    * ``uns["normalization_arguments"]`` — the ``reference_query`` and ``by``
      values used, for audit purposes.
    * ``varm["PCs"]`` — PCA components transposed, shape
      ``(n_features, n_components)``.

    All upstream ``uns`` keys are forwarded so the full chain remains available
    (e.g. filter step metadata).

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``reference`` (*str*) — pandas query expression identifying negative-
          control cells (e.g. ``"gene_symbol=='NTC'"``).
        * ``by`` (*list[str] | None*) — columns in ``obs`` used for per-group
          covariance alignment (e.g. ``["plate"]``).  When *None* no alignment
          is performed.
        * ``client``, ``dask_cluster``, ``force``, ``no_version`` — see
          :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    reference_query = getattr(arguments, "reference_query", "gene_symbol=='NTC'")
    by = arguments.by

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            logger.info("Computing dask array for TVN")
            data.X = data.X.compute()
        logger.info(f"Shape: {data.shape[0]:,} x {data.shape[1]:,}")
        result = typical_variation_normalization(
            data, reference_query=reference_query, by=by
        )
        # Store the TVN embedding in obsm["X_tvn"] (scanpy/AnnData convention)
        # in addition to X so downstream tools can access it via the standard key.
        result.obsm["X_tvn"] = np.asarray(result.X, dtype=np.float32)
        # TVN sets its own uns keys and varm["PCs"]; upstream keys fill in the rest.
        _merge_uns(data, result)
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-agg
# ---------------------------------------------------------------------------

def run_pipeline_map_agg(arguments: argparse.Namespace) -> None:
    """Aggregate single-cell profiles to perturbation-level profiles.

    Optionally applies a minimum-cell filter before aggregation and supports a
    two-step barcode → perturbation aggregation via ``agg_by_barcode``.  All
    upstream ``uns`` entries (including TVN backprojection matrices) and
    ``varm`` are forwarded to the aggregated output so that
    :func:`~scallops.features.backprojection.top_features_from_backprojection`
    can be called directly on the result.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``by`` (*list[str]*) — columns in ``obs`` to aggregate by (e.g.
          ``["gene_symbol"]``).
        * ``perturbation`` (*str*) — column used for the ``min_cells`` check
          (default ``"gene_symbol"``).
        * ``method`` (*str*) — aggregation function: ``"mean"`` or
          ``"median"``.
        * ``min_cells`` (*int | None*) — exclude perturbations with fewer than
          this many cells before aggregating.
        * ``barcode`` (*str*) — column identifying guide barcodes; used only
          when ``agg_by_barcode`` is *True* (default ``"barcode_0"``).
        * ``agg_by_barcode`` (*bool*) — when *True*, first aggregate by
          ``by + [barcode]``, then aggregate by ``by``.  Useful when multiple
          guides target the same gene.
        * ``client``, ``dask_cluster``, ``force``, ``no_version`` — see
          :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by
    method = arguments.method
    min_cells = arguments.min_cells
    barcode_column = arguments.barcode
    agg_by_barcode = arguments.agg_by_barcode
    perturbation_column = arguments.perturbation

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths)
        logger.info(f"Input: {data.shape[0]:,} cells, {data.shape[1]:,} features")

        if min_cells is not None and perturbation_column is not None:
            counts = data.obs[perturbation_column].value_counts()
            keep = counts[counts >= min_cells].index
            data = _slice_anndata(
                data, data.obs[perturbation_column].isin(keep)
            )
            logger.info(
                f"After min-cells ({min_cells}) filter: {data.shape[0]:,} cells, "
                f"{len(keep):,} perturbations retained"
            )

        if agg_by_barcode and barcode_column is not None:
            barcode_by = list(by) + [barcode_column] if by else [barcode_column]
            intermediate = agg_features(data, by=barcode_by, agg_func=method)
            _merge_uns(data, intermediate)
            data = intermediate
            logger.info(f"After barcode aggregation: {data.shape[0]:,} profiles")

        result = agg_features(data, by=by, agg_func=method)
        _merge_uns(data, result)
        logger.info(
            f"After aggregation: {result.shape[0]:,} profiles, "
            f"{result.shape[1]:,} features"
        )
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-center
# ---------------------------------------------------------------------------

def run_pipeline_map_center(arguments: argparse.Namespace) -> None:
    """Center profiles by subtracting the reference (e.g. NTC) mean.

    Calls :func:`~scallops.features.normalize.normalize_features` with
    ``centering=True, scaling=False`` so only the mean of the reference
    observations is subtracted.  Upstream ``uns`` and ``varm`` are forwarded.

    This step is typically applied *after* aggregation (``map-agg``) and
    *before* similarity-matrix computation (``map-similarity``).

    .. note::
       After centering, the NTC profile becomes the zero vector.  Cosine
       similarity of the zero vector is undefined, so NTC rows should be
       excluded in the subsequent ``map-similarity`` step (use
       ``--exclude-reference``).

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``reference`` (*str*) — query expression identifying the reference
          profiles whose mean is subtracted (e.g. ``"gene_symbol=='NTC'"``).
        * ``by`` (*list[str] | None*) — columns in ``obs`` to stratify
          centering by groups; *None* centers globally.
        * ``robust`` (*bool*) — use median instead of mean.
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    reference_query = getattr(arguments, "reference_query", "gene_symbol=='NTC'")
    by = arguments.by
    robust = arguments.robust

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    with _create_default_dask_config():
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            data.X = data.X.compute()
        logger.info(f"Shape: {data.shape[0]:,} x {data.shape[1]:,}")
        result = normalize_features(
            data,
            reference_query=reference_query,
            by=by,
            centering=True,
            scaling=False,
            robust=robust,
        )
        _merge_uns(data, result)
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-similarity
# ---------------------------------------------------------------------------

def run_pipeline_map_similarity(arguments: argparse.Namespace) -> None:
    """Compute the pairwise similarity matrix between perturbation profiles.

    The output is an AnnData where both ``obs`` and ``var`` are indexed by the
    perturbation labels and ``X`` contains the ``(n_perturb, n_perturb)``
    similarity matrix.  Upstream ``uns`` (including TVN backprojection
    parameters) is forwarded so the similarity output can be used as input to
    :func:`run_pipeline_map_recall`.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s) containing
          perturbation-level profiles.
        * ``output`` (*str*) — output Zarr path.
        * ``metric`` (*str*) — similarity metric: ``"cosine"`` or
          ``"pearson"`` (default ``"cosine"``).
        * ``perturbation`` (*str*) — column in ``obs`` used as row/column
          labels in the similarity matrix (default ``"gene_symbol"``).
        * ``exclude_reference`` (*str | None*) — pandas query expression
          identifying profiles to *exclude* before computing similarities
          (e.g. ``"gene_symbol=='NTC'"``).  Recommended when centering has
          been applied because NTC becomes the zero vector.
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    metric = arguments.metric
    perturbation_column = arguments.perturbation
    exclude_reference_query = getattr(arguments, "exclude_reference_query", None)
    output_format = getattr(arguments, "output_format", "matrix")
    cluster_method = getattr(arguments, "cluster_method", None)
    cluster_auto_params = getattr(arguments, "cluster_auto_params", True)
    cluster_n = getattr(arguments, "cluster_n_clusters", None)
    cluster_linkage = getattr(arguments, "cluster_linkage", "ward")
    cluster_max_n = int(getattr(arguments, "cluster_max_n_clusters", 50))
    cluster_min_cs = getattr(arguments, "cluster_min_cluster_size", None)
    cluster_min_samples = getattr(arguments, "cluster_min_samples", None)
    cluster_resolution = getattr(arguments, "cluster_resolution", None)
    cluster_threshold = float(getattr(arguments, "cluster_similarity_threshold", 0.3))
    cluster_elbow_n = int(getattr(arguments, "cluster_elbow_n_range", 20))
    cluster_res_min = float(getattr(arguments, "cluster_leiden_res_min", 0.05))
    cluster_res_max = float(getattr(arguments, "cluster_leiden_res_max", 2.0))
    cluster_seed = int(getattr(arguments, "cluster_random_state", 0))

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    with _create_default_dask_config():
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            data.X = data.X.compute()
        logger.info(f"Input: {data.shape[0]:,} profiles, {data.shape[1]:,} features")

        # Save uns/varm before any slicing — _slice_anndata does not preserve uns.
        upstream_uns = dict(data.uns)
        upstream_varm = dict(data.varm)

        if exclude_reference_query is not None:
            ref_idx = _query_anndata(data, exclude_reference_query).index
            data = _slice_anndata(data, ~data.obs.index.isin(ref_idx))
            logger.info(f"After excluding reference: {data.shape[0]:,} profiles")

        sims = pairwise_similarities(data, metric=metric)

        if perturbation_column is not None and perturbation_column in data.obs.columns:
            labels = data.obs[perturbation_column].astype(str).values
        else:
            labels = data.obs.index.astype(str).values

        if output_format == "matrix":
            # Classic format: obs = var = perturbation labels, X = square sim matrix.
            # Required by map-recall set_benchmark / pairwise_benchmark as-is.
            sim_adata = anndata.AnnData(
                X=sims.astype(np.float32),
                obs=pd.DataFrame(index=labels),
                var=pd.DataFrame(index=labels),
            )
            for key, value in upstream_uns.items():
                sim_adata.uns[key] = value

        else:  # "anndata"
            # AnnData-convention format (scanpy-compatible):
            #   X                  → profiles (n_perturb × n_features)
            #   obs.index          → perturbation labels (consistent with matrix format)
            #   obs (columns)      → full upstream metadata (plate, well, gene_symbol…)
            #   var                → feature names
            #   obsp["similarity"] → (n_perturb × n_perturb) similarity matrix
            #   varm               → PCA loadings and other per-feature matrices
            # obs.index is set to the perturbation labels so that map-recall and
            # set_benchmark can match genes against the correct rows/columns.
            obs_copy = data.obs.copy()
            obs_copy.index = pd.Index(labels, name=perturbation_column)
            sim_adata = anndata.AnnData(
                X=np.asarray(data.X, dtype=np.float32),
                obs=obs_copy,
                var=data.var.copy(),
            )
            for key, value in upstream_uns.items():
                sim_adata.uns[key] = value
            if sim_adata.var.index.equals(data.var.index):
                for key, value in upstream_varm.items():
                    sim_adata.varm[key] = value
            sim_adata.obsp["similarity"] = sims.astype(np.float32)
            logger.info(
                f"AnnData format: X=profiles {sim_adata.shape}, "
                f"obsp['similarity'] ({len(labels)} × {len(labels)})"
            )

        # --- Optional clustering (reorders rows/columns by cluster) ---
        if cluster_method is not None and cluster_method != "none":
            logger.info(f"Clustering with method='{cluster_method}'")
            sim_adata = cluster_similarity(
                sim_adata,
                method=cluster_method,
                auto_params=cluster_auto_params,
                n_clusters=cluster_n,
                linkage_method=cluster_linkage,
                max_n_clusters=cluster_max_n,
                min_cluster_size=cluster_min_cs,
                min_samples=cluster_min_samples,
                resolution=cluster_resolution,
                similarity_threshold=cluster_threshold,
                elbow_n_range=cluster_elbow_n,
                leiden_res_min=cluster_res_min,
                leiden_res_max=cluster_res_max,
                random_state=cluster_seed,
            )
            n_cl = sim_adata.uns["clustering"]["n_clusters"]
            logger.info(f"Clustering done: {n_cl} clusters, obs['cluster'] populated")

        _save_zarr(sim_adata, output, metadata)


# ---------------------------------------------------------------------------
# map-recall
# ---------------------------------------------------------------------------

def run_pipeline_map_cluster(arguments: argparse.Namespace) -> None:
    """Apply clustering to an existing similarity AnnData Zarr and reorder it.

    A convenience wrapper around
    :func:`~scallops.features.map_cluster.cluster_similarity` that reads a
    similarity AnnData Zarr (either format from ``map-similarity``), runs the
    requested clustering algorithm, reorders the matrix so same-cluster
    perturbations are adjacent, and writes a new Zarr with
    ``obs["cluster"]`` populated.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — similarity AnnData Zarr from
          ``map-similarity``.
        * ``output`` (*str*) — output AnnData Zarr path.
        * ``method`` (*str*) — ``"hierarchical"`` (default), ``"hdbscan"``,
          or ``"leiden"``.
        * ``auto_params`` (*bool*) — estimate the main hyperparameter via
          elbow criterion.
        * ``n_clusters`` (*int | None*) — target cluster count for hierarchical.
        * ``linkage`` (*str*) — linkage method for hierarchical.
        * ``max_n_clusters`` (*int*) — upper bound for auto-estimated n_clusters.
        * ``min_cluster_size`` (*int | None*) — HDBSCAN min cluster size.
        * ``min_samples`` (*int | None*) — HDBSCAN min_samples.
        * ``resolution`` (*float | None*) — Leiden resolution.
        * ``similarity_threshold`` (*float*) — edge-weight threshold for Leiden
          graph construction.
        * ``elbow_n_range`` (*int*) — grid size for elbow hyperparameter search.
        * ``leiden_res_min``, ``leiden_res_max`` (*float*) — resolution search
          bounds for Leiden elbow.
        * ``random_state`` (*int*) — random seed for Leiden.
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    with _create_default_dask_config():
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            data.X = data.X.compute()
        if "similarity" in data.obsp and isinstance(data.obsp["similarity"], da.Array):
            data.obsp["similarity"] = np.asarray(data.obsp["similarity"].compute())

        logger.info(
            f"Input: {data.shape[0]:,} perturbations, "
            f"format={'anndata' if 'similarity' in data.obsp else 'matrix'}"
        )

        result = cluster_similarity(
            data,
            method=getattr(arguments, "method", "hierarchical"),
            auto_params=getattr(arguments, "auto_params", True),
            n_clusters=getattr(arguments, "n_clusters", None),
            linkage_method=getattr(arguments, "linkage", "ward"),
            max_n_clusters=int(getattr(arguments, "max_n_clusters", 50)),
            min_cluster_size=getattr(arguments, "min_cluster_size", None),
            min_samples=getattr(arguments, "min_samples", None),
            resolution=getattr(arguments, "resolution", None),
            similarity_threshold=float(getattr(arguments, "similarity_threshold", 0.3)),
            elbow_n_range=int(getattr(arguments, "elbow_n_range", 20)),
            leiden_res_min=float(getattr(arguments, "leiden_res_min", 0.05)),
            leiden_res_max=float(getattr(arguments, "leiden_res_max", 2.0)),
            random_state=int(getattr(arguments, "random_state", 0)),
        )
        n_cl = result.uns["clustering"]["n_clusters"]
        logger.info(
            f"Clustering done: {n_cl} clusters, "
            f"obs['cluster'] added to {result.shape[0]:,} perturbations"
        )
        _save_zarr(result, output, metadata)


def _corum_to_gene_sets(path: str) -> dict[str, list[str]]:
    """Build a {complex_name: [gene, ...]} dict from a raw CORUM file.

    The CORUM tab-separated format has columns ``complex_name`` and
    ``subunits_gene_name`` (semicolon-separated gene symbols).  This is
    the format expected by :func:`~scallops.features.map_eval.read_corum`,
    but that function returns pairwise rows, not gene-set lists.  This
    helper builds the gene-set dict required by
    :func:`~scallops.features.map_eval.set_benchmark`.

    :param path: Path to the CORUM file.
    :return: Mapping from complex name to list of gene symbols.
    """
    import fsspec

    result: dict[str, set] = {}
    with fsspec.open(path, "r") as fh:
        header = fh.readline()
        cols = [c.strip() for c in header.split("\t")]
        name_col = cols.index("complex_name")
        genes_col = cols.index("subunits_gene_name")
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) <= max(name_col, genes_col):
                continue
            complex_name = parts[name_col]
            genes = [g.strip() for g in parts[genes_col].split(";") if g.strip()]
            if complex_name not in result:
                result[complex_name] = set()
            result[complex_name].update(genes)
    return {k: list(v) for k, v in result.items()}


def _memory_monitor_start(warn_pct: float = 80.0, critical_pct: float = 90.0, interval_sec: int = 20):
    """Start a daemon thread that logs loud warnings as RAM fills up.

    Returns a stop-event; call ``stop_event.set()`` to shut it down.
    """
    import threading
    try:
        import psutil
    except ImportError:
        logger.debug("psutil not available — memory monitoring disabled")
        return None

    stop = threading.Event()

    def _watch():
        while not stop.is_set():
            mem = psutil.virtual_memory()
            used_gb  = mem.used   / 1e9
            total_gb = mem.total  / 1e9
            avail_gb = mem.available / 1e9
            pct = mem.percent
            if pct >= critical_pct:
                logger.critical(
                    "!!! CRITICAL MEMORY: %.0f / %.0f GB used (%.0f%%) — "
                    "%.0f GB remaining — OOM KILL IMMINENT !!!",
                    used_gb, total_gb, pct, avail_gb,
                )
            elif pct >= warn_pct:
                logger.warning(
                    "Memory pressure: %.0f / %.0f GB used (%.0f%%) — "
                    "%.0f GB remaining",
                    used_gb, total_gb, pct, avail_gb,
                )
            stop.wait(timeout=interval_sec)

    t = threading.Thread(target=_watch, daemon=True, name="scallops-mem-monitor")
    t.start()
    return stop


def _apply_filter_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Filter cells and features.

    Variance is computed **per plate × well** (stratified), matching gould's
    ``create_steps`` behaviour.  This uses the median group-variance so that a
    feature is only removed if it is uninformative *within* wells, not just
    between wells.  Cell-level filtering (max_fraction_not_finite) is always
    global.

    Three execution paths are selected automatically:

    * **Parquet column-batch** — when ``data.uns["_parquet_sources"]`` exists
      (set by ``_read_data`` for parquet inputs).  Reads all row groups in
      parallel for N features at a time.  ~3 min for 14.3 M × 9 K on S3.

    * **Zarr row-batch** — when ``data.X`` is a dask array without parquet
      sources (local or S3 zarr).  Reads one row-chunk at a time with bounded
      concurrency (16 local / 50 remote).

    * **In-memory** — h5ad or any numpy-backed AnnData.
    """
    from scallops.features.preprocessing import (
        filter_data, filter_batch_correlated, remove_correlated_features,
        _col_batch_filter_parquet,
        _streaming_cell_and_variance_filter,
        _streaming_materialise,
    )

    plate = getattr(args, "plate_column", "plate")
    well  = getattr(args, "well_column",  "well")
    by_cols = [c for c in [plate, well] if c in data.obs.columns] or None

    max_fnf  = getattr(args, "max_fraction_not_finite", 0.25)
    min_var  = getattr(args, "min_variance", 0.1)
    max_var  = getattr(args, "max_variance", None)

    obs_all = data.obs

    # ── Pass 0: obs-only masks (no feature reads) ─────────────────────────
    # Compute label_mask from obs metadata before any I/O.
    label_filter = getattr(args, "label_filter", None)
    null_mask = np.ones(len(obs_all), dtype=bool)
    if label_filter:
        for _col in obs_all.columns:
            if _col in label_filter and obs_all[_col].isna().any():
                _null = obs_all[_col].isna().to_numpy()
                null_mask &= ~_null
                logger.info(
                    "map run [filter]: null guard — dropping %s cells"
                    " where '%s' is null",
                    f"{int(_null.sum()):,}", _col,
                )

    if label_filter:
        keep_idx   = _query_anndata(
            _slice_anndata(data, null_mask) if not null_mask.all() else data,
            label_filter,
        ).index
        label_mask = null_mask & obs_all.index.isin(keep_idx)
    else:
        label_mask = null_mask

    n_total = len(label_mask)
    logger.info(
        "map run [filter]: obs masks → %s / %s cells kept"
        " (%s dropped by null guard, %s by label_filter)",
        f"{label_mask.sum():,}", f"{n_total:,}",
        f"{int((~null_mask).sum()):,}",
        f"{int((null_mask & ~label_mask).sum()):,}",
    )

    parquet_sources = data.uns.get("_parquet_sources")

    if parquet_sources:
        # ── Parquet column-batch path ──────────────────────────────────────
        logger.info(
            "map run [filter]: parquet column-batch mode (%d sources)",
            len(parquet_sources),
        )
        _mem_stop = _memory_monitor_start()
        try:
            X_filtered, cell_keep, feat_keep = _col_batch_filter_parquet(
                parquet_sources, obs_all, label_mask, by_cols,
                max_fnf, min_var, max_var,
                # Use data.var.index (intersection across all files after concat)
                # not sources[0]["feat_cols"] (one file's features, may differ).
                feat_cols=list(data.var.index),
                batch_size=getattr(args, "filter_batch_size", 500_000),
            )
        except MemoryError as exc:
            logger.critical(
                "!!! OUT OF MEMORY during parquet column-batch filter !!!\n  %s", exc,
            )
            raise
        finally:
            if _mem_stop is not None:
                _mem_stop.set()

        logger.info(
            "map run [filter]: done — %s cells × %s features",
            f"{int(cell_keep.sum()):,}", f"{int(feat_keep.sum()):,}",
        )
        result = anndata.AnnData(
            X=X_filtered,
            obs=obs_all.iloc[cell_keep].copy(),
            var=data.var.iloc[feat_keep].copy(),
            uns=dict(data.uns),
        )
        _merge_uns(data, result)
        # Centroid columns needed by local-zscore may have been dropped by the
        # variance/NaN filter.  Read them directly from parquet for kept cells.
        if getattr(args, "scale_method", "global") == "local":
            _cy = getattr(args, "localz_centroid_y", "Nuclei_AreaShape_Center_Y")
            _cx = getattr(args, "localz_centroid_x", "Nuclei_AreaShape_Center_X")
            _cent_need = [
                c for c in [_cy, _cx]
                if c not in result.obs.columns and c in list(data.var.index)
            ]
            if _cent_need:
                import pyarrow.dataset as _pads
                import pyarrow.fs as _pafs2
                _cfs, _ = _pafs2.FileSystem.from_uri(parquet_sources[0]["path"])
                _cpaths = [_pafs2.FileSystem.from_uri(src["path"])[1]
                           for src in parquet_sources]
                _cds = _pads.dataset(_cpaths, filesystem=_cfs, format="parquet")
                _ctab = (
                    _cds.scanner(columns=_cent_need, use_threads=True)
                    .to_table().to_pandas()
                )
                for col in _cent_need:
                    if col in _ctab.columns:
                        result.obs[col] = _ctab[col].values[cell_keep]
                        logger.info(
                            "map run [filter]: saved centroid '%s' to obs "
                            "(dropped by variance/NaN filter, needed by local-zscore)", col
                        )

    elif isinstance(data.X, da.Array):
        # ── Zarr row-batch path ────────────────────────────────────────────
        zarr_is_remote = data.uns.get("_zarr_is_remote", False)
        n_prefetch = 50 if zarr_is_remote else 16
        X_orig = data.X

        n_chunks = X_orig.numblocks[0]
        try:
            chunk_gb = X_orig.chunks[0][0] * X_orig.shape[1] * 4 / 1e9
        except Exception:
            chunk_gb = float("nan")

        logger.info(
            "map run [filter]: zarr row-batch mode (n_prefetch=%d, remote=%s) — "
            "%d chunks × %.1f GB each",
            n_prefetch, zarr_is_remote, n_chunks, chunk_gb,
        )

        _mem_stop = _memory_monitor_start()
        try:
            cell_keep, feat_var = _streaming_cell_and_variance_filter(
                X_orig, obs_all, label_mask, by_cols, max_fnf,
                n_prefetch=n_prefetch,
            )
            logger.info(
                "map run [filter]: pass 1/2 done — %s cells, %s features examined",
                f"{cell_keep.sum():,}", f"{X_orig.shape[1]:,}",
            )

            feat_keep = np.isfinite(feat_var)
            if min_var is not None:
                feat_keep &= feat_var >= min_var
            if max_var is not None:
                feat_keep &= feat_var <= max_var
            logger.info(
                "map run [filter]: %s / %s features pass variance filter",
                f"{feat_keep.sum():,}", f"{X_orig.shape[1]:,}",
            )

            out_gb = cell_keep.sum() * feat_keep.sum() * 4 / 1e9
            logger.info(
                "map run [filter]: pass 2/2 — materialising %s × %s (%.1f GB) …",
                f"{cell_keep.sum():,}", f"{feat_keep.sum():,}", out_gb,
            )
            X_filtered = _streaming_materialise(X_orig, cell_keep, feat_keep,
                                                n_prefetch=n_prefetch)
            logger.info(
                "map run [filter]: materialised %.1f GB", X_filtered.nbytes / 1e9
            )

        except MemoryError as exc:
            logger.critical(
                "!!! OUT OF MEMORY during zarr streaming filter !!!\n  %s", exc,
            )
            raise
        finally:
            if _mem_stop is not None:
                _mem_stop.set()

        result = anndata.AnnData(
            X=X_filtered,
            obs=obs_all.iloc[cell_keep].copy(),
            var=data.var.iloc[feat_keep].copy(),
            uns=dict(data.uns),
        )
        _merge_uns(data, result)

    else:
        # ── In-memory path (h5ad or small numpy-backed data) ──────────────
        data_for_filter = (
            _slice_anndata(data, label_mask) if not label_mask.all() else data
        )
        result_tmp = filter_data(
            data_for_filter,
            max_fraction_not_finite=max_fnf,
            min_variance=min_var,
            max_variance=max_var,
            by=by_cols,
        )
        X_filtered = (
            result_tmp.X if isinstance(result_tmp.X, np.ndarray)
            else result_tmp.X.compute()
        )
        cell_keep = obs_all.index.isin(result_tmp.obs.index)
        feat_keep = data.var.index.isin(result_tmp.var.index)

        logger.info(
            "map run [filter]: in-memory — %s cells × %s features",
            f"{int(cell_keep.sum()):,}", f"{int(feat_keep.sum()):,}",
        )
        result = anndata.AnnData(
            X=X_filtered,
            obs=obs_all.iloc[cell_keep].copy(),
            var=data.var.iloc[feat_keep].copy(),
            uns=dict(data.uns),
        )
        _merge_uns(data, result)

    # ── Post-filter steps (run on already-materialised numpy array) ────────
    if getattr(args, "batch_column", None):
        result = filter_batch_correlated(
            result,
            batch_column=args.batch_column,
            reference_query=getattr(args, "batch_reference", None),
            pvalue_threshold=getattr(args, "batch_pvalue", 0.05),
            method=getattr(args, "batch_method", "kruskal"),
        )

    if getattr(args, "max_correlation", None) is not None:
        result = remove_correlated_features(result, threshold=args.max_correlation)

    return result


def _apply_transform_yj_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Apply Yeo-Johnson transform per plate × well.

    Fitting the transform independently per well (as gould's pipeline does)
    ensures that the power-transform parameters are not skewed by inter-well
    differences in the marginal distributions.
    """
    from scallops.features.preprocessing import transform_features_yj
    plate = getattr(args, "plate_column", "plate")
    well  = getattr(args, "well_column",  "well")
    by_cols = [c for c in [plate, well] if c in data.obs.columns] or None

    # Preserve centroid columns in obs before the NaN pre-filter may drop them
    # from var.  The local-z-score scale step needs them from obs regardless.
    if getattr(args, "scale_method", "global") == "local":
        cy = getattr(args, "localz_centroid_y", "Nuclei_AreaShape_Center_Y")
        cx = getattr(args, "localz_centroid_x", "Nuclei_AreaShape_Center_X")
        for col in [cy, cx]:
            if col in data.var.index and col not in data.obs.columns:
                idx = data.var.index.get_loc(col)
                X_vals = np.asarray(data.X[:, idx], dtype=np.float64)
                data.obs[col] = X_vals
                logger.info(
                    "map run [transform-yj]: saved '%s' from X to obs "
                    "(centroid column, preserved before NaN pre-filter)", col
                )

    # _slice_anndata drops uns; save it so _merge_uns can propagate it to result.
    _saved_uns = dict(data.uns)

    # Drop any feature with even one NaN in valid cells before fitting the
    # power transform — PowerTransformer produces NaN output for such features.
    max_fnf = getattr(args, "max_fraction_not_finite", 0.25)
    if max_fnf is not None:
        invalid_per_cell = (~np.isfinite(data.X)).sum(axis=1)
        keep_cells = invalid_per_cell <= int(data.shape[1] * max_fnf)
        invalid_per_feat = (~np.isfinite(data.X[keep_cells])).sum(axis=0)
        keep_feats = invalid_per_feat == 0
        n_dropped_cells = int((~keep_cells).sum())
        n_dropped_feats = int((~keep_feats).sum())
        if n_dropped_cells or n_dropped_feats:
            logger.info(
                "map run [transform-yj]: pre-filter dropped %s cells,"
                " %s features with any NaN",
                f"{n_dropped_cells:,}", f"{n_dropped_feats:,}",
            )
            data = _slice_anndata(data, keep_cells, keep_feats)
            data.uns.update(_saved_uns)

    result = transform_features_yj(data, by=by_cols)
    _merge_uns(data, result)
    return result


def _apply_scale_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Scale features within each plate × well group.

    ``--scale-method global`` *(default)*
        Standard well-level z-score: for each feature subtract the well mean
        and divide by the well std, computed across **all** cells in that well.

    ``--scale-method local``
        Spatial k-NN z-score: each cell is normalised relative to its *k*
        nearest spatial neighbours within the same plate × well.  Removes
        both the global well bias **and** local spatial gradients.  Requires
        centroid columns in ``obs`` (see ``--localz-centroid-y/x``).

    Local z-score is a strictly more expressive option and subsumes the global
    well z-score; the two modes are **mutually exclusive** — never combine them.
    """
    plate  = getattr(args, "plate_column", "plate")
    well   = getattr(args, "well_column",  "well")
    method = getattr(args, "scale_method", "global")

    if method == "local":
        cy = getattr(args, "localz_centroid_y", "Nuclei_AreaShape_Center_Y")
        cx = getattr(args, "localz_centroid_x", "Nuclei_AreaShape_Center_X")

        # CellProfiler names centroid columns Nuclei_AreaShape_Center_Y which
        # looks like a morphological feature, so they land in data.X (feature
        # matrix), not in data.obs.  normalize_features local-zscore reads
        # centroids from obs.coords, so we must copy them there.
        for centroid_col in [cy, cx]:
            if centroid_col not in data.obs.columns:
                if centroid_col in data.var.index:
                    idx = data.var.index.get_loc(centroid_col)
                    X_vals = data.X[:, idx]
                    if isinstance(X_vals, da.Array):
                        X_vals = X_vals.compute()
                    data.obs[centroid_col] = np.asarray(X_vals, dtype=np.float64)
                    logger.info(
                        f"scale [local]: copied '{centroid_col}' from X to obs "
                        f"for spatial k-NN lookup"
                    )
                else:
                    raise ValueError(
                        f"--localz-centroid column '{centroid_col}' not found in "
                        f"obs or var.  Check --localz-centroid-y / --localz-centroid-x."
                    )

        # batch_size caps the intermediate (batch × neighbors × features) array.
        # 100K × 75 × 5K × 4 bytes = 150 GB — acceptable on ≥256 GB machines.
        localz_batch = int(getattr(args, "localz_batch_size", 50_000))
        result = normalize_features(
            data,
            normalize="local-zscore",
            n_neighbors=int(getattr(args, "localz_neighbors", 75)),
            by=[plate, well],
            max_value=getattr(args, "localz_max_value", 5.0),
            centroid_column_names=(cy, cx),
            batch_size=localz_batch,
        )
    else:
        result = normalize_features(data, normalize="zscore", by=[plate, well])

    _merge_uns(data, result)
    return result


def _pca_view(data: anndata.AnnData) -> anndata.AnnData:
    """Return a temporary AnnData whose X is the PCA embedding stored in obsm.

    Used by sphere and TVN so they operate on the PC-space representation while
    ``data.X`` continues to hold the original scaled features.
    """
    if "X_pca" not in data.obsm:
        raise ValueError(
            "_pca_view: 'X_pca' not found in obsm.  "
            "Run the pca step before sphere or tvn."
        )
    pca_coords = data.obsm["X_pca"]
    n_pcs = pca_coords.shape[1]
    return anndata.AnnData(
        X=pca_coords.copy(),
        obs=data.obs.copy(),
        var=pd.DataFrame(index=[f"PC{i + 1}" for i in range(n_pcs)]),
    )


def _apply_pca_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Fit PCA on the reference subset and project all cells.

    Convention:
      ``X``            stays as the scaled features (N × p) — unchanged.
      ``obsm["X_pca"]``   PCA embedding (N × K).
      ``uns["map_pca"]``  PCA model parameters (components, mean, variance) stored
                          under the ``"map_pca"`` key so they are NOT confused with
                          TVN's internal PCA which lands in ``uns["pca"]``.
    """
    ref_q      = getattr(args, "reference_query", None)
    n_comp     = getattr(args, "pca_components", 128)
    batch_size = getattr(args, "pca_batch_size", 200_000)
    # 0 / None / negative → non-incremental (full-dataset) PCA
    if not batch_size or batch_size < 0:
        batch_size = None
    if isinstance(data.X, da.Array):
        data.X = data.X.compute()

    if ref_q is not None:
        ref_data = _slice_anndata(data, _query_anndata(data, ref_q).index)
    else:
        ref_data = data

    pca_result = pca_embed(ref_data, n_components=n_comp, batch_size=batch_size,
                           whiten=False, standardize=False)
    pca_info = pca_result.uns["pca"]
    PCs_f32  = pca_info["PCs"].T.astype(np.float32)  # (p, K)
    mean_f32 = (pca_info["mean"].astype(np.float32)
                if pca_info["mean"] is not None else None)

    n_total    = data.shape[0]
    n_pcs      = PCs_f32.shape[1]
    proj_batch = batch_size if batch_size else n_total
    X_pca      = np.empty((n_total, n_pcs), dtype=np.float32)
    for start in range(0, n_total, proj_batch):
        end   = min(start + proj_batch, n_total)
        chunk = np.asarray(data.X[start:end], dtype=np.float32)
        if mean_f32 is not None:
            chunk -= mean_f32
        X_pca[start:end] = chunk @ PCs_f32

    # Keep X = scaled features; store PCA coords in obsm
    result = anndata.AnnData(
        X=np.asarray(data.X, dtype=np.float32),   # scaled features preserved
        obs=data.obs.copy(),
        var=data.var.copy(),                        # original feature names preserved
    )
    result.obsm["X_pca"] = X_pca
    result.uns["map_pca"] = pca_info              # original PCA for backprojection chain
    _merge_uns(data, result)
    return result


def _apply_pca_select_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Select significant PC dimensions; keep only those columns in obsm["X_pca"]."""
    from scallops.features.decomposition import select_pca_components
    if "X_pca" not in data.obsm:
        return data   # no PCA has been run; nothing to select

    # Build a temporary AnnData in PC space to run the selection logic
    tmp = _pca_view(data)
    # Attach the PCA uns so select_pca_components can read variance info
    if "map_pca" in data.uns:
        tmp.uns["pca"] = data.uns["map_pca"]

    selected = select_pca_components(
        tmp,
        method=getattr(args, "pca_select_method", "variance"),
        min_variance_fraction=getattr(args, "pca_variance_fraction", 0.95),
        pval=0.05, n_perms=50,
    )
    n_keep = selected.shape[1]
    # Trim the X_pca embedding to the selected columns
    result = anndata.AnnData(
        X=np.asarray(data.X, dtype=np.float32),
        obs=data.obs.copy(),
        var=data.var.copy(),
    )
    result.obsm["X_pca"] = data.obsm["X_pca"][:, :n_keep]
    _merge_uns(data, result)
    return result


def _apply_sphere_inmem(data: anndata.AnnData) -> anndata.AnnData:
    """Apply ZCA sphering to obsm["X_pca"]; X (scaled features) is untouched."""
    from scallops.features.decomposition import sphere
    if "X_pca" not in data.obsm:
        return data
    tmp     = _pca_view(data)
    sphered = sphere(tmp)
    result  = anndata.AnnData(
        X=np.asarray(data.X, dtype=np.float32),
        obs=data.obs.copy(),
        var=data.var.copy(),
    )
    result.obsm["X_pca"] = np.asarray(sphered.X, dtype=np.float32)
    _merge_uns(data, result)
    return result


def _apply_tvn_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Run TVN on the PCA embedding; store result in obsm["X_tvn"].

    Convention:
      ``X``            stays as the scaled features (N × p) — unchanged.
      ``obsm["X_pca"]``  the (sphered) PCA embedding — unchanged.
      ``obsm["X_tvn"]``  TVN output (N × K).
      ``uns["pca"]``   TVN's internal PCA model (operates in PC space).
      ``uns["tvn_*"]`` TVN backprojection parameters.
      ``varm["PCs"]``  TVN component matrix (K × K, in PC space).
    """
    from scallops.features.normalize import typical_variation_normalization
    ref_q  = getattr(args, "reference_query", "gene_symbol=='NTC'")
    by_col = getattr(args, "tvn_by", None)   # only TVN uses this in map run

    # TVN operates on the PCA embedding, not on the raw scaled features
    tmp = _pca_view(data)
    tvn_result = typical_variation_normalization(tmp, reference_query=ref_q, by=by_col)

    # Assemble the updated cells AnnData
    result = anndata.AnnData(
        X=np.asarray(data.X, dtype=np.float32),   # scaled features untouched
        obs=data.obs.copy(),
        var=data.var.copy(),                        # original feature names (p columns)
    )
    result.obsm["X_pca"] = data.obsm["X_pca"]     # (sphered) PCA embedding unchanged
    result.obsm["X_tvn"] = np.asarray(tvn_result.X, dtype=np.float32)

    # Copy TVN uns (pca model, tvn_pre_scale_*, covariance_alignment_inv, …)
    for k, v in tvn_result.uns.items():
        result.uns[k] = v

    # Do NOT copy tvn_result.varm into result: TVN's varm["PCs"] has shape (K, K)
    # but result.var has p columns (original features), so the dimensions are
    # incompatible.  The TVN PCA information lives in uns["pca"] and is used
    # by backproject_tvn / top_features_from_backprojection directly from there.

    # Preserve map_pca and all other upstream uns/obsm (excluding newly set keys)
    _merge_uns(data, result)
    return result


def _apply_localz_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Local z-score: normalise each cell relative to its k spatial neighbours.

    Uses :func:`~scallops.features.normalize.normalize_features` with
    ``normalize="local-zscore"``.  Requires centroid columns in ``obs``
    (``Nuclei_AreaShape_Center_Y`` / ``_Center_X`` by default).

    This is an *optional* step placed between ``scale`` and ``pca``.  It
    captures within-well spatial gradients that the global well-level z-score
    cannot correct.
    """
    from scallops.features.normalize import normalize_features
    ref_q      = getattr(args, "reference_query", None)
    n_neighbors = int(getattr(args, "localz_neighbors", 75))
    max_value   = getattr(args, "localz_max_value", 5.0)
    plate_col   = getattr(args, "plate_column", "plate")
    well_col    = getattr(args, "well_column",  "well")
    centroid_y  = getattr(args, "localz_centroid_y", "Nuclei_AreaShape_Center_Y")
    centroid_x  = getattr(args, "localz_centroid_x", "Nuclei_AreaShape_Center_X")
    result = normalize_features(
        data,
        reference_query=ref_q,
        normalize="local-zscore",
        n_neighbors=n_neighbors,
        by=[plate_col, well_col],
        max_value=max_value,
        centroid_column_names=(centroid_y, centroid_x),
    )
    _merge_uns(data, result)
    return result


def _apply_agg_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Aggregate obsm["X_tvn"] (TVN profiles) to perturbation-level means.

    profiles.zarr structure:
      ``X``     — mean TVN embedding per perturbation (n_pert × K)
      ``var``   — PC names (PC1 … PCK)
      ``varm``  — PCA loadings propagated from cells (for backprojection)
      ``uns``   — all backprojection parameters propagated from cells
    """
    from scallops.features.agg import agg_features
    pert   = getattr(args, "perturbation", "gene_symbol")
    by_col = getattr(args, "agg_by", None) or [pert]
    mc     = getattr(args, "min_cells", None)

    # Use the TVN embedding as the source for aggregation
    if "X_tvn" in data.obsm:
        X_src = data.obsm["X_tvn"]
        n_pcs = X_src.shape[1]
        var_src = pd.DataFrame(index=[f"PC{i + 1}" for i in range(n_pcs)])
    else:
        # Fallback: aggregate X directly (no TVN step was run)
        if isinstance(data.X, da.Array):
            data.X = data.X.compute()
        X_src   = np.asarray(data.X, dtype=np.float32)
        var_src = data.var.copy()

    # Build a temporary AnnData with TVN coords as X for aggregation
    tmp = anndata.AnnData(
        X=X_src.copy(), obs=data.obs.copy(), var=var_src
    )
    if mc is not None and pert in tmp.obs.columns:
        counts = tmp.obs[pert].value_counts()
        keep   = counts[counts >= mc].index
        tmp    = _slice_anndata(tmp, tmp.obs[pert].isin(keep))

    result = agg_features(tmp, by=by_col, agg_func=getattr(args, "agg_method", "mean"))
    # Propagate backprojection uns + varm from cells into profiles
    _merge_uns(data, result)
    return result


def _apply_center_inmem(data: anndata.AnnData, args: argparse.Namespace) -> anndata.AnnData:
    """Center profiles (subtract NTC mean); operates on X = TVN profiles."""
    from scallops.features.normalize import normalize_features
    if isinstance(data.X, da.Array):
        data.X = data.X.compute()
    ref_q  = getattr(args, "reference_query", "gene_symbol=='NTC'")
    result = normalize_features(data, reference_query=ref_q,
                                centering=True, scaling=False)
    _merge_uns(data, result)
    return result


def _apply_similarity_inmem(
    data: anndata.AnnData, args: argparse.Namespace
) -> anndata.AnnData:
    """Compute pairwise cosine / Pearson similarity on the profile X."""
    if isinstance(data.X, da.Array):
        data.X = data.X.compute()
    upstream_uns = dict(data.uns)
    ref_q = getattr(args, "reference_query", "gene_symbol=='NTC'")
    pert  = getattr(args, "perturbation", "gene_symbol")
    if ref_q:
        ref_idx = _query_anndata(data, ref_q).index
        data    = _slice_anndata(data, ~data.obs.index.isin(ref_idx))
    sims   = pairwise_similarities(data, metric=getattr(args, "metric", "cosine"))
    labels = (data.obs[pert].astype(str).values
              if pert in data.obs.columns else data.obs.index.astype(str).values)
    obs_df = pd.DataFrame(index=labels)
    out    = anndata.AnnData(
        X=sims.astype(np.float32),
        obs=obs_df, var=obs_df.copy(),
    )
    for k, v in upstream_uns.items():
        out.uns[k] = v
    return out


def _provenance_steps(data: anndata.AnnData) -> set[str]:
    """Return the set of step names recorded in ``uns["scallops"]``."""
    import json as _json
    raw = data.uns.get(_SCALLOPS_UNS_KEY, "[]")
    try:
        chain = _json.loads(raw) if isinstance(raw, str) else []
    except Exception:
        chain = []
    return {entry.get("map_run_step", "") for entry in chain
            if isinstance(entry, dict)}


def _save_step(data: anndata.AnnData, path: str, step_name: str,
               no_version: bool) -> None:
    """Write *data* to *path*, embedding *step_name* in the provenance entry."""
    meta: dict = {} if no_version else cli_metadata()
    meta["map_run_step"] = step_name
    _save_zarr(data, path, meta)


def run_pipeline_map_run(arguments: argparse.Namespace) -> None:
    """Run the full map-building pipeline on a single machine.

    **Three output files, not one per step.**

    All cell-level transformations (filter through TVN) accumulate in a single
    ``cells.zarr``, following the AnnData conventions established in the pipeline:

    * ``X``               — current working representation (updated at each step)
    * ``obsm["X_pca"]``   — PCA embedding (set by the pca step)
    * ``obsm["X_tvn"]``   — TVN embedding (set by the tvn step)
    * ``varm["PCs"]``     — PCA loadings for backprojection (set by tvn)
    * ``uns["scallops"]`` — provenance chain with one entry *per step*

    Aggregated perturbation profiles are written to ``profiles.zarr`` and the
    similarity matrix to ``similarity.zarr``.

    **Resume via provenance chain.**

    Step completion is tracked in ``uns["scallops"]`` (not by file existence).
    If ``cells.zarr`` already exists, the pipeline reads it, checks which steps
    are recorded in the provenance chain, and skips those steps.  Use
    ``--force`` to ignore provenance and re-run everything.

    Step order:  filter → transform-yj → scale → pca → pca-select → sphere
                 → tvn  [all → cells.zarr]
                 → agg  [→ profiles.zarr]
                 → center → similarity [→ similarity.zarr]
                 → recall [→ recall.parquet + recall_annotated.zarr]

    :param arguments: Parsed CLI namespace from ``map run``.
    """
    import time as _time
    import json as _json

    out_dir    = arguments.output_dir.rstrip("/")
    force      = arguments.force
    no_version = arguments.no_version

    # ── Expand glob patterns in --input (e.g. "s3://bucket/50p/*.parquet") ───
    # Scallops itself does not rely on the shell for expansion, so S3 globs and
    # local globs both work here via fsspec.
    import fsspec as _fsspec

    raw_inputs = list(arguments.input)
    expanded_inputs: list[str] = []
    for pat in raw_inputs:
        if any(c in pat for c in ("*", "?", "[")):
            _fs_pat, _ = _fsspec.url_to_fs(pat)
            matched = _fs_pat.glob(pat) if hasattr(_fs_pat, "glob") else []
            matched = [_fs_pat.unstrip_protocol(p) for p in matched]
            if not matched:
                raise FileNotFoundError(f"--input pattern matched no files: {pat!r}")
            expanded_inputs.extend(sorted(matched))
        else:
            expanded_inputs.append(pat)
    if len(expanded_inputs) != len(raw_inputs):
        logger.info(
            f"map run: --input expanded {len(raw_inputs)} pattern(s) → "
            f"{len(expanded_inputs)} file(s)"
        )
    arguments = argparse.Namespace(**{**vars(arguments), "input": expanded_inputs})

    # Create the output directory — works for local paths, S3, GCS, and Azure
    # because anndata/zarr uses fsspec for all I/O.
    import fsspec as _fsspec
    try:
        _fs_out, _ = _fsspec.url_to_fs(out_dir)
        _fs_out.makedirs(out_dir, exist_ok=True)
    except (NotImplementedError, AttributeError):
        pass   # some fsspec implementations (e.g. S3) don't need explicit mkdir

    _sep = "/" if "://" in out_dir else os.sep

    # ── Three semantic output files, not one per step ─────────────────────────
    # Cloud paths use "/" separator; local paths use os.sep.
    cells_zarr    = f"{out_dir}{_sep}cells.zarr"         # filter → TVN (cell-level)
    profiles_zarr = f"{out_dir}{_sep}profiles.zarr"      # after agg
    sim_zarr      = f"{out_dir}{_sep}similarity.zarr"    # after center + similarity
    recall_pq     = f"{out_dir}{_sep}recall.parquet"
    recall_zarr   = f"{out_dir}{_sep}recall_annotated.zarr"

    all_steps = [
        "filter", "transform-yj", "scale",   # scale can be global or local (--scale-method)
        "pca", "pca-select", "sphere", "tvn", # → all accumulate in cells.zarr
        "agg",                                # → profiles.zarr
        "center", "similarity",               # → similarity.zarr
        "recall",                             # → recall.parquet + recall_annotated.zarr
    ]
    requested = arguments.steps.lower()
    steps = set(all_steps) if requested == "all" else {s.strip() for s in requested.split(",")}

    logger.info(f"map run: output directory → {out_dir}")
    logger.info(f"map run: three output files: cells.zarr / profiles.zarr / similarity.zarr")
    logger.info(f"map run: steps = {', '.join(s for s in all_steps if s in steps)}")

    timings: dict[str, float] = {}

    # ── Helper: which steps already ran (from provenance chain) ───────────────
    def _completed(zarr_path: str) -> set[str]:
        if not force and is_anndata_zarr(zarr_path):
            try:
                d = _read_data([zarr_path])
                return _provenance_steps(d)
            except Exception:
                pass
        return set()

    # ============================================================
    # PHASE 1 — cell-level transforms → single cells.zarr
    # ============================================================
    cell_steps = ["filter", "transform-yj", "scale", "pca", "pca-select", "sphere", "tvn"]
    cell_steps_wanted = [s for s in cell_steps if s in steps]

    already_done_cells = _completed(cells_zarr)
    steps_todo = [s for s in cell_steps_wanted if s not in already_done_cells]

    # ── Condition column: create a derived obs column from a well→condition map ──
    # Needed when the biological condition (e.g. GIRED vs DMSO) is encoded in
    # the well number and not stored explicitly in the data.
    condition_column = getattr(arguments, "condition_column", None)
    condition_map_raw = getattr(arguments, "condition_map", None)
    condition_source = getattr(arguments, "condition_source_column", "well")

    def _add_condition_column(data: anndata.AnnData) -> anndata.AnnData:
        """Ensure the requested condition column exists in obs.

        Two modes:

        * ``--condition-map`` **provided**: derive the column from
          ``--condition-source-column`` using the supplied dict mapping.
          Raises if any source value is missing from the map.

        * ``--condition-map`` **omitted**: assume the column already exists in
          the input data and just verify it is present.  This covers cases where
          the parquet/zarr files were pre-labelled before ingestion.
        """
        if not condition_column:
            return data

        if not condition_map_raw:
            # Pre-existing column mode: just verify it's there
            if condition_column not in data.obs.columns:
                raise ValueError(
                    f"--condition-column '{condition_column}' was specified without "
                    f"--condition-map, so the column is expected to already exist in "
                    f"the input data — but it was not found in obs.  "
                    f"Either add --condition-map or pre-label your input with this column."
                )
            counts = data.obs[condition_column].value_counts().to_dict()
            logger.info(
                f"map run: using pre-existing obs['{condition_column}']  ({counts})"
            )
            return data

        # Derived column mode: apply the well→condition mapping
        import json as _j
        mapping = _j.loads(condition_map_raw) if isinstance(condition_map_raw, str) else condition_map_raw
        if condition_source not in data.obs.columns:
            raise ValueError(
                f"--condition-source-column '{condition_source}' not found in obs.  "
                f"Available: {list(data.obs.columns)}"
            )
        data.obs[condition_column] = (
            data.obs[condition_source].astype(str).map(mapping)
        )
        n_unmapped = int(data.obs[condition_column].isna().sum())
        if n_unmapped:
            unique_vals = data.obs[condition_source].astype(str).unique().tolist()
            raise ValueError(
                f"{n_unmapped:,} cells have '{condition_source}' values not in "
                f"--condition-map {mapping}.  "
                f"Values seen in data: {unique_vals}"
            )
        counts = data.obs[condition_column].value_counts().to_dict()
        logger.info(
            f"map run: created obs['{condition_column}'] from obs['{condition_source}'] "
            f"via {mapping}  ({counts})"
        )
        return data
    # ──────────────────────────────────────────────────────────────────────────

    if not steps_todo:
        logger.info("map run [cell-level]: all steps already in provenance — loading cells.zarr")
        cells = _read_data([cells_zarr])
    else:
        # Load the starting data: either existing cells.zarr (partial run) or raw input
        if already_done_cells and is_anndata_zarr(cells_zarr):
            logger.info(f"map run [cell-level]: resuming from cells.zarr "
                        f"(completed: {', '.join(sorted(already_done_cells))})")
            cells = _read_data([cells_zarr])
            if isinstance(cells.X, da.Array):
                cells.X = cells.X.compute()
        else:
            cells = _read_parquet_for_map(list(arguments.input))
            # ── DO NOT call cells.X.compute() here. ─────────────────────────────
            # Raw parquet files contain ~9,000 columns × float64. Materialising
            # all of them at once requires (n_cells × n_cols × 8) bytes — easily
            # exceeding RAM on large datasets.  Instead we keep the dask array and
            # let filter_data (which is dask-aware) reduce the column count first.
            # The materialisation happens automatically inside _apply_pca_inmem
            # which needs a numpy array for sklearn, by which time feature count
            # has dropped from ~9,000 to ~5,000, cutting memory by ~45 %.
            # ─────────────────────────────────────────────────────────────────────
            logger.info(
                f"map run: raw input loaded lazily ({cells.shape[0]:,} cells × "
                f"{cells.shape[1]:,} features).  Will materialise after filter step."
            )
            # Inject the condition column before any processing step uses it
            cells = _add_condition_column(cells)

        for step in steps_todo:
            if step not in steps:
                continue
            t0 = _time.perf_counter()
            logger.info(f"map run [{step}]: running in memory → accumulates in cells.zarr")
            if step == "filter":
                cells = _apply_filter_inmem(cells, arguments)
                # Streaming path already returns numpy; dask path may not.
                if isinstance(cells.X, da.Array):
                    logger.info(
                        f"map run [filter]: materialising {cells.shape[0]:,} × "
                        f"{cells.shape[1]:,} (post-filter) into RAM …"
                    )
                    cells.X = cells.X.compute()
                logger.info(
                    f"map run [filter]: materialised — "
                    f"{cells.X.nbytes / 1e9:.1f} GB in RAM"
                )
            elif step == "transform-yj":
                cells = _apply_transform_yj_inmem(cells, arguments)
            elif step == "scale":
                cells = _apply_scale_inmem(cells, arguments)
            elif step == "pca":
                cells = _apply_pca_inmem(cells, arguments)
            elif step == "pca-select":
                cells = _apply_pca_select_inmem(cells, arguments)
            elif step == "sphere":
                cells = _apply_sphere_inmem(cells)
            elif step == "tvn":
                cells = _apply_tvn_inmem(cells, arguments)
            timings[step] = _time.perf_counter() - t0
            # Write the updated AnnData to cells.zarr after each step so the
            # provenance chain reflects the progress and we can resume here.
            _save_step(cells, cells_zarr, step, no_version)
            logger.info(f"map run [{step}]: done ({timings[step]:.1f}s) — cells.zarr updated")

    # ============================================================
    # PHASE 2 — aggregate → profiles.zarr
    # ============================================================
    if "agg" in steps:
        already_done_prof = _completed(profiles_zarr)
        if "agg" in already_done_prof:
            logger.info("map run [agg]: already in provenance — loading profiles.zarr")
            profiles = _read_data([profiles_zarr])
        else:
            t0 = _time.perf_counter()
            logger.info("map run [agg]: aggregating cell profiles → profiles.zarr")
            profiles = _apply_agg_inmem(cells, arguments)
            timings["agg"] = _time.perf_counter() - t0
            _save_step(profiles, profiles_zarr, "agg", no_version)
            logger.info(f"map run [agg]: done ({timings['agg']:.1f}s) — profiles.zarr saved")
    else:
        profiles = cells   # unlikely but handle gracefully

    # ============================================================
    # PHASE 3 — center + similarity → similarity.zarr
    # ============================================================
    already_done_sim = _completed(sim_zarr)

    if "center" in steps and "center" not in already_done_sim:
        t0 = _time.perf_counter()
        logger.info("map run [center]: centering profiles in memory")
        profiles = _apply_center_inmem(profiles, arguments)
        timings["center"] = _time.perf_counter() - t0

    if "similarity" in steps:
        if "similarity" in already_done_sim:
            logger.info("map run [similarity]: already in provenance — loading similarity.zarr")
        else:
            t0 = _time.perf_counter()
            logger.info("map run [similarity]: computing similarity → similarity.zarr")
            sim = _apply_similarity_inmem(profiles, arguments)
            cluster_meth = getattr(arguments, "cluster_method", "hierarchical")
            if cluster_meth and cluster_meth != "none":
                sim = cluster_similarity(
                    sim,
                    method=cluster_meth,
                    auto_params=getattr(arguments, "cluster_auto_params", True),
                    n_clusters=getattr(arguments, "cluster_n_clusters", None),
                    linkage_method=getattr(arguments, "cluster_linkage", "ward"),
                    max_n_clusters=int(getattr(arguments, "cluster_max_n_clusters", 50)),
                    min_cluster_size=getattr(arguments, "cluster_min_cluster_size", None),
                    min_samples=getattr(arguments, "cluster_min_samples", None),
                    resolution=getattr(arguments, "cluster_resolution", None),
                    similarity_threshold=float(getattr(arguments, "cluster_similarity_threshold", 0.3)),
                    elbow_n_range=int(getattr(arguments, "cluster_elbow_n_range", 20)),
                    leiden_res_min=float(getattr(arguments, "cluster_leiden_res_min", 0.05)),
                    leiden_res_max=float(getattr(arguments, "cluster_leiden_res_max", 2.0)),
                    random_state=int(getattr(arguments, "cluster_random_state", 0)),
                )
            # Carry provenance from profiles into the similarity AnnData
            for k, v in profiles.uns.items():
                if k not in sim.uns:
                    sim.uns[k] = v
            _save_step(sim, sim_zarr, "similarity", no_version)
            if "center" in timings:
                # inject center into provenance as well
                _save_step(sim, sim_zarr, "center", no_version)
            timings["similarity"] = _time.perf_counter() - t0
            logger.info(f"map run [similarity]: done ({timings['similarity']:.1f}s) — similarity.zarr saved")

    # ============================================================
    # PHASE 4 — recall (optional)
    # ============================================================
    has_recall = any([
        getattr(arguments, "corum", None),
        getattr(arguments, "gmt", None),
        getattr(arguments, "string_fetch", False),
    ])
    if "recall" in steps and has_recall:
        if not force and is_parquet_file(recall_pq):
            logger.info("map run [recall]: skipping (parquet exists)")
        else:
            t0 = _time.perf_counter()
            run_pipeline_map_recall(argparse.Namespace(
                input=[sim_zarr], output=recall_pq,
                force=force, no_version=no_version,
                corum=getattr(arguments, "corum", None) or [],
                gmt=getattr(arguments, "gmt", None) or [],
                string=None,
                string_fetch=getattr(arguments, "string_fetch", False),
                string_threshold=getattr(arguments, "string_threshold", 400),
                string_species=getattr(arguments, "string_species", 9606),
                string_network_type=getattr(arguments, "string_network_type", "full"),
                reactome=None,
                min_genes=getattr(arguments, "min_genes", 5),
                min_pairs=getattr(arguments, "min_pairs", 10),
                inject_zarr=recall_zarr,
            ))
            timings["recall"] = _time.perf_counter() - t0
    elif "recall" in steps:
        logger.info("map run [recall]: skipped (no --corum / --gmt / --string-fetch)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(timings.values())
    logger.info("=" * 60)
    logger.info("map run: complete")
    for s in all_steps:
        if s in timings:
            t = timings[s]
            logger.info(f"  {s:<20s}  {t//3600:02.0f}h:{(t%3600)//60:02.0f}m:{t%60:04.1f}s")
    logger.info(f"  {'TOTAL':<20s}  {total//3600:02.0f}h:{(total%3600)//60:02.0f}m:{total%60:04.1f}s")
    logger.info(f"  Outputs → {cells_zarr}")
    logger.info(f"          → {profiles_zarr}")
    logger.info(f"          → {sim_zarr}")
    if is_parquet_file(recall_pq):
        logger.info(f"          → {recall_pq}")


def run_pipeline_map_recall(arguments: argparse.Namespace) -> None:
    """Evaluate the similarity matrix against multiple reference databases.

    Supports four reference types that can be combined freely in one run:

    * **CORUM** — protein complexes; uses :func:`set_benchmark` (KS test on
      within-complex vs between-complex similarities).
    * **GMT** — gene sets in ``.gmt`` format (Reactome, KEGG, GO, MSigDB);
      uses :func:`set_benchmark`.
    * **STRING** — pairwise protein-protein interactions; uses
      :func:`pairwise_benchmark` (recall of interacting pairs).
    * **Reactome FI** — Reactome Functional Interactions; uses
      :func:`pairwise_benchmark`.

    All sources produce rows in the output Parquet with a ``source`` column
    indicating the file name and a ``method`` column (``"set_benchmark"`` or
    ``"pairwise_recall"``).

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — similarity matrix Zarr from ``map-similarity``.
        * ``output`` (*str*) — output Parquet path.
        * ``corum`` (*list[str] | None*) — CORUM complex file(s).
        * ``gmt`` (*list[str] | None*) — GMT gene-set file(s)
          (Reactome, KEGG, GO, MSigDB, …).
        * ``string`` (*list[str] | None*) — STRING TSV file(s)
          (``preferredName_A``, ``preferredName_B``, ``score`` columns).
        * ``string_fetch`` (*bool*) — when *True*, query the STRING REST API
          for genes in the similarity matrix.
        * ``string_threshold`` (*int*) — STRING combined-score cutoff (0–1000,
          default 400).
        * ``string_species`` (*int*) — NCBI taxonomy ID for the STRING query
          (default 9606 = human).
        * ``string_network_type`` (*str*) — ``"full"`` or ``"physical"``.
        * ``reactome`` (*list[str] | None*) — Reactome FI file(s).
        * ``min_genes`` (*int*) — minimum gene count for set-based benchmarks.
        * ``min_pairs`` (*int*) — minimum pair count for pairwise benchmarks.
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    corum_paths = getattr(arguments, "corum", None) or []
    gmt_paths = getattr(arguments, "gmt", None) or []
    string_paths = getattr(arguments, "string", None) or []
    string_fetch = getattr(arguments, "string_fetch", False)
    string_threshold = int(getattr(arguments, "string_threshold", 400))
    string_species = int(getattr(arguments, "string_species", 9606))
    string_network_type = getattr(arguments, "string_network_type", "full")
    reactome_paths = getattr(arguments, "reactome", None) or []
    min_genes = int(getattr(arguments, "min_genes", 10))
    min_pairs = int(getattr(arguments, "min_pairs", 10))

    if not force and is_parquet_file(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return

    if not output.lower().endswith((".parquet", ".pq")):
        output = output + ".parquet"

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    raw = _read_data(paths)
    if isinstance(raw.X, da.Array):
        raw.X = raw.X.compute()

    # Support both map-similarity output formats:
    #   matrix format:  X is the square (n×n) similarity matrix
    #   anndata format: X is profiles, obsp["similarity"] is the square matrix
    if "similarity" in raw.obsp:
        logger.info(
            "Detected AnnData format (obsp['similarity']): "
            f"{raw.shape[0]:,} profiles, building square similarity AnnData for recall"
        )
        sims = np.asarray(raw.obsp["similarity"])
        labels = raw.obs.index.astype(str).values
        data = anndata.AnnData(
            X=sims.astype(np.float32),
            obs=pd.DataFrame(index=labels),
            var=pd.DataFrame(index=labels),
            uns=dict(raw.uns),
        )
    else:
        data = raw
    logger.info(f"Similarity matrix: {data.shape[0]:,} x {data.shape[1]:,}")

    results: list[pd.DataFrame] = []

    # ------------------------------------------------------------------ #
    # 1. CORUM — set-based (KS test)
    # ------------------------------------------------------------------ #
    for path in corum_paths:
        source = os.path.basename(path)
        logger.info(f"CORUM: {source}")
        gene_sets = _corum_to_gene_sets(path)
        df = set_benchmark(data, gene_sets, min_genes=min_genes)
        if len(df):
            df.insert(0, "method", "set_benchmark")
            df.insert(0, "source", source)
            results.append(df)

    # ------------------------------------------------------------------ #
    # 2. GMT (Reactome, KEGG, GO, MSigDB, …) — set-based (KS test)
    # ------------------------------------------------------------------ #
    for path in gmt_paths:
        source = os.path.basename(path)
        logger.info(f"GMT: {source}")
        gmt_df = read_gmt(path)
        gene_sets = gmt_to_gene_sets(gmt_df)
        df = set_benchmark(data, gene_sets, min_genes=min_genes)
        if len(df):
            df.insert(0, "method", "set_benchmark")
            df.insert(0, "source", source)
            results.append(df)

    # ------------------------------------------------------------------ #
    # 3. STRING flat files — pairwise recall
    # ------------------------------------------------------------------ #
    for path in string_paths:
        source = os.path.basename(path)
        logger.info(f"STRING (file): {source}")
        pairs = read_string(path, score_threshold=string_threshold)
        df = pairwise_benchmark(data, pairs, min_pairs=min_pairs)
        if len(df):
            df.insert(0, "method", "pairwise_recall")
            df.insert(0, "source", source)
            results.append(df)

    # ------------------------------------------------------------------ #
    # 4. STRING REST API — pairwise recall
    # ------------------------------------------------------------------ #
    if string_fetch:
        genes = list(data.obs.index)
        logger.info(f"STRING API: querying {len(genes)} genes (threshold={string_threshold})")
        pairs = fetch_string(
            genes,
            species_id=string_species,
            score_threshold=string_threshold,
            network_type=string_network_type,
        )
        df = pairwise_benchmark(data, pairs, min_pairs=min_pairs)
        if len(df):
            df.insert(0, "method", "pairwise_recall")
            df.insert(0, "source", f"STRING_{string_network_type}_score{string_threshold}")
            results.append(df)

    # ------------------------------------------------------------------ #
    # 5. Reactome FI — pairwise recall
    # ------------------------------------------------------------------ #
    for path in reactome_paths:
        source = os.path.basename(path)
        logger.info(f"Reactome FI: {source}")
        pairs = read_reactome_fi(path)
        df = pairwise_benchmark(data, pairs, min_pairs=min_pairs)
        if len(df):
            df.insert(0, "method", "pairwise_recall")
            df.insert(0, "source", source)
            results.append(df)

    if not results:
        logger.warning("No recall results produced — check that at least one reference source was provided.")
        result_df = pd.DataFrame()
    else:
        result_df = pd.concat(results, ignore_index=True)

    result_df.to_parquet(output, index=False)
    logger.info(f"Saved {len(result_df):,} recall entries to {output}")

    # --- Optionally inject recall results into the input similarity AnnData ---
    inject_zarr = getattr(arguments, "inject_zarr", None)
    if inject_zarr is not None:
        if not inject_zarr.lower().endswith(".zarr"):
            inject_zarr = inject_zarr + ".zarr"
        logger.info(f"Injecting recall results into AnnData Zarr → {inject_zarr}")
        # Build recall dict: {source: JSON string of records}.
        # anndata/zarr cannot store a list-of-dicts natively; each source's
        # records are JSON-encoded.  Readers reconstruct them with:
        #   pd.DataFrame(json.loads(data.uns["recall"]["source_name"]))
        recall_by_source: dict = {}
        for source in result_df["source"].unique():
            subset = result_df[result_df["source"] == source]
            recall_by_source[source] = json.dumps(
                subset.to_dict(orient="records"), default=str
            )
        # Store in uns and write a new zarr (don't mutate the original in-place)
        raw.uns["recall"] = recall_by_source
        raw.uns["recall_parquet"] = output
        if not (getattr(arguments, "force", True)) and is_anndata_zarr(inject_zarr):
            logger.info(f"{inject_zarr} already exists, skipping. Use --force to overwrite.")
        else:
            metadata_inject = {}
            if not getattr(arguments, "no_version", False):
                metadata_inject.update(cli_metadata())
            _save_zarr(raw, inject_zarr, metadata_inject)


# ---------------------------------------------------------------------------
# map-sphere
# ---------------------------------------------------------------------------


def run_pipeline_map_sphere(arguments: argparse.Namespace) -> None:
    """Apply ZCA sphering (whitening) to decorrelate features.

    The data is centred and rescaled so that the sample covariance of the
    output approximates the identity matrix.  When *by* is set the transform
    is fitted and applied independently within each group (e.g. per plate or
    condition), which corrects for group-level covariance differences.

    This step is used in the pre-TVN pipeline to produce a sphered
    representation before Typical Variation Normalization.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s).
        * ``output`` (*str*) — output Zarr path.
        * ``by`` (*list[str] | None*) — columns in ``obs`` to stratify the
          sphering transform (e.g. ``["condition"]``).
        * ``epsilon`` (*float*) — regularisation constant for the SVD
          inversion (default 1e-5).
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by
    epsilon = float(getattr(arguments, "epsilon", 1e-5))

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    with _create_default_dask_config():
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            logger.info("Computing dask array for sphering")
            data.X = data.X.compute()
        logger.info(f"Shape: {data.shape[0]:,} x {data.shape[1]:,}")
        result = sphere(data, by=by, epsilon=epsilon)
        _merge_uns(data, result)
        _save_zarr(result, output, metadata)


# ---------------------------------------------------------------------------
# map-pca-select
# ---------------------------------------------------------------------------


def run_pipeline_map_pca_select(arguments: argparse.Namespace) -> None:
    """Select statistically significant PCA components via the Tracy-Widom distribution.

    Reads ``uns["pca"]["variance"]`` written by :func:`run_pipeline_map_pca`
    and retains only informative PCA components using the chosen *method*.

    **Recommended for morphological data**: use ``method="variance"`` or
    ``method="permutation"``.  The ``"tracy_widom"`` test assumes i.i.d.
    Gaussian entries under the null — an assumption strongly violated by
    correlated morphological features (Cell Painting, OPS) where even
    noise-only data produces eigenvalues above the Tracy-Widom threshold.

    The output is a sliced AnnData retaining only the selected ``PC1 … PCk``
    columns.

    :param arguments: Parsed CLI namespace.  Expected attributes:

        * ``input`` (*list[str]*) — input Zarr or Parquet path(s) from
          ``map-pca``.
        * ``output`` (*str*) — output Zarr path.
        * ``method`` (*str*) — ``"variance"`` (default), ``"permutation"``,
          or ``"tracy_widom"``.
        * ``min_variance_fraction`` (*float*) — minimum cumulative variance
          fraction for ``method="variance"`` (default 0.95).
        * ``pval`` (*float*) — significance level for ``"permutation"`` or
          ``"tracy_widom"`` (default 0.05).
        * ``n_perms`` (*int*) — number of permutation replicates for
          ``method="permutation"`` (default 100).
        * ``max_components`` (*int | None*) — hard cap on retained components.
        * ``n_features`` (*int | None*) — number of original features (for
          ``"tracy_widom"``); inferred from ``uns["pca"]["features"]`` when
          *None*.
        * ``force``, ``no_version`` — see :func:`run_pipeline_map_filter`.
    """
    paths = arguments.input
    output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    method = getattr(arguments, "method", "variance")
    min_variance_fraction = float(getattr(arguments, "min_variance_fraction", 0.95))
    pval = getattr(arguments, "pval", 0.05)
    n_perms = int(getattr(arguments, "n_perms", 100))
    max_components = getattr(arguments, "max_components", None)
    n_features = getattr(arguments, "n_features", None)

    if _skip_if_exists(output, force):
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())

    with _create_default_dask_config():
        data = _read_data(paths)
        if isinstance(data.X, da.Array):
            data.X = data.X.compute()
        logger.info(f"Input: {data.shape[0]:,} obs × {data.shape[1]:,} PCs")
        result = select_pca_components(
            data,
            method=method,
            min_variance_fraction=min_variance_fraction,
            pval=pval,
            n_perms=n_perms,
            max_components=max_components,
            n_features=n_features,
        )
        _merge_uns(data, result)
        logger.info(f"After selection: {result.shape[1]:,} PCs retained")
        _save_zarr(result, output, metadata)
