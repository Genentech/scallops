import logging
from collections.abc import Sequence
from tokenize import NAME

import anndata
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from anndata._core.index import _normalize_index
from anndata.typing import Index
from pandas.core.computation.parsing import BACKTICK_QUOTED_STRING, tokenize_string

from scallops.features.constants import _metadata_columns_whitelist_str
from scallops.io import read_anndata_zarr

logger = logging.getLogger("scallops")


def pandas_to_anndata(
    df: pd.DataFrame | dd.DataFrame, features: Sequence[str] | None = None
) -> anndata.AnnData:
    """Convert a data frame to AnnData representation
    :param df: data frame
    :param features: Features to use. If not provided, features are inferred.
    :return: AnnData object

    """
    if features is None:
        features = infer_feature_columns(df)
    # https://github.com/dask/dask/issues/12411
    data = (
        df[features].values
        if not isinstance(df, dd.DataFrame)
        else df[features].to_dask_array(lengths=tuple(df.map_partitions(len).compute()))
    )

    df = df.drop(columns=features)
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    obs = df.reset_index(drop=df.index.name is None)
    skip = [s for s in ["barcode_Q_0", "barcode_Q_1"] if s in df.columns]
    if len(skip) > 0:
        obs = obs.drop(skip, axis=1)
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if pd.api.types.is_object_dtype(obs[c]):
            obs[c] = obs[c].astype(str)  # to save with anndata
    return anndata.AnnData(
        obs=obs,
        var=pd.DataFrame(index=features),
        X=data,
    )


def _query_anndata(data: anndata.AnnData, query: str):
    fields = _get_names_from_pd_query(query)
    obs = data.obs
    added_fields = []
    added_indices = []
    for field in fields:
        if field not in obs.columns:
            index = data.var.index.get_indexer_for([field])
            if index[0] != -1:
                index = index[0]
                added_fields.append(field)
                added_indices.append(index)
    if len(added_fields) > 0:
        values = data.X[:, added_indices]
        if isinstance(values, da.Array):
            values = values.compute()
        obs = obs.copy()
        for i in range(len(added_fields)):
            obs[added_fields[i]] = values[:, i]

    return obs.query(query)


def _slice_anndata(
    data: anndata.AnnData,
    obs: Index | None,
    var: Index | None = None,
) -> anndata.AnnData:
    """Slice an AnnData object without AnnData's copy-on-write behavior.
    Note that this method slices the fields `X`, `layers`, `obsm`, `varm`, `obs`,
    and `var`.

    :param data: AnnData object
    :param obs: Slice for observations
    :param var: Slice for variables
    :return: Sliced AnnData object
    """
    obs_indices = None
    var_indices = None

    if obs is not None:
        obs_indices = _normalize_index(obs, data.obs.index)
    if var is not None:
        var_indices = _normalize_index(var, data.var.index)
    X = data.X
    layers = dict(data.layers)
    obsm = dict(data.obsm)
    varm = dict(data.varm)

    if obs_indices is not None:
        X = X[obs_indices]
        for key in layers:
            layers[key] = layers[key][obs_indices]
        for key in obsm:
            obsm[key] = obsm[key][obs_indices]
    if var_indices is not None:
        X = X[:, var_indices]
        for key in layers:
            layers[key] = layers[key][:, var_indices]
        for key in varm:
            varm[key] = varm[key][var_indices]
    obs = data.obs.iloc[obs_indices] if obs_indices is not None else data.obs
    var = data.var.iloc[var_indices] if var_indices is not None else data.var
    return anndata.AnnData(X=X, obs=obs, var=var, layers=layers, obsm=obsm, varm=varm)


def _update_coords(
    df: pd.DataFrame,
    df_coords: bool | str | Sequence[str],
    coord_name: str,
    coords_keys: set,
    xarray_coords: dict,
):
    if df_coords:
        xarray_coords[coord_name] = df.index.to_numpy(copy=False)
        if isinstance(df_coords, str):
            columns = [df_coords]
        elif isinstance(df_coords, Sequence):
            columns = df_coords
        else:
            columns = df.columns
        for c in columns:
            counter = 1
            coord = c
            while coord in coords_keys:
                coord = f"{c}_{counter}"
                counter += 1
            coords_keys.add(coord)
            xarray_coords[coord] = (coord_name, df[c].to_numpy(copy=False))


def _anndata_to_xr(
    data: anndata.AnnData,
    obs_coords: bool | str | Sequence[str] = True,
    var_coords: bool | str | Sequence[str] = False,
) -> xr.DataArray:
    coords = dict()
    coords_keys = {"obs", "var"}
    _update_coords(
        df=data.obs,
        df_coords=obs_coords,
        coord_name="obs",
        coords_keys=coords_keys,
        xarray_coords=coords,
    )
    _update_coords(
        df=data.var,
        df_coords=var_coords,
        coord_name="var",
        coords_keys=coords_keys,
        xarray_coords=coords,
    )
    return xr.DataArray(data.X, dims=("obs", "var"), name="", coords=coords)


def _join_metadata(
    data: anndata.AnnData, join_df: pd.DataFrame | dd.DataFrame, on: Sequence[str]
):
    # match data type
    for field in on:
        if join_df[field].dtype != data.obs[field].dtype:
            data.obs[field] = data.obs[field].astype(join_df[field].dtype)
    if isinstance(join_df, dd.DataFrame):
        join_df = join_df.compute()
    join_df = join_df.set_index(on)
    data.obs = data.obs.join(join_df, on=on)


def _read_data(
    paths: Sequence[str] | str, features: Sequence[str] | None = None
) -> anndata.AnnData:
    if isinstance(paths, str):
        paths = [paths]
    assert len(paths) == len(set(paths)), "Duplicate path"
    data_arrays = []
    parquet_sources: list = []   # collected outside the loop — survives concat
    zarr_is_remote: bool | None = None
    for path in paths:
        if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
            import dask
            import pyarrow as _pa
            import pyarrow.fs as _pafs
            import pyarrow.parquet as _pq

            # ── 1. Open the parquet file and read metadata only ────────────
            # Use PyArrow native filesystem throughout (not fsspec) so that
            # pre_buffer=True can coalesce S3 range requests efficiently.
            _pa_fs, _pa_fpath = _pafs.FileSystem.from_uri(path)
            with _pa_fs.open_input_file(_pa_fpath) as _f:
                _schema  = _pq.read_schema(_f)
                _pq_meta = _pq.read_metadata(_f)

            # ── 2. Identify feature vs obs columns ─────────────────────────
            # schema.names includes all data columns (not the hidden index).
            # Feature columns follow CellProfiler naming: Compartment_Type_…
            _all_data_cols = _schema.names
            _feat_cols = list(features) if features else [
                c for c in _all_data_cols
                if c.split("_")[0] in {"Cells", "Nuclei", "Cytoplasm"}
            ]
            _feat_set = set(_feat_cols)

            # ── 3. Obs cols: skip expensive nested types ───────────────────
            # list<double> columns (e.g. barcode_Q_0) can take minutes to
            # deserialise from S3 even with column pruning — exclude them.
            _obs_cols = [
                c for c in _all_data_cols
                if c not in _feat_set
                and not _pa.types.is_list(_schema.field(c).type)
                and not _pa.types.is_large_list(_schema.field(c).type)
                and not _pa.types.is_struct(_schema.field(c).type)
            ]

            # ── 4. Obs: eager read via pyarrow (correct index restoration) ─
            # pyarrow reads the pandas index metadata stored in the footer
            # and restores it automatically (RangeIndex by name, non-range
            # indices as actual columns).  This avoids a dask bug where unnamed
            # parquet indices cause `[None] not in index` KeyErrors.
            _obs_table = _pq.read_table(
                _pa_fpath, columns=_obs_cols, filesystem=_pa_fs, pre_buffer=True,
            )
            _obs_df = _obs_table.to_pandas()

            # ── 5. Name the obs.index ──────────────────────────────────────
            # If the parquet stored an unnamed RangeIndex (__index_level_0__
            # or None), rename it to 'label' — it often represents a cell ID.
            if _obs_df.index.name in (None, "__index_level_0__"):
                _obs_df.index.name = (
                    "label" if "label" not in _obs_df.columns else "__cell_id__"
                )

            _obs_df.index = _obs_df.index.astype(str)
            for _c in _obs_df.columns:
                if pd.api.types.is_object_dtype(_obs_df[_c]):
                    _obs_df[_c] = _obs_df[_c].astype(str)

            # ── 6. Feature data: truly lazy per-row-group dask array ───────
            # We bypass dd.read_parquet (which also triggers the dask index
            # bug on compute) and build the dask array from dask.delayed
            # calls, one per parquet row group.  Each row group is only read
            # from S3 when its chunk is actually needed.

            def _read_rg(rg_idx: int) -> np.ndarray:
                """Read one parquet row group, return feature numpy array."""
                import pyarrow.fs as _pafs2
                import pyarrow.parquet as _pq2
                _rg_pa_fs, _rg_pa_fp = _pafs2.FileSystem.from_uri(path)
                with _rg_pa_fs.open_input_file(_rg_pa_fp) as _rg_f:
                    _pf = _pq2.ParquetFile(_rg_f, pre_buffer=True)
                    _tbl = _pf.read_row_group(rg_idx, columns=_feat_cols)
                return _tbl.to_pandas().values.astype(np.float32)

            _rg_arrays = []
            for _rg_i in range(_pq_meta.num_row_groups):
                _rg_n = _pq_meta.row_group(_rg_i).num_rows
                _rg_arrays.append(
                    da.from_delayed(
                        dask.delayed(_read_rg)(_rg_i),
                        shape=(_rg_n, len(_feat_cols)),
                        dtype=np.float32,
                    )
                )
            _feat_arr = da.concatenate(_rg_arrays, axis=0) if _rg_arrays else da.empty(
                (0, len(_feat_cols)), dtype=np.float32
            )

            d = anndata.AnnData(
                X=_feat_arr,
                obs=_obs_df,
                var=pd.DataFrame(index=_feat_cols),
            )
            # Collect source info outside the loop so anndata.concat does not
            # lose it — only the first file's uns survives concat.
            parquet_sources.append({
                "path":           path,
                "feat_cols":      _feat_cols,
                "n_row_groups":   _pq_meta.num_row_groups,
                "row_group_sizes": [
                    _pq_meta.row_group(i).num_rows
                    for i in range(_pq_meta.num_row_groups)
                ],
            })
        elif path.lower().endswith(".h5ad"):
            d = anndata.read_h5ad(path)   # h5py-backed; X is numpy → in-memory path
            if features is not None and len(features) > 0:
                d = d[:, features]
        else:
            zarr_is_remote = path.lower().startswith(("s3://", "gs://", "az://", "abfs://"))
            d = read_anndata_zarr(path, dask=True)
            if features is not None and len(features) > 0:
                d = d[:, features]
        data_arrays.append(d)
    if len(data_arrays) == 0:
        raise RuntimeError("No data found.")

    data = (
        data_arrays[0]
        if len(data_arrays) == 1
        else anndata.concat(data_arrays, index_unique="-")
    )

    # Set source metadata on the final (possibly concatenated) object so that
    # _apply_filter_inmem always sees the full list regardless of how many
    # input files were passed.
    if parquet_sources:
        data.uns["_parquet_sources"] = parquet_sources
    if zarr_is_remote is not None:
        data.uns["_zarr_is_remote"] = zarr_is_remote

    return data


def _get_names_from_pd_query(source) -> set[str]:
    tokens = tokenize_string(source)
    result = set()

    for token in tokens:
        if token[0] == NAME or token[0] == BACKTICK_QUOTED_STRING:
            result.add(token[1])
    return result


def infer_feature_columns(df: dd.DataFrame | dd.DataFrame) -> Sequence[str]:
    """Find Cell Profiler named feature columns from a data frame.

    :param df: Data frame
    :return: Feature column names
    """
    compartments = {"Cells", "Nuclei", "Cytoplasm"}
    data_types = {
        "AreaShape",
        "Correlation",
        "Granularity",
        "Intensity",
        "Location",
        "Neighbors",
        "ObjectSkeleton",
        "RadialDistribution",
        "Spots",
        "Texture",
    }
    features = []
    columns = df.columns[~df.columns.str.contains(_metadata_columns_whitelist_str)]
    for i in range(len(columns)):
        tokens = columns[i].split("_")
        if (
            tokens[0] in compartments
            and tokens[1] in data_types
            and np.issubdtype(df[columns[i]], np.number)
        ):
            features.append(columns[i])
    return features
