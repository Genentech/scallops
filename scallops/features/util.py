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

    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if not pd.api.types.is_string_dtype(obs[c]) and pd.api.types.is_object_dtype(
            obs[c]
        ):
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
    for path in paths:
        if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
            df = pd.read_parquet(path)
            d = pandas_to_anndata(df, features)
        else:
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
    return data


def _read_parquet_for_map(
    paths: Sequence[str], features: Sequence[str] | None = None
) -> anndata.AnnData:
    """Read parquet files for the map pipeline.

    Reads obs columns eagerly (needed for label filtering and groupby) and
    stores source metadata in ``uns["_parquet_sources"]`` so that
    ``_apply_filter_inmem`` can run the PyArrow dataset scanner directly on
    the original files rather than materialising the full feature matrix.

    Zarr / h5ad inputs fall through to ``_read_data``.
    """
    import pyarrow as _pa
    import pyarrow.fs as _pafs
    import pyarrow.parquet as _pq

    parquet_sources: list = []
    zarr_is_remote:  bool | None = None
    data_arrays: list = []

    for path in paths:
        if path.lower().endswith(".h5ad"):
            # h5ad is HDF5 — load fully into memory (numpy X → in-memory filter path)
            d = anndata.read_h5ad(path)
            if features is not None and len(features) > 0:
                d = d[:, features]
            data_arrays.append(d)
            continue

        if not (path.lower().endswith(".parquet") or path.lower().endswith(".pq")):
            # Zarr or other dask-readable format
            zarr_is_remote = path.lower().startswith(("s3://", "gs://", "az://", "abfs://"))
            d = read_anndata_zarr(path, dask=True)
            if features is not None and len(features) > 0:
                d = d[:, features]
            data_arrays.append(d)
            continue

        # ── Schema + metadata (no data yet) ──────────────────────────────
        _pa_fs, _pa_fpath = _pafs.FileSystem.from_uri(path)
        with _pa_fs.open_input_file(_pa_fpath) as _f:
            _schema  = _pq.read_schema(_f)
            _pq_meta = _pq.read_metadata(_f)

        _all_cols  = _schema.names
        _feat_cols = list(features) if features else [
            c for c in _all_cols
            if c.split("_")[0] in {"Cells", "Nuclei", "Cytoplasm"}
        ]
        _feat_set = set(_feat_cols)

        # Obs: skip list/struct typed columns (e.g. barcode_Q_0)
        _obs_cols = [
            c for c in _all_cols
            if c not in _feat_set
            and not _pa.types.is_list(_schema.field(c).type)
            and not _pa.types.is_large_list(_schema.field(c).type)
            and not _pa.types.is_struct(_schema.field(c).type)
        ]

        # ── Obs: eager read, restores named RangeIndex from parquet footer ─
        _obs_df = _pq.read_table(
            _pa_fpath, columns=_obs_cols, filesystem=_pa_fs, pre_buffer=True,
        ).to_pandas()

        if _obs_df.index.name in (None, "__index_level_0__"):
            _obs_df.index.name = (
                "label" if "label" not in _obs_df.columns else "__cell_id__"
            )
        _obs_df.index = _obs_df.index.astype(str)
        for _c in _obs_df.columns:
            if pd.api.types.is_object_dtype(_obs_df[_c]):
                _obs_df[_c] = _obs_df[_c].astype(str)

        # ── X: placeholder dask array (never computed; scanner reads directly) ─
        import dask
        import dask.array as _da

        def _read_rg(rg_idx: int, _path: str = path, _fc: list = _feat_cols) -> np.ndarray:
            import pyarrow.fs as _pf2
            import pyarrow.parquet as _pq2
            _fs2, _fp2 = _pf2.FileSystem.from_uri(_path)
            with _fs2.open_input_file(_fp2) as _f2:
                return _pq2.ParquetFile(_f2, pre_buffer=True).read_row_group(
                    rg_idx, columns=_fc
                ).to_pandas().values.astype(np.float32)

        _rg_arrays = [
            _da.from_delayed(
                dask.delayed(_read_rg)(i),
                shape=(_pq_meta.row_group(i).num_rows, len(_feat_cols)),
                dtype=np.float32,
            )
            for i in range(_pq_meta.num_row_groups)
        ]
        _feat_arr = _da.concatenate(_rg_arrays, axis=0) if _rg_arrays else _da.empty(
            (0, len(_feat_cols)), dtype=np.float32
        )

        d = anndata.AnnData(
            X=_feat_arr, obs=_obs_df, var=pd.DataFrame(index=_feat_cols),
        )

        parquet_sources.append({
            "path":            path,
            "feat_cols":       _feat_cols,
            "n_row_groups":    _pq_meta.num_row_groups,
            "row_group_sizes": [_pq_meta.row_group(i).num_rows
                                for i in range(_pq_meta.num_row_groups)],
        })
        data_arrays.append(d)

    if len(data_arrays) == 0:
        raise RuntimeError("No data found.")

    if len(data_arrays) == 1:
        data = data_arrays[0]
    else:
        # anndata.concat with inner join on var: only features present in ALL
        # files survive.  Log any that are dropped so users know.
        all_feat_sets = [set(d.var.index) for d in data_arrays]
        union_feats   = set.union(*all_feat_sets)
        inter_feats   = set.intersection(*all_feat_sets)
        dropped = union_feats - inter_feats
        if dropped:
            logger.warning(
                "%d feature(s) dropped because they are absent from at least "
                "one input file (inner join across %d files). "
                "Examples: %s",
                len(dropped), len(data_arrays),
                ", ".join(sorted(dropped)[:5]) + (" …" if len(dropped) > 5 else ""),
            )
        data = anndata.concat(data_arrays, index_unique="-")

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
