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

    data = (
        df[features].values
        if not isinstance(df, dd.DataFrame)
        else df[features].to_dask_array(lengths=True)
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
    """Slice an AnnData object without copy-on-write AnnData's behavior.
    Note that this method only slices the fields `X`, `obs`, and `var`.

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
    if obs_indices is not None:
        X = X[obs_indices]
    if var_indices is not None:
        X = X[:, var_indices]
    obs = data.obs.iloc[obs_indices] if obs_indices is not None else data.obs
    var = data.var.iloc[var_indices] if var_indices is not None else data.var
    return anndata.AnnData(X=X, obs=obs, var=var)


def _update_coords(
    df: pd.DataFrame,
    df_coords: bool | str | Sequence[str],
    coord_name: str,
    coords_keys: set,
    xarray_coords: dict,
):
    if df_coords:
        xarray_coords[coord_name] = df.index
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
    adata: anndata.AnnData,
    obs_coords: bool | str | Sequence[str] = True,
    var_coords: bool | str | Sequence[str] = False,
) -> xr.DataArray:
    coords = dict()
    coords_keys = {"obs", "var"}
    _update_coords(
        df=adata.obs,
        df_coords=obs_coords,
        coord_name="obs",
        coords_keys=coords_keys,
        xarray_coords=coords,
    )
    _update_coords(
        df=adata.var,
        df_coords=var_coords,
        coord_name="var",
        coords_keys=coords_keys,
        xarray_coords=coords,
    )
    return xr.DataArray(adata.X, dims=("obs", "var"), name="", coords=coords)


def _join_metadata(
    adata: anndata.AnnData, join_df: pd.DataFrame | dd.DataFrame, on: Sequence[str]
):
    # match data type
    for field in on:
        if join_df[field].dtype != adata.obs[field].dtype:
            adata.obs[field] = adata.obs[field].astype(join_df[field].dtype)
    if isinstance(join_df, dd.DataFrame):
        join_df = join_df.compute()
    join_df = join_df.set_index(on)
    adata.obs = adata.obs.join(join_df, on=on)


def _read_data(
    paths: Sequence[str] | str, features: Sequence[str] | None = None
) -> anndata.AnnData:
    if isinstance(paths, str):
        paths = [paths]
    assert len(paths) == len(set(paths)), "Duplicate path"
    data_arrays = []
    for path in paths:
        if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
            df = dd.read_parquet(path)
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
