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


def _is_obs_col(column: str) -> bool:
    """Return True for compartment-prefixed columns that are cell metadata, not
    morphological measurements.

    CellProfiler output mixes spatial/QC columns (e.g. centroid coordinates,
    boundary flags, neighbor counts) with measurement features (intensity,
    texture, etc.) under the same ``Cells_`` / ``Nuclei_`` / ``Cytoplasm_``
    namespace.  The former belong in ``obs``; the latter in ``X``.

    Pattern rules (no hardcoded names):
    - Coordinate columns: name ends with ``_X`` or ``_Y``
    - Boundary QC: name contains ``IntersectsBoundary``
    - Topology (object refs only): name contains ``_Neighbors_ObjectNumber``
      (quantitative neighbor measurements like Distance/Angle/Count stay in X)
    - ISS / PHENO QC: name contains ``_ISS_PHENO`` or ends with
      ``-pheno-to-iss-qc`` (all registration QC columns excluded from X
      regardless of the base measurement type)
    """
    import re
    if re.search(r'_(X|Y)$', column):
        return True
    if 'IntersectsBoundary' in column:
        return True
    if '_Neighbors_' in column and 'ObjectNumber' in column:
        return True
    if '_ISS_PHENO' in column or column.endswith('-pheno-to-iss-qc'):
        return True
    return False


def _keep_channels(
    columns: Sequence[str],
    valid_channels: "set[str] | None",
    include_measurement_types: "set[str] | None" = None,
) -> list[str]:
    """Filter feature column names by channel and/or measurement type.

    A column is kept when **either** of the following is true:

    1. **Channel rule** — every ``Channel<N>`` token in its name has ``<N>``
       in *valid_channels* (or the column has no channel tokens at all).
    2. **Measurement-type override** — the second ``_``-separated token of the
       column name (the measurement type, e.g. ``"Spots_Count"``) matches any
       value in *include_measurement_types*.  This lets you pull in specific
       feature families from channels that are otherwise excluded — for example,
       adding ``Spots_Count`` features from FISH channels (0–3) while keeping
       only IF channels (4–13) for everything else.

    When both *valid_channels* and *include_measurement_types* are ``None``
    all columns are returned unchanged.

    :param columns: Iterable of feature column names to filter.
    :param valid_channels: Set of channel-number strings to keep for the
        channel rule.  ``None`` = keep all channels.
    :param include_measurement_types: Set of measurement-type strings
        (e.g. ``{"Spots_Count"}``).  Features whose type matches are kept
        regardless of their channel.  ``None`` = no type-based override.
    :return: Filtered list of column names.

    Example::

        # IF channels 4–13 + Spots_Count from all channels (incl. FISH 0–3)
        feats = _keep_channels(all_feats,
                               valid_channels={str(i) for i in range(4, 14)},
                               include_measurement_types={"Spots_Count"})
    """
    if valid_channels is None and not include_measurement_types:
        return list(columns)
    import re
    _chan_pat = re.compile(r'Channel(\d+)')
    out = []
    for col in columns:
        # Measurement-type override: second token of underscore-separated name
        # e.g. "Cells_Spots_Count_Channel0" → type token = "Spots_Count"
        # (join tokens 1 onward until we hit a Channel token or end)
        if include_measurement_types:
            parts = col.split("_")
            # Reconstruct measurement type: everything between compartment and
            # first Channel/numeric-looking token.
            mtype_parts = []
            for p in parts[1:]:
                if _chan_pat.match(p) or (p.isdigit()):
                    break
                mtype_parts.append(p)
            mtype = "_".join(mtype_parts)
            if mtype in include_measurement_types:
                out.append(col)
                continue

        if valid_channels is None:
            out.append(col)
            continue

        refs = _chan_pat.findall(col)
        if all(r in valid_channels for r in refs):   # vacuously True when refs=[]
            out.append(col)
    return out


def _expand_pattern_inputs(
    paths: Sequence[str],
    pattern: "str | None" = None,
) -> "list[tuple[str, dict[str, str]]]":
    """Expand inputs that contain ``{name}`` capture groups.

    Supports two calling styles:

    1. **Inline pattern** — the ``{name}`` groups are embedded in the path
       itself::

           _expand_pattern_inputs(["s3://bucket/ER-{plate}-{well}.zarr"])
           # → [("s3://bucket/ER-A-1.zarr", {"plate": "A", "well": "1"}), …]

    2. **Separate pattern** — paths are directories and *pattern* is the
       filename template::

           _expand_pattern_inputs(["s3://bucket/ER/"], pattern="ER-{plate}-{well}.zarr")

    For paths without ``{name}`` groups (and no *pattern*), the function
    behaves like a passthrough that returns ``(path, {})`` pairs.

    :param paths: Input paths or glob patterns.
    :param pattern: Optional filename pattern with ``{name}`` capture groups.
        When given, each path is treated as a directory to search.
    :return: List of ``(resolved_path, captures_dict)`` pairs.
    """
    import re as _re
    import fsspec as _fsspec
    from scallops.io import _create_file_regex

    _CAPTURE_RE = _re.compile(r'\{(\w+)[^}]*\}')

    def _has_captures(s: str) -> bool:
        return bool(_CAPTURE_RE.search(s))

    def _list_dir(dir_url: str) -> list[str]:
        """List immediate children of a directory (local or cloud)."""
        _fs, _path = _fsspec.url_to_fs(dir_url)
        try:
            children = _fs.ls(_path, detail=False)
        except Exception:
            return []
        return [_fs.unstrip_protocol(c) for c in children]

    def _match_entries(dir_url: str, pat: str) -> "list[tuple[str, dict]]":
        """List dir, match each entry against pat, return (path, captures) pairs."""
        regex, _suffix, _keys = _create_file_regex(pat)
        results = []
        for entry in sorted(_list_dir(dir_url)):
            basename = entry.rstrip("/").rsplit("/", 1)[-1]
            m = regex.fullmatch(basename)
            if m:
                results.append((entry, m.groupdict()))
        return results

    result: list[tuple[str, dict]] = []

    for path in paths:
        if pattern is not None:
            # Directory + separate pattern
            entries = _match_entries(path.rstrip("/") + "/", pattern)
            if not entries:
                raise FileNotFoundError(
                    f"--input-pattern {pattern!r} matched no files in {path!r}"
                )
            result.extend(entries)
        elif _has_captures(path):
            # Inline pattern: split at first {, use prefix as directory
            prefix_end = _CAPTURE_RE.search(path).start()
            dir_url    = path[:prefix_end].rstrip("/") + "/"
            pat        = path[prefix_end - len(path.rsplit("/", 1)[-1].split("{")[0]):]
            # Extract just the filename template portion
            _last_slash = path[:prefix_end].rfind("/")
            dir_url  = path[:_last_slash + 1] if _last_slash >= 0 else "./"
            filename_pat = path[_last_slash + 1:]
            entries = _match_entries(dir_url, filename_pat)
            if not entries:
                raise FileNotFoundError(
                    f"Pattern {path!r} matched no files in {dir_url!r}"
                )
            result.extend(entries)
        else:
            # Plain path — no expansion, no captures
            result.append((path, {}))

    if any(captures for _, captures in result):
        logger.info(
            "pattern input: expanded %d path(s) → %d file(s) with captures: %s",
            len(paths), len(result),
            sorted({k for _, c in result for k in c}),
        )
    return result


def _read_map_inputs(
    paths: Sequence[str],
    features: Sequence[str] | None = None,
    valid_channels: "set[str] | None" = None,
    obs_captures: "dict[str, dict[str, str]] | None" = None,
    include_measurement_types: "set[str] | None" = None,
    obs_force: "set[str] | None" = None,
) -> anndata.AnnData:
    """Read parquet/zarr inputs for the map pipeline.

    :param obs_captures: Optional mapping ``path → {col: value}`` returned by
        :func:`_expand_pattern_inputs`.  For each file, columns in the capture
        dict are injected into ``obs`` if they are not already present.
    :param include_measurement_types: Measurement-type names (e.g.
        ``{"Spots_Count"}``) to include regardless of channel.  Passed through
        to :func:`_keep_channels`.
    """
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

    def _inject_captures(d: anndata.AnnData, path: str) -> anndata.AnnData:
        """Inject obs_captures metadata for *path* into *d* (missing columns only)."""
        if not obs_captures:
            return d
        for col, val in (obs_captures.get(path) or {}).items():
            if col not in d.obs.columns:
                d.obs[col] = val
        return d

    for path in paths:
        if path.lower().endswith(".h5ad"):
            # h5ad is HDF5 — load fully into memory (numpy X → in-memory filter path)
            d = anndata.read_h5ad(path)
            if features is not None and len(features) > 0:
                d = d[:, features]
            elif valid_channels is not None:
                _keep = _keep_channels(list(d.var.index), valid_channels,
                                       include_measurement_types)
                d = d[:, _keep]
            d = _inject_captures(d, path)
            data_arrays.append(d)
            continue

        if not (path.lower().endswith(".parquet") or path.lower().endswith(".pq")):
            # Zarr or other dask-readable format
            zarr_is_remote = path.lower().startswith(("s3://", "gs://", "az://", "abfs://"))
            d = read_anndata_zarr(path, dask=True)
            if features is not None and len(features) > 0:
                d = d[:, features]
            elif valid_channels is not None:
                _keep = _keep_channels(list(d.var.index), valid_channels,
                                       include_measurement_types)
                d = d[:, _keep]
            d = _inject_captures(d, path)
            data_arrays.append(d)
            continue

        # ── Schema + metadata (no data yet) ──────────────────────────────
        _pa_fs, _pa_fpath = _pafs.FileSystem.from_uri(path)
        with _pa_fs.open_input_file(_pa_fpath) as _f:
            _schema  = _pq.read_schema(_f)
            _pq_meta = _pq.read_metadata(_f)

        _all_cols  = _schema.names
        _COMPARTMENTS = {"Cells", "Nuclei", "Cytoplasm"}
        _feat_cols = list(features) if features else [
            c for c in _all_cols
            if c.split("_")[0] in _COMPARTMENTS and not _is_obs_col(c)
        ]
        # Apply channel filter (e.g. restrict to IF channels 4-13)
        _feat_cols = _keep_channels(_feat_cols, valid_channels,
                                    include_measurement_types)
        # Force label-filter columns into obs so the null guard never triggers
        # a full feature-matrix materialisation (CellProfiler filter columns
        # like Cells_Location_* / Nuclei_Correlation_* have compartment prefixes
        # and would otherwise be classified as lazy feature columns).
        if obs_force:
            _feat_cols = [c for c in _feat_cols if c not in obs_force]
        _feat_set = set(_feat_cols)
        # All compartment-prefix columns not selected as features should be
        # DROPPED (not loaded into obs).  With channel selection active,
        # ~3,000 excluded FISH/other-channel columns would otherwise fall into
        # obs and inflate the obs DataFrame by hundreds of GB.
        _all_compartment_cols = {
            c for c in _all_cols if c.split("_")[0] in _COMPARTMENTS
        }

        # Obs: skip list/struct typed columns AND dropped compartment columns
        _obs_cols = [
            c for c in _all_cols
            if c not in _feat_set
            and c not in (_all_compartment_cols - set(obs_force or []))
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
                # Keep Python bool values as bool so label filters like
                # ``== False`` work correctly.  Convert everything else to str.
                _non_null = _obs_df[_c].dropna()
                if len(_non_null) and _non_null.apply(
                    lambda x: isinstance(x, (bool, np.bool_))
                ).all():
                    pass  # leave as nullable-bool object dtype
                else:
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
        d = _inject_captures(d, path)
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
