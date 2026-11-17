"""SCALLOPS Zarr I/O Utility Functions Module.

This module provides a collection of utility functions for reading and writing data in
Zarr format, specifically tailored for OME-Zarr. These functions facilitate the
handling of image and label data, including metadata management and downsampling.

Authors:
- The SCALLOPS development team
"""

import logging
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property
from os import environ
from pathlib import Path
from typing import Any, Literal

import dask
import dask.array as da
import fsspec
import numpy as np
import ome_types
import xarray as xr
import zarr
from dask.array import from_zarr
from dask.delayed import Delayed
from dask.graph_manipulation import bind
from numpy.typing import NDArray
from ome_zarr import USE_DASK_ARRAY_KWARGS
from ome_zarr.axes import KNOWN_AXES
from ome_zarr.format import Format, FormatV04, FormatV05
from ome_zarr.io import parse_url
from ome_zarr.types import JSONDict
from ome_zarr.writer import write_multiscale
from zarr.errors import ContainsGroupError
from zarr.storage import StoreLike

from scallops._anndata_io import write_basic_dask_dask_dense  # noqa: F401
from scallops.experiment.abc import _LazyLoadData
from scallops.utils import _fix_json

logger = logging.getLogger("scallops")


def _current_format() -> Format:
    version = environ.get("SCALLOPS_OME_ZARR_VERSION", "0.4")
    if version == "0.4":
        return FormatV04()
    elif version == "0.5":
        return FormatV05()
    raise ValueError(f"Unrecognized version: {version}")


def _tolerate_concurrent_create(
    create: Callable[[], zarr.Group], open_existing: Callable[[], zarr.Group]
) -> zarr.Group:
    """Call ``create``, falling back to ``open_existing`` if it loses a create race.

    ``create`` is expected to be check-then-create: it looks a group up and creates
    it only when the lookup misses. Processes that share an output store -- one
    registration or stitching task per well, all writing the same zarr -- can all
    miss, and every loser fails with ``ContainsGroupError``. The group they wanted
    now exists, so ``open_existing`` should open it with a plain read (no create
    attempt of its own), which cannot race no matter how many losers there are.

    :param create: Attempts to create the group; raises ``ContainsGroupError`` if
        another process created it first.
    :param open_existing: Opens the group assuming it already exists.
    :return: The existing or newly created group.
    """
    try:
        return create()
    except ContainsGroupError:
        logger.debug("Group was created concurrently; opening the existing one")
        return open_existing()


def _require_group(parent: zarr.Group, name: str) -> zarr.Group:
    """Get or create a child group, tolerating a concurrent writer.

    An *array* at ``name`` still raises, as it does without this wrapper.

    :param parent: Group to create the child under.
    :param name: Name of the child group.
    :return: The existing or newly created group.
    """
    return _tolerate_concurrent_create(
        lambda: parent.require_group(name, overwrite=False),
        lambda: parent[name],
    )


def _create_array_kwargs(
    fmt: Format | None = None, dims: Sequence[Hashable] | None = None
) -> dict[str, Any]:
    """Return keyword arguments for :meth:`zarr.Group.create_array`.

    :param fmt: OME-Zarr format. Defaults to the current format.
    :param dims: Dimension names of the array. Written as ``dimension_names`` for
        Zarr v3 (OME-Zarr 0.5 requires them to match the axis names).
    :return: The keyword arguments.
    """
    if fmt is None:
        fmt = _current_format()
    zarr_array_kwargs: dict[str, Any] = dict()

    if fmt.zarr_format == 2:
        zarr_array_kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
    elif dims is not None:
        zarr_array_kwargs["dimension_names"] = [str(d) for d in dims]
    return zarr_array_kwargs


def _da_to_zarr_kwargs(
    fmt: Format | None = None, dims: Sequence[Hashable] | None = None
) -> dict[str, Any]:
    """Return keyword arguments for :func:`dask.array.to_zarr`.

    :param fmt: OME-Zarr format. Defaults to the current format.
    :param dims: Dimension names of the array. Written as ``dimension_names`` for
        Zarr v3 (OME-Zarr 0.5 requires them to match the axis names).
    :return: The keyword arguments.
    """
    if fmt is None:
        fmt = _current_format()
    zarr_array_kwargs: dict[str, Any] = dict()
    if fmt.zarr_format == 3 and dims is not None:
        zarr_array_kwargs["dimension_names"] = [str(d) for d in dims]

    if USE_DASK_ARRAY_KWARGS:
        if fmt.zarr_format == 2:
            zarr_array_kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
    elif fmt.zarr_format == 2:
        zarr_array_kwargs["dimension_separator"] = "/"
    if fmt.zarr_format == 2:
        zarr_array_kwargs["zarr_format"] = fmt.zarr_format
    return zarr_array_kwargs


def _get_store_path(group: zarr.Group):
    if hasattr(group.store, "root"):
        return str(group.store.root)
    if hasattr(group.store, "path"):
        return group.store.path
    return ""


def is_ome_zarr_array(node: zarr.Group) -> bool:
    """Check if a Zarr node is an OME-Zarr array.

    This function checks whether the given Zarr group node contains OME-Zarr metadata,
        indicating that it is an OME-Zarr array.

    :param node: zarr.Group
        The Zarr group node to check.
    :return: bool
        True if the node is an OME-Zarr array, False otherwise.

    :example:

        .. code-block:: python

            import zarr
            from scallops.zarr_io import is_ome_zarr_array

            # Create a Zarr group with OME-Zarr metadata
            store = zarr.DirectoryStore("example.zarr")
            root = zarr.group(store=store)
            root.attrs["multiscales"] = [{"version": "0.1"}]

            # Check if the group is an OME-Zarr array
            result = is_ome_zarr_array(root)
            print(result)  # Output: True
    """
    return node is not None and _has_multiscales(node.attrs)


def _get_fs(group: zarr.Group):
    if hasattr(group.store, "fs"):
        return group.store.fs
    return fsspec.url_to_fs(_get_store_path(group))[0]


def _get_sep(group: zarr.Group) -> str:
    if hasattr(group.store, "fs"):
        return group.store.fs.sep
    return "/"


def _omero_channels(channel_names, image_dtype: np.dtype) -> dict:
    # Napari does not like '#' at start of hex color string
    # Hex colors match Napari defaults
    colors = ["00FFFF", "FFFF00", "FF00FF", "FF0000", "008000", "0000FF"]

    max_value = (
        int(np.iinfo(image_dtype).max)
        if np.issubdtype(image_dtype, np.integer)
        else float(np.finfo(image_dtype).max)
    )

    # Napari requires that colors are specified if channel names are specified
    channels = (
        [
            dict(
                label=str(channel_names[i]),
                color=colors[i % len(colors)],
                window=dict(min=0, max=max_value, start=0, end=max_value),
            )
            for i in range(len(channel_names))
        ]
        if not np.isscalar(channel_names)
        else [np.asarray(channel_names).tolist()]
    )
    return dict(channels=channels)


def _create_omero_metadata(
    coords: Mapping[Hashable, Any] | None,
    dims: Sequence[Hashable] | None,
    image_dtype: np.dtype,
) -> dict | None:
    """Create OMERO metadata for a DataArray.

    This function generates OMERO metadata for a given DataArray based on its
    coordinates and dimensions. The metadata includes channel names and their
    corresponding colors, which are required by visualization tools like Napari.

    :param coords: The coordinates of the DataArray.
    :param dims: The dimensions of the DataArray.
    :return: dictionary containing OMERO metadata if channel names are present,
        otherwise None.

    :example:

        .. code-block:: python

            import xarray as xr
            import numpy as np
            from scallops.zarr_io import _create_omero_metadata

            data = np.random.rand(5, 10, 512, 512)
            dims = ("c", "z", "y", "x")
            coords = {"c": ["DAPI", "FITC", "TRITC", "Cy5", "Cy7"]}
            array = xr.DataArray(data, dims=dims, coords=coords)

            # Create OMERO metadata
            omero_metadata = _create_omero_metadata(array.coords, array.dims, data.dtype)
            print(omero_metadata)
            # Output: {'channels': [{'label': 'DAPI', 'color': '00FFFF'},
                {'label': 'FITC', 'color': 'FFFF00'}, ...]}
    """
    if dims is None or coords is None:
        return None

    channel_names = coords["c"] if "c" in dims and "c" in coords else None
    if channel_names is not None:
        if isinstance(channel_names, xr.DataArray):
            channel_names = channel_names.values
        return _omero_channels(channel_names, image_dtype)
    return None


def _fix_attrs(d: dict) -> None:
    """Recursively fix attributes for dictionary serialization.

    This function traverses a dictionary and modifies its values to ensure they are
    serializable. It handles nested dictionaries, lists, and specific object types
    such as `ome_types.OME` and `zarr.Group`.

    :param d: The dictionary containing attributes to be fixed.

    :example:

        .. code-block:: python

            from ome_types import OME
            import zarr

            attrs = {
                "metadata": {
                    "ome": OME(),
                    "group": zarr.group(),
                    "nested": {"ome": OME(), "list": [OME(), zarr.group()]},
                }
            }

            # Fix attributes for JSON serialization
            _fix_attrs(attrs)
            print(attrs)
            # Output: {'metadata': {'ome': {...}, 'group': '...', 'nested':
                {'ome': {...}, 'list': [{...}, '...']}}}
    """
    for key in d:
        value = d[key]
        if isinstance(value, dict):
            _fix_attrs(value)
        elif isinstance(value, ome_types.OME):
            # Hack to prevent OverflowError:
            # Overlong 4 byte UTF-8 sequence detected when encoding string
            d[key] = d[key].model_dump()
        elif isinstance(value, zarr.Group):
            d[key] = str(value)
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], dict):
                    _fix_attrs(value[i])
                elif isinstance(value[i], ome_types.OME):
                    value[i] = value[i].dict()
                elif isinstance(value[i], zarr.Group):
                    value[i] = str(value)


def _attrs_axes_scales(
    image_attrs: dict | None,
    coords: Mapping[Hashable, Any] | None,
    dims: Sequence[Hashable] | None,
    image_dtype: np.dtype,
) -> tuple[dict, list[dict] | None, dict[Hashable, float] | None]:
    """Prepare attributes, axes, and coordinate transformations for Zarr storage.

    Processes the attributes, coordinates, and dimensions of a DataArray to generate
    metadata suitable for Zarr storage. It includes handling OMERO metadata, physical
    pixel sizes, and coordinate transformations.

    :param image_attrs: The attributes of the image DataArray.
    :param coords: The coordinates of the DataArray.
    :param dims: The dimensions of the DataArray.
    :return: A tuple containing:
        - Updated image attributes dictionary.
        - List of axes dictionaries or None.
        - Scale dict or None.
    """

    omero = _create_omero_metadata(coords, dims, image_dtype)
    if image_attrs is None:
        image_attrs = dict()
    image_attrs = image_attrs.copy()
    if omero is not None:
        image_attrs["omero"] = omero

    if coords is not None:
        for key in coords.keys():
            if key not in ("c", "z", "y", "x"):
                vals = coords[key]
                if vals is not None:
                    if isinstance(vals, xr.DataArray):
                        vals = vals.values
                    if isinstance(vals, np.ndarray):
                        vals = vals.tolist()
                    image_attrs[key] = vals

    physical_pixel_sizes = image_attrs.pop("physical_pixel_sizes", None)
    physical_pixel_units = image_attrs.pop("physical_pixel_units", None)

    axes = None
    scale_dict: dict[Hashable, float] | None = None
    if dims is not None:
        axes = []
        scale_dict = dict()
        space_index = 0
        if "z" in dims:
            if physical_pixel_sizes is not None and len(physical_pixel_sizes) == 2:
                physical_pixel_sizes = [1.0] + list(physical_pixel_sizes)
            if physical_pixel_units is not None and len(physical_pixel_units) == 2:
                physical_pixel_units = ["micrometer"] + list(physical_pixel_units)

        for d in dims:
            axis = {"name": d, "type": KNOWN_AXES.get(d)}
            if physical_pixel_sizes is not None and axis["type"] == "space":
                scale_dict[d] = physical_pixel_sizes[space_index]
                unit = (
                    physical_pixel_units[space_index]
                    if physical_pixel_units is not None
                    else "micrometer"
                )
                axis["unit"] = unit
                space_index = space_index + 1
            axes.append(axis)

    _fix_attrs(image_attrs)
    image_attrs = _fix_json(image_attrs)
    return image_attrs, axes, scale_dict


def _group_name(name: str) -> str:
    """Return the Zarr group name an image or label is stored under.

    :param name: Image or label name.
    :return: ``name`` with ``/`` replaced by ``_``, since Zarr treats ``/`` as a path
        separator.
    """
    return name.replace("/", "_")


def _write_zarr_image(
    name: str | None,
    root: zarr.Group | str | Path,
    image: da.Array | NDArray | xr.DataArray,
    metadata: dict[str, Any] | None = None,
    group: str | None = "images",
    zarr_format: Literal["ome_zarr", "zarr"] = "ome_zarr",
    compute: bool = True,
    storage_options: JSONDict | None = None,
    group_metadata: dict | None = None,
) -> list[Delayed]:
    """Write image in zarr format.

    :param name: Zarr group name to store image
    :param root: Root zarr group.
    :param image: Image to write.
    :param group: Group name under root to write image to
    :param metadata: Additional metadata to store
    :param zarr_format: Either ome_zarr or zarr. Use zarr for storing non-ome zarr
        compliant images with dimensions other than (t,c,z,y,x)
    :param compute: If true compute immediately otherwise a list
        of :class:`dask.delayed.Delayed` is returned.
    :param storage_options: Options to be passed on to the storage backend.
    :param group_metadata: Optional group level metadata.
    :return: Empty list if the compute flag is True, otherwise it returns a list
        of :class:`dask.delayed.Delayed` representing the value to be computed by dask.
    """

    if isinstance(root, (str, Path)):
        root = open_ome_zarr(root, mode="a")
    dest_grp = root
    if group is not None and name is not None:
        images_grp = _require_group(root, group)
        dest_grp = images_grp.create_group(_group_name(name), overwrite=True)
        if group_metadata is not None:
            _add_metadata(dest_grp, group_metadata)
    image_attrs = None
    coords: Mapping[Hashable, Any] | None = None
    dims: Sequence[Hashable] | None = None
    if isinstance(image, xr.DataArray):
        data = image.data
        image_attrs = image.attrs.copy()
        coords = image.coords
        dims = image.dims
    else:
        data = image
        if image.ndim == 2:
            dims = ["y", "x"]
        elif image.ndim == 5:
            dims = ["t", "c", "z", "y", "x"]
        elif zarr_format == "ome_zarr":
            raise ValueError("Unable to infer image axes.")
    return write_zarr(
        grp=dest_grp,
        data=data,
        image_attrs=image_attrs,
        coords=coords,
        dims=dims,
        storage_options=storage_options,
        metadata=metadata,
        zarr_format=zarr_format,
        compute=compute,
    )


def _write_zarr_labels(
    name: str,
    root: zarr.Group | str | Path,
    labels: NDArray | xr.DataArray | da.Array,
    metadata: dict[str, Any] | None = None,
    group_metadata: dict[str, Any] | None = None,
    compute: bool = True,
    storage_options: JSONDict | None = None,
) -> list[Delayed]:
    """Write label in zarr format.

    :param name: Zarr group name to store label
    :param root: Root zarr group.
    :param labels: Labels to write.
    :param metadata: Optional label metadata.
    :param group_metadata: Optional group level metadata.
    :param compute: If true compute immediately otherwise a list
        of :class:`dask.delayed.Delayed` is returned.
    :param storage_options: Optional storage options.
    :return: Empty list if the compute flag is True, otherwise it returns a list
        of :class:`dask.delayed.Delayed` representing the value to be computed by dask.
    """

    group_metadata = group_metadata.copy() if group_metadata is not None else dict()
    if "image-label" not in group_metadata:
        group_metadata["image-label"] = {}

    if isinstance(root, (str, Path)):
        root = open_ome_zarr(root, mode="a")
    return _write_zarr_image(
        name=name,
        root=root,
        image=labels,
        metadata=metadata,
        group_metadata=group_metadata,
        group="labels",
        compute=compute,
        storage_options=storage_options,
    )


def _ome_attrs(attrs: Mapping) -> Mapping:
    """Return the OME-Zarr attributes of a group.

    OME-Zarr 0.5 nests them under ``"ome"``. Keys under ``"ome"`` take precedence
    over top-level keys, which are kept so that 0.4 stores and 0.5 stores written
    with OME keys at the top level can still be read.

    Calling it on attributes it already returned is cheap (they have no ``"ome"``
    key, so they are returned as is), so helpers that take attributes call it
    rather than requiring callers to unwrap first.

    :param attrs: Zarr group attributes.
    :return: The OME-Zarr attributes (``multiscales``, ``omero``, ``labels``, ...).
    """
    ome = attrs.get("ome")
    if isinstance(ome, Mapping):
        return {**{k: v for k, v in attrs.items() if k != "ome"}, **ome}
    return attrs


# Keys that OME-Zarr 0.5 nests under "ome"
_OME_KEYS = frozenset(
    {
        "multiscales",
        "omero",
        "image-label",
        "labels",
        "plate",
        "well",
        "bioformats2raw.layout",
    }
)


def _add_metadata(
    grp: zarr.Group, attrs: Mapping[str, Any], fmt: Format | None = None
) -> None:
    """Add attributes to a group, laid out for the OME-Zarr format ``fmt``.

    For OME-Zarr 0.5, the OME keys (``multiscales``, ``omero``, ``image-label``,
    ``labels``, ...) and any ``"ome"`` entry in ``attrs`` are written under
    ``"ome"`` together with the format version. Other keys are written at the top
    level. Dict values are merged into existing dict values; other values replace
    them.

    This follows :func:`ome_zarr.writer.add_metadata`, which can't be used directly:
    it requires the group's Zarr format to match ``fmt`` (2 for 0.4, 3 for 0.5),
    while scallops also writes OME-Zarr 0.4 metadata to Zarr v3 groups.

    :param grp: Zarr group to update.
    :param attrs: Attributes to add.
    :param fmt: OME-Zarr format. Defaults to the current format.
    """

    def merge(existing: Mapping, new: Mapping) -> dict:
        merged = dict(existing)
        for key, value in new.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    fmt = fmt or _current_format()
    attrs = dict(attrs)
    if fmt.version == "0.5":
        ome = dict(attrs.pop("ome", None) or {})
        for key in _OME_KEYS & attrs.keys():
            ome[key] = attrs.pop(key)
        if ome:
            existing = grp.attrs.get("ome")
            ome = merge(existing if isinstance(existing, Mapping) else {}, ome)
            attrs["ome"] = {**ome, "version": fmt.version}
    if attrs:
        grp.attrs.update(merge({k: grp.attrs.get(k) for k in attrs}, attrs))


def _has_multiscales(attrs: Mapping | None) -> bool:
    """Whether group attributes describe an OME-Zarr image (0.4 or 0.5).

    :param attrs: Zarr group attributes.
    :return: True if the attributes contain ``multiscales``.
    """
    return isinstance(attrs, Mapping) and "multiscales" in _ome_attrs(attrs)


def _multiscale0(attrs: Mapping | None) -> Mapping[str, Any] | None:
    """Return the first ``multiscales`` entry of an OME-Zarr image (0.4 or 0.5).

    :param attrs: Zarr group attributes.
    :return: ``multiscales[0]`` or None if there is none.
    """
    if not isinstance(attrs, Mapping):
        return None
    multiscales = _ome_attrs(attrs).get("multiscales")
    return multiscales[0] if multiscales else None


def _axis_scales(
    multiscale0: Mapping[str, Any], names: Sequence[str]
) -> tuple[list[str], list[float], list[str]]:
    """Return the scale of the full-resolution dataset for the axes in ``names`` that
    have a unit, using its first transformation of type ``scale``.

    :param multiscale0: First ``multiscales`` entry.
    :param names: Axis names to look up, in the order to return them.
    :return: ``(axis names, scales, units)`` for the axes found with a unit.
    """
    axes = multiscale0.get("axes") or []
    dims = [axis.get("name") for axis in axes]
    datasets = multiscale0.get("datasets") or []
    transforms = (
        (datasets[0].get("coordinateTransformations") or []) if datasets else []
    )
    scale = next((t.get("scale") for t in transforms if t.get("type") == "scale"), None)
    found: tuple[list[str], list[float], list[str]] = ([], [], [])
    if scale is None:
        return found
    for name in names:
        if name in dims:
            index = dims.index(name)
            unit = axes[index].get("unit")
            if unit not in (None, ""):
                found[0].append(name)
                found[1].append(scale[index])
                found[2].append(unit)
    return found


def _multiscales_metadata(attrs: Mapping | None) -> Mapping[str, Any]:
    """Return the scallops metadata of an OME-Zarr image
    (``multiscales[0]["metadata"]``), or an empty dict if there is none.

    :param attrs: Zarr group attributes.
    :return: The metadata.
    """
    multiscale0 = _multiscale0(attrs)
    return (multiscale0.get("metadata") if multiscale0 else None) or {}


def _dataset_path(attrs: Mapping | None) -> str:
    """Return the path of the full-resolution array of an OME-Zarr image
    (``multiscales[0]["datasets"][0]["path"]``), or ``"0"`` if it is not given.

    :param attrs: Zarr group attributes.
    :return: The array path relative to the group.
    """
    multiscale0 = _multiscale0(attrs)
    datasets = (multiscale0.get("datasets") if multiscale0 else None) or []
    return datasets[0].get("path", "0") if datasets else "0"


def _physical_pixel_size(
    attrs: Mapping | None,
) -> tuple[list[float], str | None, str | None] | None:
    """Return the (y, x) physical pixel size of an OME-Zarr image and its units.

    Uses ``physical_size_y``/``physical_size_x`` from ``multiscales[0]["metadata"]``
    if present, otherwise the scale transformation of the full-resolution dataset if
    the ``y`` and ``x`` axes have a unit (as written by scallops).

    :param attrs: Zarr group attributes.
    :return: ``([y, x], y_unit, x_unit)`` or None if the size is unknown.
    """
    multiscale0 = _multiscale0(attrs)
    if multiscale0 is None:
        return None
    metadata = multiscale0.get("metadata") or {}
    if "physical_size_y" in metadata and "physical_size_x" in metadata:
        return (
            [metadata["physical_size_y"], metadata["physical_size_x"]],
            metadata.get("physical_size_y_unit"),
            metadata.get("physical_size_x_unit"),
        )
    names, scales, units = _axis_scales(multiscale0, ("y", "x"))
    if names == ["y", "x"]:
        return scales, units[0], units[1]
    return None


def _create_zarr_attrs(
    fmt, grp, dims, image_attrs, axes, scale_dict, image_label: dict | None = None
):
    """Create OME-Zarr group attributes for a single-scale image.

    :param fmt: OME-Zarr format.
    :param grp: Zarr group the attributes are for (its name is used).
    :param dims: Dimension names.
    :param image_attrs: Image attributes; ``omero`` is written as channel metadata
        and the rest as multiscales metadata.
    :param axes: OME-Zarr axes.
    :param scale_dict: Map from dimension name to scale.
    :param image_label: If not None, write as ``image-label`` metadata so that the
        group is recognized as a label image.
    :return: The OME attributes, to be written with :func:`_add_metadata`.
    """
    datasets: list[dict[str, Any]] = [{"path": "s0"}]
    if scale_dict is None:
        scale_dict = dict()
    scale = []
    translation = []
    if dims is None:
        return {}
    for d in dims:
        scale.append(scale_dict.get(d, 1.0))
        translation.append(0.0)
    coordinate_transformations = [
        {"type": "scale", "scale": scale},
        {"type": "translation", "translation": translation},
    ]
    datasets[0]["coordinateTransformations"] = coordinate_transformations

    multiscales: list[dict[str, Any]] = [
        dict(version=fmt.version, datasets=datasets, name=grp.name)
    ]
    zarr_attrs: dict[str, Any] = {"multiscales": multiscales}
    if axes is not None:
        multiscales[0]["axes"] = axes

    image_attrs = dict(image_attrs or {})
    omero = image_attrs.pop("omero", None)
    if omero:
        zarr_attrs["omero"] = dict(omero)
    if image_label is not None:
        zarr_attrs["image-label"] = image_label
    multiscales[0]["metadata"] = image_attrs
    if fmt.version == "0.5":
        # OME-Zarr 0.5 holds the version in "ome", not in each multiscales entry.
        del multiscales[0]["version"]
    return zarr_attrs


def write_zarr(
    grp: zarr.Group,
    data: NDArray | da.Array | xr.DataArray,
    image_attrs: dict | None,
    coords: Mapping[Hashable, Any] | None,
    dims: Sequence[Hashable] | None,
    metadata: dict[str, Any] | None = None,
    zarr_format: Literal["ome_zarr", "zarr"] = "ome_zarr",
    compute: bool = True,
    storage_options: JSONDict | None = None,
) -> list[Delayed]:
    """Write data to a Zarr group with optional metadata and scaling.

    Writes data to a specified Zarr group, with options for including metadata, and choosing between OME-Zarr
    and standard Zarr formats.

    :param grp: The Zarr group to write the data to.
    :param data: The data to write. If a Dask array is provided, it will be rechunked
        if necessary.
    :param image_attrs: Attributes of the image DataArray. These will be stored as
        metadata.
    :param coords: Coordinates of the DataArray.
    :param dims: Dimensions of the DataArray.
    :param metadata: Additional metadata to store.
    :param zarr_format: Format to use for storing the data. Use "zarr" for non-OME
        Zarr compliant data with dimensions other than (t, c, z, y, x). Default is
        "ome_zarr".
    :param compute: If True, compute immediately. Otherwise, return a list of
        dask.delayed. Delayed objects representing the value to be computed by dask.
        Default is True.
    :param storage_options: Options to be passed on to the storage backend.
    :return: Empty list if the compute flag is True, otherwise a list of
        dask.delayed.Delayed objects.

    :example:

        .. code-block:: python

            import numpy as np
            import dask.array as da
            import zarr
            from scallops.zarr_io import write_zarr, open_ome_zarr

            # Create a Zarr group
            root = open_ome_zarr("example.zarr")
            grp = root.create_group("test_group")

            # Write data to the Zarr group
            write_zarr(
                grp,
                np.random.rand(100, 100),
                image_attrs={"description": "Test data"},
                coords=None,
                dims=["y", "x"],
            )
    """

    if isinstance(data, xr.DataArray):
        data = data.data
    if isinstance(data, da.Array):
        data = rechunk(data)
    if image_attrs is None:
        image_attrs = {}

    if metadata is not None:
        image_attrs.update(metadata)
    image_attrs, axes, scale_dict = _attrs_axes_scales(
        image_attrs, coords, dims, data.dtype
    )

    dask_delayed = []
    fmt = _current_format()
    if zarr_format == "zarr":  # No axis validation
        if isinstance(data, da.Array):
            d = da.to_zarr(
                arr=data,
                url=grp.store,
                component=str(Path(grp.path, "s0")),
                compute=compute,
                **_da_to_zarr_kwargs(fmt, dims),
            )
            if not compute:
                dask_delayed.append(d)
        elif not isinstance(data, zarr.Array):
            create_kwds = _create_array_kwargs(fmt, dims)

            if (
                storage_options is not None
                and storage_options.get("chunks") is not None
            ):
                create_kwds["chunks"] = storage_options["chunks"]
            grp.create_array("s0", data=data, overwrite=True, **create_kwds)

        zarr_attrs = _create_zarr_attrs(fmt, grp, dims, image_attrs, axes, scale_dict)

        if len(dask_delayed) > 0:

            @dask.delayed
            def _write_metadata_delayed(grp, d):
                _add_metadata(grp, d, fmt)

            return dask_delayed + [
                bind(_write_metadata_delayed, dask_delayed)(grp, zarr_attrs)
            ]
        else:
            _add_metadata(grp, zarr_attrs, fmt)
            return dask_delayed
    else:
        image_attrs = {"metadata": image_attrs}
        # ome-zarr>=0.21 takes the axis order from scale.keys(), so every axis
        # must be present and in order, and units must be passed via axes_units
        axes_units = None
        if axes is not None:
            scale_dict = {
                a["name"]: (scale_dict or {}).get(a["name"], 1.0) for a in axes
            }
            axes_units = {a["name"]: a["unit"] for a in axes if "unit" in a} or None
        return write_multiscale(
            pyramid=[data],
            group=grp,
            fmt=fmt,
            storage_options=[storage_options] if storage_options is not None else None,
            scale_factors=[1.0],
            axes=axes,
            axes_units=axes_units,
            compute=compute,
            scale=scale_dict,
            **image_attrs,
        )


def rechunk(image: xr.DataArray | da.Array) -> xr.DataArray | da.Array:
    """Rechunk a Dask array or DataArray.

    This function checks if the provided Dask array or DataArray has irregular chunks
    and rechunks it to "auto" if necessary.

    :param image: The input Dask array or DataArray to be rechunked.
    :return: The rechunked Dask array or DataArray.

    :example:

        .. code-block:: python

            import dask.array as da
            import xarray as xr
            from scallops.zarr_io import rechunk

            # Create a Dask array with irregular chunks
            data = da.random.random((100, 100), chunks=(50, 30, 20))
            array = xr.DataArray(data, dims=["x", "y"])

            # Rechunk the DataArray
            rechunked_array = rechunk(array)
            print(rechunked_array.chunks)
            # Output: ((50, 50), (50, 50))
    """
    data = image.data if isinstance(image, xr.DataArray) else image
    if not da.core._check_regular_chunks(data.chunks):
        if isinstance(image, xr.DataArray):
            image.data = data.rechunk("auto")
        else:
            image = data.rechunk("auto")

    return image


def _read_zarr_attrs(attrs) -> tuple[dict | None, dict, list[str] | None]:
    """Read attributes from Zarr.

    This function reads and processes the attributes, coordinates, and dimensions from
    the first multiscale dataset in a Zarr group. It also handles physical pixel sizes
    and units if available.

    :param attrs: Zarr attributes.
    :return: A tuple containing:
        - coords: Dictionary of coordinates.
        - attrs: Dictionary of attributes.
        - dims: List of dimension names.
    """
    # v3
    # ome/omero for channel metadata
    # ome/multiscales[0]/metadata for other metadata

    # v2:
    # omero for channel metadata
    # multiscales[0]/metadata for other metadata

    attrs = _ome_attrs(attrs)
    multiscale0 = _multiscale0(attrs)
    if multiscale0 is None or "axes" not in multiscale0:
        # not an ome-zarr
        return None, attrs, None
    axes = multiscale0["axes"]
    dims = [axis["name"] for axis in axes]
    metadata = multiscale0.get("metadata")
    if metadata is None:
        metadata = {}
    coords = {d: metadata[d] for d in dims if d in metadata and d != "c"}
    if "c" in dims and "omero" in attrs:
        channel_names = attrs["omero"].get("channels")
        if channel_names is not None:
            coords["c"] = [c["label"] for c in channel_names]
    names, scales, units = _axis_scales(multiscale0, ("z", "y", "x"))
    if len(names) > 0:
        metadata["physical_pixel_sizes"] = tuple(scales)
        metadata["physical_pixel_units"] = tuple(units)
    return coords, metadata, dims


def _read_ome_zarr_array(
    node: str | Path | zarr.Group,
) -> tuple[zarr.Array, list[str] | None, dict | None, dict] | None:
    """Read an image or label at the given zarr node.

    :param node: The zarr group or path to a zarr group
    :return: zarr array, dims, coords, attrs or None
    """

    if isinstance(node, (str, Path)):
        _node = node  # for error message
        node = zarr.open(node, mode="r")
        if node is None:
            raise ValueError(f"{_node} not found")
    node_attrs = _ome_attrs(node.attrs.asdict())
    if _has_multiscales(node_attrs):
        coords, attrs, dims = _read_zarr_attrs(node_attrs)
        array = node[_dataset_path(node_attrs)]
        return array, dims, coords, attrs
    elif (
        "images" in node.keys()
    ):  # see if user passed test.zarr and zarr file only has one image
        images = node["images"]
        image_keys = list(images.keys())
        if len(image_keys) == 1:
            return _read_ome_zarr_array(images[image_keys[0]])

    else:  # is non-ome zarr
        logger.warning(f"multiscales not found in attrs for {node} ")
    return None


def read_ome_zarr_array(
    node: str | Path | zarr.Group, dask: bool = False
) -> xr.DataArray:
    """Read an image or label at the given zarr node.

    :param node: The zarr group or path to a zarr group
    :param dask: Whether data in DataArray is dask array
    :return: A DataArray
    """
    result = _read_ome_zarr_array(node)

    if result is None:
        raise ValueError(f"No OME-Zarr image found at {node}")
    array, dims, coords, attrs = result
    if not dask:
        array = array[...]
    else:
        array = from_zarr(array)
    return xr.DataArray(array, dims=dims, coords=coords, attrs=attrs)


def open_ome_zarr(url: Path | str, mode: str = "a") -> zarr.Group | None:
    """Open an OME-Zarr store.

    This function parses the given URL and opens the corresponding OME-Zarr store in
    the specified mode.

    :param url: The URL or path to the OME-Zarr store.
    :param mode: The mode to open the store in. Default is "a"
        (read/write, create if not exists).
    :return: The opened Zarr group, or None if the store could not be opened.

    :example:

        .. code-block:: python

            from scallops.zarr_io import open_ome_zarr

            # Open OME-Zarr store in read/write mode
            zarr_group = open_ome_zarr("example.zarr", mode="a")
    """

    try:
        fmt = _current_format()
        loc = parse_url(url, mode=mode, fmt=fmt)
        if loc is None:
            return None
        if mode != "a":
            return zarr.open(loc.store, mode=mode, zarr_format=fmt.zarr_format)
        # mode="a" is also check-then-create: another process can create the root
        # group between our read and our create, no matter how many processes are
        # racing. Fall back to "r+" (must exist, no create attempt), which can't race.
        return _tolerate_concurrent_create(
            lambda: zarr.open(loc.store, mode=mode, zarr_format=fmt.zarr_format),
            lambda: zarr.open(loc.store, mode="r+", zarr_format=fmt.zarr_format),
        )
    except Exception as e:
        logger.error(f"Failed to open OME-Zarr store: {url}")
        raise e


def _read_zarr_experiment(
    store: StoreLike, dask: bool = False, subset: Callable | None = None
):
    """Read an experiment saved using Experiment.save.

    This function reads an experiment from a Zarr store, loading images and labels as
    lazy-loaded data. It allows for optional filtering of the data to be loaded using
    a subset function.

    :param store: The Zarr store or path to the Zarr store containing the experiment
        data.
    :param dask: Whether to load the data as Dask arrays. Default is False.
    :param subset: A function to filter the data to be loaded. It should take a key
        as input and return True if the data should be included. Default is None,
        which includes all data.
    :return: An Experiment object containing the loaded images and labels.

    :example:

        .. code-block:: python

            from scallops.zarr_io import _read_zarr_experiment


            # Define a subset function to include only specific keys
            def subset_function(key):
                return key in ["image1", "label1"]


            experiment = _read_zarr_experiment(
                "example.zarr", dask=True, subset=subset_function
            )
    """
    if subset is None:

        def include_all(*args):
            return True

        subset = include_all

    root = zarr.open(store, mode="r")
    images = {}
    labels = {}

    from scallops.experiment.elements import Experiment

    if "images" in root:
        images = {
            k: _LazyLoadZarrData(v, dask)
            for k, v in root["images"].groups()
            if subset(k)
        }
    if "labels" in root:
        labels = {
            k: _LazyLoadZarrData(v, dask)
            for k, v in root["labels"].groups()
            if subset(k)
        }

    return Experiment(images=images, labels=labels)


class _LazyLoadZarrData(_LazyLoadData):
    """Class for lazy loading Zarr data.

    This class provides a mechanism to lazily load data from a Zarr group.
    The data is only loaded when it is accessed for the first time, which can help to
    optimize memory usage and performance.

    :param group: The Zarr group from which to load the data.
    :param dask: Whether to load the data as a Dask array. Default is False.

    :example:

        .. code-block:: python

            import zarr
            from scallops.zarr_io import _LazyLoadZarrData, open_ome_zarr

            # Create a Zarr group with some data

            root = open_ome_zarr(store=store)
            group = root.create_group("test_group")
            group.create_array("0", data=[1, 2, 3, 4, 5])

            # Create a _LazyLoadZarrData instance
            lazy_data = _LazyLoadZarrData(group, dask=True)

            # Access the data (this will trigger the lazy loading)
            data = lazy_data.data
            print(data)
    """

    def __init__(self, group: zarr.Group, dask: bool = False):
        self._group = group
        self._dask = dask

    @cached_property
    def data(self) -> xr.DataArray:
        """Property to access the data.

        This property loads the data from the Zarr group if it has not been loaded yet.

        :return: xr.DataArray The loaded data as an Xarray DataArray.
        """
        return read_ome_zarr_array(self._group, self._dask)
