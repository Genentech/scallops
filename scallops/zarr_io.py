"""SCALLOPS Zarr I/O Utility Functions Module.

This module provides a collection of utility functions for reading and writing data in
Zarr format, specifically tailored for OME-Zarr. These functions facilitate the
handling of image and label data, including metadata management and downsampling.

Authors:
- The SCALLOPS development team
"""

import logging
from collections.abc import Callable, Hashable
from functools import lru_cache
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
from ome_zarr.axes import KNOWN_AXES
from ome_zarr.format import CurrentFormat, FormatV04
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import write_image
from packaging.version import Version
from xarray.core.coordinates import DataArrayCoordinates
from zarr.storage import StoreLike

from scallops.experiment.abc import _LazyLoadData
from scallops.utils import _fix_json

logger = logging.getLogger("scallops")


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
    return node is not None and ("ome" in node.attrs or "multiscales" in node.attrs)


def _get_fs(group: zarr.Group):
    if hasattr(group.store, "fs"):
        return group.store.fs
    return fsspec.url_to_fs(_get_store_path(group))[0]


def _get_store_path(group: zarr.Group):
    if hasattr(group.store, "root"):
        return str(group.store.root)
    if hasattr(group.store, "path"):
        return group.store.path
    return ""


def _get_sep(group: zarr.Group) -> str:
    if hasattr(group.store, "fs"):
        return group.store.fs.sep
    return "/"


def _create_omero_metadata(
    coords: DataArrayCoordinates, dims: tuple[Hashable, ...]
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
            omero_metadata = _create_omero_metadata(array.coords, array.dims)
            print(omero_metadata)
            # Output: {'channels': [{'label': 'DAPI', 'color': '00FFFF'},
                {'label': 'FITC', 'color': 'FFFF00'}, ...]}
    """
    channel_names = coords["c"] if "c" in dims and "c" in coords else None
    if channel_names is not None:
        if isinstance(channel_names, xr.DataArray):
            channel_names = channel_names.values
        # Napari does not like '#' at start of hex color string
        # Hex colors match Napari defaults
        colors = ["00FFFF", "FFFF00", "FF00FF", "FF0000", "008000", "0000FF"]
        # Napari requires that colors are specified if channel names are specified
        channels = (
            [
                dict(label=channel_names[i], color=colors[i % len(colors)])
                for i in range(len(channel_names))
            ]
            if not np.isscalar(channel_names)
            else [channel_names.tolist()]
        )
        return dict(channels=channels)
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


def _attrs_axes_coordinates(
    image_attrs: dict, coords: DataArrayCoordinates, dims: tuple[Hashable, ...]
) -> tuple[dict, list[dict], list[dict] | None]:
    """Prepare attributes, axes, and coordinate transformations for Zarr storage.

    Processes the attributes, coordinates, and dimensions of a DataArray to generate
    metadata suitable for Zarr storage. It includes handling OMERO metadata, physical
    pixel sizes, and coordinate transformations.

    :param image_attrs: The attributes of the image DataArray.
    :param coords: The coordinates of the DataArray.
    :param dims: The dimensions of the DataArray.
    :return: A tuple containing:
        - Updated image attributes dictionary.
        - List of axes dictionaries.
        - List of coordinate transformations dictionaries or None.

    :example:

        .. code-block:: python

            import xarray as xr
            import numpy as np
            from scallops.zarr_io import _attrs_axes_coordinates

            data = np.random.rand(5, 10, 512, 512)
            dims = ("c", "z", "y", "x")
            coords = {"c": ["DAPI", "FITC", "TRITC", "Cy5", "Cy7"]}
            array = xr.DataArray(data, dims=dims, coords=coords)
            image_attrs = {
                "physical_pixel_sizes": [0.1, 0.1, 0.5],
                "physical_pixel_units": ["um", "um", "um"],
            }

            # Prepare attributes, axes, and coordinate transformations
            updated_attrs, axes, coord_transformations = _attrs_axes_coordinates(
                image_attrs, array.coords, array.dims
            )
            print(updated_attrs)
            print(axes)
            print(coord_transformations)
    """
    image_attrs = _fix_json(image_attrs)
    omero = _create_omero_metadata(coords, dims)
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

    coordinate_transformations = None
    physical_pixel_units = None
    if "physical_pixel_sizes" in image_attrs and "physical_pixel_units" in image_attrs:
        physical_pixel_sizes = image_attrs.pop("physical_pixel_sizes")
        physical_pixel_units = image_attrs.pop("physical_pixel_units")
        non_space_dims = [d for d in dims if d not in ("z", "y", "x")]
        scale = list((1.0,) * len(non_space_dims)) + list(physical_pixel_sizes)
        coordinate_transformations = [{"scale": scale, "type": "scale"}]

    axes = []
    space_index = 0
    for d in dims:
        axis = {"name": d, "type": KNOWN_AXES.get(d)}
        if physical_pixel_units is not None and axis["type"] == "space":
            axis["unit"] = physical_pixel_units[space_index]
            space_index = space_index + 1
        axes.append(axis)

    return image_attrs, axes, coordinate_transformations


def _write_zarr_image(
    name: str | None,
    root: zarr.Group | str | Path,
    image: da.Array | np.ndarray | xr.DataArray,
    scaler: Scaler = None,
    metadata: None | dict[str, Any] = None,
    group: str | None = "images",
    zarr_format: Literal["ome_zarr", "zarr"] = "ome_zarr",
    compute: bool = True,
) -> list[Delayed]:
    """Write image in zarr format.

    :param name: Zarr group name to store image
    :param root: Root zarr group.
    :param image: Image to write. A downsampling of the image will be computed if the
        scaler argument is non-None. Image.attrs will be stored if image is an instance
        of xr.DataArray
    :param scaler: Scaler implementation for downsampling the image argument. If None,
        no downsampling is performed.
    :param group: Group name under root to write image to
    :param metadata: Additional metadata to store
    :param zarr_format: Either ome_zarr or zarr. Use zarr for storing non-ome zarr
        compliant images with dimensions other than (t,c,z,y,x)
    :param compute: If true compute immediately otherwise a list
        of :class:`dask.delayed.Delayed` is returned.
    :return: Empty list if the compute flag is True, otherwise it returns a list
        of :class:`dask.delayed.Delayed` representing the value to be computed by dask.
    """
    if zarr_format == "zarr" and scaler is not None:
        raise NotImplementedError("Scaler not implemented for zarr format")

    if isinstance(root, (str, Path)):
        root = open_ome_zarr(root, mode="a")
    dest_grp = root
    if group is not None and name is not None:
        images_grp = root.require_group(group, overwrite=False)
        dest_grp = images_grp.create_group(name.replace("/", "-"), overwrite=True)
    image_attrs = None
    coords = None
    dims = None
    if isinstance(image, xr.DataArray):
        data = image.data
        image_attrs = image.attrs.copy()
        coords = image.coords
        dims = image.dims
    else:
        data = image
    return write_zarr(
        grp=dest_grp,
        data=data,
        image_attrs=image_attrs,
        coords=coords,
        dims=dims,
        scaler=scaler,
        metadata=metadata,
        zarr_format=zarr_format,
        compute=compute,
    )


@lru_cache
def _zarr_v3() -> bool:
    try:
        import zarr
    except ImportError:
        return False
    else:
        return Version(zarr.__version__).major >= 3


def write_zarr(
    grp: zarr.Group,
    data: np.ndarray | da.Array | xr.DataArray,
    image_attrs: dict | None,
    coords: dict | None,
    dims: list[str] | tuple[Hashable, ...] | None,
    scaler: Scaler = None,
    metadata: dict[str, Any] | None = None,
    zarr_format: Literal["ome_zarr", "zarr"] = "ome_zarr",
    compute: bool = True,
) -> list[Delayed]:
    """Write data to a Zarr group with optional metadata and scaling.

    Writes data to a specified Zarr group, with options for including metadata,
    downsampling using a scaler, and choosing between OME-Zarr and standard Zarr
    formats.

    :param grp: The Zarr group to write the data to.
    :param data: The data to write. If a Dask array is provided, it will be rechunked
        if necessary.
    :param image_attrs: Attributes of the image DataArray. These will be stored as
        metadata.
    :param coords: Coordinates of the DataArray.
    :param dims: Dimensions of the DataArray.
    :param scaler: Scaler implementation for downsampling the data. If None,
        no downsampling is performed.
    :param metadata: Additional metadata to store.
    :param zarr_format: Format to use for storing the data. Use "zarr" for non-OME
        Zarr compliant data with dimensions other than (t, c, z, y, x). Default is
        "ome_zarr".
    :param compute: If True, compute immediately. Otherwise, return a list of
        dask.delayed. Delayed objects representing the value to be computed by dask.
        Default is True.
    :return: Empty list if the compute flag is True, otherwise a list of
        dask.delayed.Delayed objects.

    :raises NotImplementedError:
        If scaler is provided and zarr_format is "zarr".

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
    axes = None
    coordinate_transformations = None
    if image_attrs is not None:
        # Metadata can't be numpy arrays or python classes so do a round trip
        # conversion to convert to JSON serializable
        _fix_attrs(image_attrs)
        if metadata is not None:
            image_attrs.update(metadata)
        image_attrs, axes, coordinate_transformations = _attrs_axes_coordinates(
            image_attrs, coords, dims
        )
    dask_delayed = []
    if zarr_format == "zarr":  # No axis validation
        kwargs = dict()
        zarr_version = 3 if _zarr_v3() else 2
        fmt = CurrentFormat() if zarr_version else FormatV04()
        if zarr_version == 2:
            kwargs["dimension_separator"] = "/"
        else:
            kwargs["chunk_key_encoding"] = fmt.chunk_key_encoding
        if isinstance(data, da.Array):
            d = da.to_zarr(
                arr=data,
                url=grp.store,
                component=str(Path(grp.path, "0")),
                compute=compute,
                **kwargs,
            )
            if not compute:
                dask_delayed.append(d)
        elif not isinstance(data, zarr.Array):
            if zarr_version == 2:
                grp.create_dataset("0", data=data, overwrite=True, **kwargs)
            else:
                grp.create_array("0", data=data, overwrite=True, **kwargs)
        # v3
        # ome/omero for channel metadata
        # ome/multiscales[0]/metadata for other metadata

        # v2:
        # omero for channel metadata
        # multiscales[0]/metadata for other metadata
        datasets = [{"path": "0"}]
        if coordinate_transformations is not None:
            datasets[0]["coordinateTransformations"] = coordinate_transformations
        multiscales = [dict(version=fmt.version, datasets=datasets, name=grp.name)]
        zarr_attrs = (
            {"multiscales": multiscales}
            if zarr_version == 2
            else {"ome": {"multiscales": multiscales}}
        )

        if axes is not None:
            multiscales[0]["axes"] = axes
        if image_attrs is not None:
            if "omero" in image_attrs:
                if zarr_version == 2:
                    omero = zarr_attrs.get("omero", {})
                    omero.update(image_attrs.pop("omero"))
                    zarr_attrs["omero"] = omero
                else:
                    omero = zarr_attrs["ome"].get("omero", {})
                    omero.update(image_attrs.pop("omero"))
                    zarr_attrs["ome"]["omero"] = omero

            multiscales[0]["metadata"] = image_attrs
        if len(dask_delayed) > 0:

            @dask.delayed
            def _write_metadata_delayed(grp, d):
                grp.attrs.update(d)

            return dask_delayed + [
                bind(_write_metadata_delayed, dask_delayed)(grp, zarr_attrs)
            ]
        else:
            grp.attrs.update(zarr_attrs)
            return dask_delayed
    else:
        return write_image(
            image=data,
            group=grp,
            scaler=scaler,
            axes=axes,
            compute=compute,
            metadata=image_attrs if image_attrs is not None else {},
            coordinate_transformations=(
                [coordinate_transformations]
                if coordinate_transformations is not None
                else None
            ),
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


def _write_zarr_labels(
    name: str,
    root: zarr.Group | str | Path,
    labels: np.ndarray | xr.DataArray | da.Array,
    metadata: dict[str, Any] = None,
    group_metadata: dict[str, Any] = None,
    scaler: Scaler = None,
    compute: bool = True,
    storage_options: JSONDict | None = None,
) -> list[Delayed]:
    """Write label in zarr format.

    :param name: Zarr group name to store label
    :param root: Root zarr group.
    :param labels: Labels to write. A downsampling of the label will be computed if
        the scaler argument is non-None.
    :param metadata: Optional label metadata.
    :param group_metadata: Optional group level  metadata.
    :param scaler: Scaler implementation for downsampling the label argument.
        If None, no downsampling will be performed.
    :param compute: If true compute immediately otherwise a list
        of :class:`dask.delayed.Delayed` is returned.
    :param storage_options: Optional storage options.
    :return: Empty list if the compute flag is True, otherwise it returns a list
        of :class:`dask.delayed.Delayed` representing the value to be computed by dask.
    """

    # stored in labels/key
    name = name.replace("/", "-")
    if isinstance(root, (str, Path)):
        root = open_ome_zarr(root, mode="a")
    labels_grp = root.require_group("labels")
    grp = labels_grp.create_group(name, overwrite=True)
    if not isinstance(labels, xr.DataArray):
        if labels.ndim == 2:
            label_axes = ["y", "x"]
        elif labels.ndim == 5:
            label_axes = ["t", "c", "z", "y", "x"]
        else:
            raise ValueError("Axes can't be inferred for 3D or 4D labels")
    else:
        label_axes = labels.dims
        labels = labels.data

    # need 'image-label' attr to be recognized as label
    group_metadata = group_metadata.copy() if group_metadata is not None else dict()
    if "image-label" not in group_metadata:
        group_metadata["image-label"] = {}
    grp.attrs.update(group_metadata)
    metadata = metadata.copy() if metadata is not None else {}
    if isinstance(labels, da.Array) or (
        isinstance(labels, xr.DataArray) and isinstance(labels.data, da.Array)
    ):
        labels = rechunk(labels)
    return write_image(
        labels,
        grp,
        scaler=scaler,
        axes=label_axes,
        metadata=metadata,
        compute=compute,
        coordinate_transformations=None,
        storage_options=storage_options,
    )


def _read_zarr_attrs(attrs) -> tuple[dict, dict, list[str]]:
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

    if "ome" in attrs:
        attrs = attrs["ome"]
    multiscales = attrs["multiscales"]
    if len(multiscales) > 0:
        multiscale0 = multiscales[0]
    else:
        return None, None, None

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
    space_indices = [dims.index(d) for d in ["z", "y", "x"] if d in dims]

    units = []
    space_indices_with_units = []
    for d in space_indices:
        unit = multiscale0["axes"][d].get("unit", None)
        if unit not in [None, ""]:
            space_indices_with_units.append(d)
            units.append(unit)

    if len(space_indices_with_units) > 0:
        scale = multiscale0["datasets"][0]["coordinateTransformations"][0]["scale"]
        physical_pixel_sizes = tuple([scale[d] for d in space_indices_with_units])
        metadata["physical_pixel_sizes"] = physical_pixel_sizes
        metadata["physical_pixel_units"] = tuple(units)
    return coords, metadata, dims


def _read_ome_zarr_array(
    node: str | Path | zarr.Group,
) -> tuple[zarr.Array, dict, list[str], dict, dict] | None:
    """Read an image or label at the given zarr node.

    :param node: The zarr group or path to a zarr group
    :return: zarr array, dims, coords, attrs or None
    """

    if isinstance(node, (str, Path)):
        _node = node  # for error message
        node = zarr.open(node, mode="r")
        if node is None:
            raise ValueError(f"{_node} not found")
    # For zarr v3, everything is under the "ome" namespace
    if "ome" in node.attrs or "multiscales" in node.attrs:
        dims = None
        coords = {}
        attrs = {}
        coords, attrs, dims = _read_zarr_attrs(node.attrs)

        array = node["0"]
        return array, dims, coords, attrs
    else:  # see if user passed test.zarr and zarr file only has one image
        if "images" in node.keys():
            images = node["images"]
            image_keys = list(images.keys())
            if len(image_keys) == 1:
                return _read_ome_zarr_array(images[image_keys[0]])
        logger.warning(f"multiscales not found in attrs for {node} ")


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
        return None
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
        loc = parse_url(url, mode=mode)
        if loc is None:
            return None
        return zarr.open(loc.store, mode=mode)
    except Exception as e:
        logger.error(f"Failed to open OME-Zarr store: {url}")
        raise e


def _read_zarr_experiment(
    store: StoreLike, dask: bool = False, subset: Callable = None
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
            group.create_dataset("0", data=[1, 2, 3, 4, 5])

            # Create a _LazyLoadZarrData instance
            lazy_data = _LazyLoadZarrData(group, dask=True)

            # Access the data (this will trigger the lazy loading)
            data = lazy_data.data
            print(data)
    """

    def __init__(self, group: zarr.Group, dask: bool = False):
        super().__init__()
        self._data = None  # Lazy load
        self._group = group
        self._dask = dask

    @property
    def data(self) -> xr.DataArray:
        """Property to access the data.

        This property loads the data from the Zarr group if it has not been loaded yet.

        :return: xr.DataArray The loaded data as an Xarray DataArray.
        """
        if self._data is None:
            self._data = read_ome_zarr_array(self._group, self._dask)
        return self._data
