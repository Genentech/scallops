import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, types
from fsspec.spec import AbstractFileSystem
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader as ZarrReader

logger = logging.getLogger("scallops")


# Same as https://github.com/bioio-devs/bioio-ome-zarr/blob/main/bioio_ome_zarr/reader.py but fixes bug in channel names
# Also checks to see if zarr path is {zarr_path}/images/image1 with only 1 image
# See https://github.com/bioio-devs/bioio-ome-zarr/pull/22
class ScallopsZarrReader(reader.Reader):
    """The main class of each reader plugin. This class is subclass of the abstract class reader
    (BaseReader) in bioio-base.

    Parameters
    ----------
    image: types.PathLike
        String or Path to the ZARR root
    fs_kwargs: Dict[str, Any]
        Ignored
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[dimensions.Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str

    # Required Methods

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=False,
            fs_kwargs=fs_kwargs,
        )

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                "Could not find a .zgroup or .zarray file at the provided path.",
            )

        self._zarr = get_zarr_reader(self._fs, self._path).zarr
        self._physical_pixel_sizes: Optional[types.PhysicalPixelSizes] = None
        self._channel_names: Optional[List[str]] = None

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            get_zarr_reader(fs, path)
            return True
        except AttributeError:
            return False

    @classmethod
    def is_supported_image(
        cls,
        image: types.ImageLike,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> bool:
        if isinstance(image, (str, Path)):
            return cls._is_supported_image(None, str(image), **kwargs)
        else:
            return reader.Reader.is_supported_image(
                cls, image, fs_kwargs=fs_kwargs, **kwargs
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            scenes = self._zarr.root_attrs["multiscales"]

            # if (each scene has a name) and (that name is unique) use name.
            # otherwise generate scene names.
            if all("name" in scene for scene in scenes) and (
                len({scene["name"] for scene in scenes}) == len(scenes)
            ):
                self._scenes = tuple(str(scene["name"]) for scene in scenes)
            else:
                self._scenes = tuple(
                    f"scene_{i}"
                    for i in range(len(self._zarr.root_attrs["multiscales"]))
                )
        return self._scenes

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        resolution_levels: Tuple[str, ...]
            Return the available resolution levels for the current scene.
            By default these are ordered from highest resolution to lowest
            resolution.
        """
        return tuple(
            rl
            for rl in range(
                len(
                    self._zarr.root_attrs["multiscales"][self.current_scene_index][
                        "datasets"
                    ]
                )
            )
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:
        data_path = self._zarr.root_attrs["multiscales"][self.current_scene_index][
            "datasets"
        ][self.current_resolution_level]["path"]
        image_data = self._zarr.load(data_path)

        axes = self._zarr.root_attrs["multiscales"][self.current_scene_index].get(
            "axes"
        )
        if axes:
            dims = [sub["name"].upper() for sub in axes]
        else:
            dims = list(reader.Reader._guess_dim_order(image_data.shape))

        if not delayed:
            image_data = image_data.compute()

        coords = self._get_coords(
            dims,
            image_data.shape,
            scene=self.current_scene,
            channel_names=self.channel_names,
        )

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={constants.METADATA_UNPROCESSED: self._zarr.root_attrs},
        )

    # Optional Methods
    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """Return the physical pixel sizes of the image."""
        if self._physical_pixel_sizes is None:
            try:
                z_size, y_size, x_size = self._get_pixel_size(
                    list(self.dims.order),
                )
            except Exception as e:
                logger.warning(f"Could not parse zarr pixel size: {e}")
                z_size, y_size, x_size = None, None, None

            self._physical_pixel_sizes = types.PhysicalPixelSizes(
                z_size, y_size, x_size
            )
        return self._physical_pixel_sizes

    def _get_pixel_size(
        self,
        dims: List[str],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # OmeZarr file may contain an additional set of "coordinateTransformations"
        # these coefficents are applied to all resolution levels.
        if (
            "coordinateTransformations"
            in self._zarr.root_attrs["multiscales"][self.current_scene_index]
        ):
            universal_res_consts = self._zarr.root_attrs["multiscales"][
                self.current_scene_index
            ]["coordinateTransformations"][0]["scale"]
        else:
            universal_res_consts = [1.0 for _ in range(len(dims))]

        coord_transform = self._zarr.root_attrs["multiscales"][
            self.current_scene_index
        ]["datasets"][self.current_resolution_level]["coordinateTransformations"]

        spatial_coeffs = {}

        for dim in [
            dimensions.DimensionNames.SpatialX,
            dimensions.DimensionNames.SpatialY,
            dimensions.DimensionNames.SpatialZ,
        ]:
            if dim in dims:
                dim_index = dims.index(dim)
                spatial_coeffs[dim] = (
                    coord_transform[0]["scale"][dim_index]
                    * universal_res_consts[dim_index]
                )
            else:
                spatial_coeffs[dim] = None

        return (
            spatial_coeffs[dimensions.DimensionNames.SpatialZ],
            spatial_coeffs[dimensions.DimensionNames.SpatialY],
            spatial_coeffs[dimensions.DimensionNames.SpatialX],
        )

    @property
    def channel_names(self) -> Optional[List[str]]:
        if self._channel_names is None:
            if "omero" in self._zarr.root_attrs:
                self._channel_names = [
                    str(channel["label"])
                    for channel in self._zarr.root_attrs["omero"]["channels"]
                ]
        return self._channel_names

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene: str,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        coords: Dict[str, Any] = {}

        # Use dims for coord determination
        if dimensions.DimensionNames.Channel in dims:
            # Generate channel names if no existing channel names
            if channel_names is None:
                coords[dimensions.DimensionNames.Channel] = [
                    f"channel_{i}"
                    for i in range(shape[dims.index(dimensions.DimensionNames.Channel)])
                ]
            else:
                coords[dimensions.DimensionNames.Channel] = channel_names

        return coords


def get_zarr_reader(fs: AbstractFileSystem, path: str) -> ZarrReader:
    if fs is not None:
        path = fs.unstrip_protocol(path)

    return ZarrReader(parse_url(path, mode="r"))
