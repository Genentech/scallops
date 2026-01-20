"""This module provides functionality for stitching and diagnosing stitching."""

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from typing import Literal

import bioio
import dask
import dask.bag as db
import fsspec
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import zarr
from dask import delayed
from ome_types import from_xml
from pint import UndefinedUnitError, UnitRegistry
from skimage.util import img_as_float, img_as_ubyte, img_as_uint

from scallops.cli.util import _group_src_attrs
from scallops.features.image_quality import power_spectrum
from scallops.io import (
    _create_image,
    _get_fs_protocol,
    _images2fov,
    _localize_path,
    read_image,
)
from scallops.stitch._radial import radial_correct
from scallops.stitch.shift_utils import _zncc

ureg = UnitRegistry()
logger = logging.getLogger("scallops")


def _read_tile(
    file_list: list[str] | zarr.Group,
    attrs: dict[str, str | list[str]],
    channel: int,
    radial_correction_k: float | None = None,
    crop_width: int | None = None,
    z_index: int | Literal["max"] = "max",
    scene_id: None | str | int = None,
) -> np.ndarray:
    img = _images2fov(file_list, attrs, dask=False, scene_id=scene_id)
    img = img.isel(t=0, c=channel, missing_dims="ignore")
    if img.ndim > 2:
        img = img.max(dim="z") if not isinstance(z_index, int) else img.isel(z=z_index)
    img = img.values

    if radial_correction_k is not None:
        target_dtype = img.dtype
        img = radial_correct(img, radial_correction_k)
        img = dtype_convert(img, target_dtype)
    if crop_width is not None and crop_width > 0:
        img = img[..., crop_width:-crop_width, crop_width:-crop_width]
    return img


_read_tile_delayed = delayed(_read_tile)


def dtype_convert(img, dtype):
    if dtype == np.uint16:
        return img_as_uint(img)
    elif dtype == np.uint8:
        return img_as_ubyte(img)
    raise ValueError(f"Unsupported type: {dtype}")


def overlapping_pairs(df: pd.DataFrame, tile_shape: tuple[int, int]) -> pd.DataFrame:
    """Compute pairwise overlapping regions from stitching coordinates.

    :param pd.DataFrame df: DataFrame containing tile locations
    :param tile_shape: Tile shape
    :return: pd.DataFrame containing overlapping regions
    """

    y = df["y"].round().values.astype(int)
    x = df["x"].round().values.astype(int)
    ysize = tile_shape[0]
    xsize = tile_shape[1]
    tiles = df["tile"].values
    sources = df["source"].values
    boxes = []
    for i in range(len(x)):
        b = shapely.box(x[i], y[i], x[i] + xsize - 1, y[i] + ysize - 1)
        boxes.append(b)
    tree = shapely.STRtree(boxes)
    results = []

    for i in range(len(boxes)):
        b = boxes[i]
        indices_intersection = tree.query(b, predicate="intersects")

        for index in indices_intersection:
            if index > i:
                overlap = shapely.intersection(b, boxes[index])
                overlap_bounds = overlap.bounds

                y_start = int(overlap_bounds[1])
                y_end = int(overlap_bounds[3])
                x_start = int(overlap_bounds[0])
                x_end = int(overlap_bounds[2])

                s1_y_start = y_start - y[i]
                s1_y_end = y_end - y[i]
                s1_x_start = x_start - x[i]
                s1_x_end = x_end - x[i]

                s2_y_start = y_start - y[index]
                s2_y_end = y_end - y[index]
                s2_x_start = x_start - x[index]
                s2_x_end = x_end - x[index]

                results.append(
                    [
                        sources[i],
                        sources[index],
                        tiles[i],
                        tiles[index],
                        overlap.area,
                        s1_y_start,
                        s1_y_end,
                        s1_x_start,
                        s1_x_end,
                        s2_y_start,
                        s2_y_end,
                        s2_x_start,
                        s2_x_end,
                        y_start,
                        y_end,
                        x_start,
                        x_end,
                    ]
                )

    return pd.DataFrame(
        results,
        columns=[
            "Source1",
            "Source2",
            "Tile1",
            "Tile2",
            "Area",
            "Tile1-Start-Y",
            "Tile1-End-Y",
            "Tile1-Start-X",
            "Tile1-End-X",
            "Tile2-Start-Y",
            "Tile2-End-Y",
            "Tile2-Start-X",
            "Tile2-End-X",
            "Start-Y",
            "End-Y",
            "Start-X",
            "End-X",
        ],
    )


def tile_source_labels(
    df: pd.DataFrame, tile_shape: tuple[int, int], offset: int = 1
) -> np.ndarray[np.uint16]:
    """Create label array indicating tile source

    :param df: DataFrame containing stitched tile locations
    :param tile_shape: Image tile size
    :param offset: Value to add to tile
    :return: Labels array
    """
    df = df.sort_values("distance_to_center", ascending=False)
    y = df["y"].round().values.astype(int)
    x = df["x"].round().values.astype(int)
    stitched_y_size = (y + tile_shape[0]).max()
    stitched_x_size = (x + tile_shape[1]).max()
    result = np.zeros((stitched_y_size, stitched_x_size), dtype=np.uint16)
    y_end = y + tile_shape[0]
    x_end = x + tile_shape[1]
    tiles = df["tile"].values
    for i in range(len(y)):
        x1, y1, x2, y2 = x[i], y[i], x_end[i], y_end[i]
        result[y1:y2, x1:x2] = tiles[i] + offset
    return result


def tile_overlap_mask(
    df: pd.DataFrame, tile_shape: tuple[int, int], fill: bool = True
) -> np.ndarray[np.uint8]:
    """Create a binary mask where zeros indicate locations where tiles overlap.

    :param df: DataFrame containing stitched tile locations
    :param fill: Whether to fill mask interior. Typically set to True when blending,
        False otherwise.
    :param tile_shape: Image tile shape for writing
    :return: Mask array
    """

    df = df.copy()
    df["x"] = df["x"].round().values.astype(int)
    df["y"] = df["y"].round().values.astype(int)
    fused_y_size = (df["y"] + tile_shape[0]).max()
    fused_x_size = (df["x"] + tile_shape[1]).max()
    mask = np.ones((fused_y_size, fused_x_size), dtype=np.uint8)

    df = df.sort_values("distance_to_center", ascending=False)
    y = df["y"].values
    x = df["x"].values
    if fill:
        pairs_df = overlapping_pairs(df, tile_shape)
        y_start = pairs_df["Start-Y"].values
        y_end = pairs_df["End-Y"].values
        x_start = pairs_df["Start-X"].values
        x_end = pairs_df["End-X"].values

        for i in range(len(y_start)):
            y1 = y_start[i]
            y2 = y_end[i]
            x1 = x_start[i]
            x2 = x_end[i]

            mask[y1 : y2 + 1, x1 : x2 + 1] = 0
    else:
        y_end = y + tile_shape[0] - 1
        x_end = x + tile_shape[1] - 1
        for i in range(len(y)):
            x1, y1, x2, y2 = x[i], y[i], x_end[i], y_end[i]
            outer_border = mask[y1 : y2 + 1, x1 : x2 + 1].max() > 0
            mask[y1 : y2 + 1, x1 : x2 + 1] = 1  # clear rectangle
            mask[y1, x1 : x2 + 1] = 0  # top
            mask[y2, x1 : x2 + 1] = 0  # bottom
            mask[y1 : y2 + 1, x1] = 0  # left
            mask[y1 : y2 + 1, x2] = 0  # right
            if outer_border:
                mask[max(0, y1 - 1), x1 : x2 + 1] = 0  # top
                mask[min(mask.shape[0] - 1, y2 + 1), x1 : x2 + 1] = 0  # bottom
                mask[y1 : y2 + 1, max(0, x1 - 1)] = 0  # left
                mask[y1 : y2 + 1, min(mask.shape[1] - 1, x2 + 1)] = 0  # right
    return mask


def read_stage_positions(filepaths: Sequence[str], stage_positions_path: str):
    if stage_positions_path.endswith(".json"):
        stage_positions = _stage_positions_from_araceli_json(
            filepaths, stage_positions_path
        )
    else:
        stage_positions = pd.read_csv(stage_positions_path, index_col="name")
        for c in ["y", "x"]:
            if c not in stage_positions.columns:
                raise ValueError(
                    f"Expected columns `name`, `y`, and `x` in {stage_positions_path}"
                )

        stage_positions = stage_positions.loc[filepaths][["y", "x"]].values
    return stage_positions


def _stage_positions_from_araceli_json(filepaths: Sequence[str], json_path: str):
    fs = fsspec.url_to_fs(json_path)[0]
    with fs.open(json_path, "r") as f:
        d = json.load(f)

    # BP65_s1_w1_z-_20250305T000146Z_7fd6580c-3fae-4271-8330-ae7c121d7a96.tiff

    stage_positions = np.zeros((len(filepaths), 2))
    wells = d["Wells"]
    # convert milimeters to micrometers
    factor = ureg.parse_expression("millimeter").to("micrometer").magnitude
    for i in range(len(filepaths)):
        tokens = os.path.basename(filepaths[i]).split("_")
        well = wells[tokens[0]]
        s1 = well["Sites"]["s1"]

        stage_positions[i, 0] = s1["YPosition"] * factor
        stage_positions[i, 1] = s1["XPosition"] * factor
    return stage_positions


def _stage_positions_from_image_metadata(filepaths: Sequence[str]) -> np.ndarray:
    # get from image metadata
    if len(filepaths) == 1:
        img = _create_image(filepaths[0])
        ome_metadata = _get_ome(img)
        if ome_metadata is None:
            raise ValueError(f"Could not extract OME metadata from {filepaths[0]}.")
        n_images = len(ome_metadata.images)
        stage_positions = np.zeros((n_images, 2))
        for i in range(n_images):
            y, x = get_tile_position(img, i)
            stage_positions[i, 0] = y
            stage_positions[i, 1] = x
    else:
        stage_positions = np.zeros((len(filepaths), 2))
        for i in range(len(filepaths)):
            img = _create_image(filepaths[i])
            y, x = get_tile_position(img)
            stage_positions[i, 0] = y
            stage_positions[i, 1] = x
    return stage_positions


def _get_ome(image: bioio.BioImage):
    try:
        metadata = image.ome_metadata
        if metadata is not None:
            return metadata
    except NotImplementedError:
        pass

    if isinstance(image.metadata, str):
        try:
            return from_xml(image.metadata)
        except:  # noqa: E722
            pass
    return None


def get_tile_position(image: bioio.BioImage, image_index: int = 0):
    ome_metadata = _get_ome(image)

    if ome_metadata is not None:
        values = [
            ome_metadata.images[image_index].pixels.planes[0].position_y,
            ome_metadata.images[image_index].pixels.planes[0].position_x,
        ]
        physical_size_y_unit = (
            ome_metadata.images[image_index].pixels.planes[0].position_y_unit.value
        )
        physical_size_x_unit = (
            ome_metadata.images[image_index].pixels.planes[0].position_x_unit.value
        )
    elif "multiscales" in image.metadata:
        metadata = image.metadata["multiscales"][0]["metadata"]
        values = [metadata["position_y"], metadata["position_x"]]
        physical_size_y_unit = metadata["position_y_unit"]
        physical_size_x_unit = metadata["position_x_unit"]
    else:
        attrs = image.xarray_dask_data.attrs
        if "unprocessed" in attrs:
            if 51123 in attrs["unprocessed"]:
                attrs = attrs["unprocessed"][51123]
                return np.array([attrs["YPositionUm"], attrs["XPositionUm"]])
            elif 50839 in attrs["unprocessed"]:
                attrs = attrs["unprocessed"][50839]
                if "Info" in attrs:
                    attrs = json.loads(attrs["Info"])
                    return np.array([attrs["YPositionUm"], attrs["XPositionUm"]])
            elif 270 in attrs["unprocessed"]:  # IXM
                attrs = attrs["unprocessed"][270]
                import xml.etree.ElementTree as ET

                try:
                    tree = ET.fromstring(attrs)
                    stage_y = tree.findall(".//prop[@id='stage-position-y']")
                    stage_x = tree.findall(".//prop[@id='stage-position-x']")
                    if len(stage_y) == 1 and len(stage_x) == 1:
                        stage_y = stage_y[0].attrib["value"]
                        stage_x = stage_x[0].attrib["value"]
                        return np.array([stage_y, stage_x])
                except:  # noqa: E722
                    pass
    if physical_size_y_unit is not None and physical_size_x_unit is not None:
        try:
            values[0] = (
                values[0]
                * ureg.parse_expression(physical_size_y_unit).to("micrometer").magnitude
            )
            values[1] = (
                values[1]
                * ureg.parse_expression(physical_size_x_unit).to("micrometer").magnitude
            )
        except UndefinedUnitError:
            logger.info("Unknown stage coordinate size units. Assuming µm")
    else:
        logger.info("Unknown stage coordinate size units. Assuming µm")
    position_microns = np.array(values, dtype=float)
    return position_microns


def _pixel_size_from_araceli_json(json_path: str):
    fs = fsspec.url_to_fs(json_path)[0]
    with fs.open(json_path, "r") as f:
        d = json.load(f)
    camera = d["Cores"]["Core0"]["Camera"]
    return np.array(
        [float(camera["ImagePixelHeightUm"]), float(camera["ImagePixelWidthUm"])]
    )


def get_pixel_size(
    filepaths: Sequence[str], stage_positions_path: str | None
) -> np.ndarray:
    if stage_positions_path is not None and stage_positions_path.lower().endswith(
        ".json"
    ):
        return _pixel_size_from_araceli_json(stage_positions_path)

    return _pixel_size_from_image(_create_image(filepaths[0]))


def _pixel_size_from_image(image: bioio.BioImage) -> np.array:
    ome_metadata = _get_ome(image)
    values = None
    physical_size_y_unit = None
    physical_size_x_unit = None
    if ome_metadata is not None:
        values = [
            ome_metadata.images[0].pixels.physical_size_y,
            ome_metadata.images[0].pixels.physical_size_x,
        ]
        physical_size_y_unit = ome_metadata.images[0].pixels.physical_size_y_unit.value
        physical_size_x_unit = ome_metadata.images[0].pixels.physical_size_x_unit.value
    elif "multiscales" in image.metadata:
        metadata = image.metadata["multiscales"][0]["metadata"]
        values = [metadata["physical_size_y"], metadata["physical_size_x"]]
        physical_size_y_unit = metadata["physical_size_y_unit"]
        physical_size_x_unit = metadata["physical_size_x_unit"]
    else:
        attrs = image.xarray_dask_data.attrs
        if "unprocessed" in attrs:
            attrs = attrs["unprocessed"]
            if 51123 in attrs:
                attrs = attrs[51123]
                if "PixelSizeUm" in attrs:
                    pixel_size = attrs["PixelSizeUm"]
                    values = np.array([pixel_size, pixel_size])
            elif 270 in attrs:
                import xml.etree.ElementTree as ET

                try:
                    tree = ET.fromstring(attrs[270])
                    y = tree.findall(".//prop[@id='spatial-calibration-y']")
                    x = tree.findall(".//prop[@id='spatial-calibration-x']")
                    if len(y) == 1 and len(x) == 1:
                        y = y[0].attrib["value"]
                        x = x[0].attrib["value"]
                        values = np.array([y, x]).astype(float)
                        units = tree.findall(".//prop[@id='spatial-calibration-units']")
                        if len(units) == 1:
                            units = units[0].attrib["value"]
                            physical_size_y_unit = units
                            physical_size_x_unit = units

                except:  # noqa: E722
                    pass
        if values is None and hasattr(image, "physical_pixel_sizes"):
            values = np.array(
                [image.physical_pixel_sizes.Y, image.physical_pixel_sizes.X]
            )
    if physical_size_y_unit is not None and physical_size_x_unit is not None:
        try:
            values[0] = (
                values[0]
                * ureg.parse_expression(physical_size_y_unit).to("micrometer").magnitude
            )
            values[1] = (
                values[1]
                * ureg.parse_expression(physical_size_x_unit).to("micrometer").magnitude
            )
        except UndefinedUnitError:
            logger.info("Unknown physical size units. Assuming µm")
    else:
        logger.info("Unknown physical size units. Assuming µm")

    return np.array(values)


def min_zncc(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Compute z-normalized cross-correlation.

    :param a: First image of dimensions (y,x) or (t,y,x) if second image is None
    :param b: Optional second image
    :return: Max z-normalized cross-correlation.
    """
    if b is not None:
        return _zncc(a, b)
    value = np.inf
    ref = a[0]
    for i in range(1, a.shape[0]):
        value = min(value, _zncc(ref, a[i]))
    return value


def n_labels(a: np.ndarray) -> np.ndarray:
    """Count number of unique labels in an image.

    :param a: Label image
    :return: Number of labels.
    """
    values = np.unique(a)
    values = values[values > 0]
    return np.full((1,) * a.ndim, len(values))


def create_composite(
    df: pd.DataFrame,
    channel: int | None = None,
    radial_correction_k: float | None = None,
    crop_width: int | None = None,
    ffp: np.ndarray | None = None,
    dfp: np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Use stitching coordinates to create a composite image.

    :param df: Stitch output dataframe (filtered to contain tiles of interest).
    :param channel: If int, write channel from tile in a separate channel in result
        image.
    :param radial_correction_k: Image radial correction k
    :param crop_width: Image crop width
    :param ffp: Image FFP
    :param dfp: Image DFP
    :return: Tuple of composite image and tile source (offset by 1 to avoid zeros)
    """
    df = df.sort_values("distance_to_center", ascending=False)
    y = df["y"].round().values.astype(int)
    x = df["x"].round().values.astype(int)
    source = df["source"].values
    tiles = df["tile"].values
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    y -= ymin
    x -= xmin
    ymax -= ymin
    xmax -= xmin
    source_local0 = _localize_path(source[0])
    img0 = read_image(source_local0 if source_local0 is not None else source[0]).isel(
        t=0, z=0, missing_dims="ignore"
    )
    separate_channels = isinstance(channel, (int, np.integer))

    if separate_channels:
        img0 = img0.isel(c=channel, missing_dims="ignore")
    tile_shape_before_crop = img0.shape[-2:]
    nchannels = img0.sizes["c"] if not separate_channels and "c" in img0.sizes else 1

    if crop_width is not None and crop_width <= 0:
        crop_width = None
    if crop_width is not None:
        img0 = img0[..., crop_width:-crop_width, crop_width:-crop_width]
    tile_shape = img0.shape[-2:]
    image_shape = (ymax + tile_shape[0], xmax + tile_shape[1])
    result_shape = (
        (len(df),) + image_shape if separate_channels else (nchannels,) + image_shape
    )
    result_image = np.zeros(result_shape, dtype=img0.dtype)
    tile_locations = []
    tile_source = np.zeros(image_shape, dtype=np.uint16)
    if separate_channels:
        if ffp is not None:
            ffp = ffp[channel]
        if dfp is not None:
            dfp = dfp[channel]
    for i in range(len(df)):
        source_local = _localize_path(source[i]) if i > 0 else source_local0
        img = (
            read_image(source_local if source_local is not None else source[i])
            .squeeze()
            .isel(t=0, z=0, missing_dims="ignore")
        )
        if source_local is not None:
            os.remove(source_local)
        if separate_channels:
            img = img.isel(c=channel, missing_dims="ignore")
        img = img.values
        # order: radial, illumination, crop
        if radial_correction_k is not None:
            img = radial_correct(img, radial_correction_k)
            img = dtype_convert(img, result_image.dtype)
        img = img_as_float(img, force_copy=True)
        if dfp is not None:
            img -= dfp
        if ffp is not None:
            img /= ffp
        img.clip(0, 1, out=img)
        img = dtype_convert(img, result_image.dtype)
        if crop_width is not None:
            img = img[..., crop_width:-crop_width, crop_width:-crop_width]
        tile_locations.append(dict(tile=tiles[i], y=y[i], x=x[i]))
        if separate_channels:
            result_image[
                i, y[i] : y[i] + tile_shape[0], x[i] : x[i] + tile_shape[1]
            ] = img
        else:
            result_image[
                ..., y[i] : y[i] + tile_shape[0], x[i] : x[i] + tile_shape[1]
            ] = img
        tile_source[y[i] : y[i] + tile_shape[0], x[i] : x[i] + tile_shape[1]] = (
            tiles[i] + 1
        )
    return xr.DataArray(
        result_image,
        dims=["c", "y", "x"],
        attrs=dict(
            tile_locations=tile_locations,
            tile_shape=tile_shape,
            tile_shape_before_crop=tile_shape_before_crop,
        ),
    ), xr.DataArray(
        tile_source,
        dims=["y", "x"],
        attrs=dict(
            tile_locations=tile_locations,
            tile_shape=tile_shape,
            tile_shape_before_crop=tile_shape_before_crop,
        ),
    )


def _should_download_path(path, suffixes=None):
    path = path.strip('"')
    ext = os.path.splitext(path)[1]
    if ext not in (".zarr", "") and (suffixes is None or ext in suffixes):
        try:
            fs, _ = fsspec.core.url_to_fs(path)
            if _get_fs_protocol(fs) != "file":
                return True
        except ValueError:
            pass
            # ignore unrecognized protocol
    return False


def _download_path(path, directory=None):
    fs, _ = fsspec.core.url_to_fs(path)
    fd, local_path = tempfile.mkstemp(dir=directory, suffix=os.path.splitext(path)[1])
    os.close(fd)
    fs.get(path, local_path)
    return local_path


def _download_paths(
    filelist: Sequence[Sequence[str]],
    suffixes: set[str] | None,
    tmp_dir: str = None,
    path_to_local_path: dict[str, str] = None,
):
    if path_to_local_path is None:
        path_to_local_path = dict()
    if suffixes is not None and len(suffixes) == 0:
        return path_to_local_path, tmp_dir
    for paths in filelist:
        for i in range(len(paths)):
            path = paths[i]
            if path not in path_to_local_path and _should_download_path(path, suffixes):
                path_to_local_path[path] = None

    if len(path_to_local_path) > 0:
        remote_paths = list(path_to_local_path.keys())
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        local_paths = (
            db.from_sequence(remote_paths)
            .map(_download_path, directory=tmp_dir)
            .compute()
        )

        for i in range(len(remote_paths)):
            path_to_local_path[remote_paths[i]] = local_paths[i]

    return path_to_local_path, tmp_dir


def _replace_remote_paths(
    filelist: Sequence[Sequence[str]], path_to_local_path: dict[str, str]
):
    new_filelist = []
    for paths in filelist:
        paths = [path_to_local_path.get(path, path) for path in paths]
        new_filelist.append(paths)

    return new_filelist


def _best_focus_z_index(img: xr.DataArray):
    if "z" not in img.dims:
        return -1
    if img.sizes["z"] == 1:
        return 0
    best_z_index = 0
    best_value = -np.inf
    for z_index in range(img.sizes["z"]):
        val = power_spectrum(img.isel(z=z_index).values)
        if val != 0 and val > best_value:
            best_z_index = z_index
            best_value = val
    return best_z_index


@delayed
def _power_spectrum_delayed(file_list, attrs, scene_id, channel, tmp_dir):
    img = _images2fov(
        file_list, attrs, dask=False, scene_id=scene_id, tmp_dir=tmp_dir
    ).isel(t=0, c=channel, missing_dims="ignore")
    return _best_focus_z_index(img)


def _init_tiles(
    image_filepaths: Sequence[Sequence[str]],
    image_metadata: dict,
    channel: int = 0,
    z_index: int | Literal["max", "focus"] | str = "max",
    expected_images: int | None = None,
    download_suffixes: set[str] | None = None,
):
    if not isinstance(z_index, int) and z_index not in ("max", "focus"):
        z_index = pd.read_parquet(z_index.format(**image_metadata["file_metadata"][0]))
        z_index["key"] = z_index["key"].apply(lambda x: tuple(x))
        z_index = z_index.set_index("key")
        assert "z_index" in z_index.columns
    n_scenes = None
    if len(image_filepaths) == 1:
        img = _create_image(image_filepaths[0])
        ome_metadata = _get_ome(img)
        n_scenes = len(ome_metadata.images)

    fileattrs = None
    metadata_fields = []
    if "file_metadata" in image_metadata:
        metadata_fields = [
            v for v in ["c", "z"] if v in image_metadata["file_metadata"][0]
        ]
    if len(metadata_fields) > 0:
        keys, filepaths, fileattrs = _group_src_attrs(
            metadata=image_metadata, metadata_fields=tuple(metadata_fields)
        )

    else:
        filepaths = []
        for i in range(len(image_filepaths)):
            filepaths.append((image_filepaths[i],))
        keys = filepaths
    if isinstance(z_index, pd.DataFrame):
        z_index = z_index.loc[keys]["z_index"].values

    if expected_images is not None:
        assert len(filepaths) == expected_images, (
            f"Expected {expected_images} but found {len(filepaths)}."
        )
    n = n_scenes if n_scenes is not None else len(filepaths)
    logger.info(f"{n:,} tiles.")
    path_to_local_path = None
    tmp_dir = None
    if isinstance(z_index, str) and z_index == "focus":
        results = []
        separate_channel_files = "c" in metadata_fields
        filepaths_ = filepaths
        fileattrs_ = fileattrs
        if separate_channel_files:
            channel_str = str(channel)
            channel = 0
            filepaths_, fileattrs_ = _filter_filepaths(
                filepaths, fileattrs, ["c"], [channel_str]
            )

        path_to_local_path, tmp_dir = _download_paths(filepaths_, download_suffixes)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        filepaths_ = _replace_remote_paths(filepaths_, path_to_local_path)
        for i in range(n):
            file_list = filepaths_[0] if n_scenes is not None else filepaths_[i]
            attrs = fileattrs_[i] if fileattrs_ is not None else None
            scene_id = i if n_scenes is not None else None
            results.append(
                _power_spectrum_delayed(file_list, attrs, scene_id, channel, tmp_dir)
            )
        z_index = dask.compute(*results)
        logger.info("Determined best focus z-index for all tiles.")
    if (
        "z" in metadata_fields
        and isinstance(z_index, (Sequence, np.ndarray, int))
        and not isinstance(z_index, str)
    ):  # only keep needed files with computed z
        filepaths, fileattrs = _filter_filepaths(filepaths, fileattrs, ["z"], [z_index])
    path_to_local_path, tmp_dir = _download_paths(
        filelist=filepaths,
        suffixes=download_suffixes,
        tmp_dir=tmp_dir,
        path_to_local_path=path_to_local_path,
    )

    return dict(
        keys=keys,
        z_index=z_index,
        n_scenes=n_scenes,
        fileattrs=fileattrs,
        filepaths=_replace_remote_paths(filepaths, path_to_local_path),
        tmp_dir=tmp_dir,
        metadata_fields=metadata_fields,
        original_filepaths=filepaths,
    )


def _filter_filepaths(filepaths, fileattrs, metadata_fields, metadata_values):
    new_filepaths = []
    new_fileattrs = []
    nfields = len(metadata_fields)
    for tile_index in range(len(filepaths)):
        file_list = filepaths[tile_index]
        attrs = fileattrs[tile_index]
        file_metadata = attrs["file_metadata"]
        file_list_ = []
        file_metadata_ = []
        for i in range(len(file_list)):
            keep = True
            md = file_metadata[i]
            for field_index in range(nfields):
                keep_value = metadata_values[field_index]
                if isinstance(keep_value, (np.ndarray, Sequence)) and not isinstance(
                    keep_value, str
                ):
                    keep_value = keep_value[tile_index]
                metadata_value = md[metadata_fields[field_index]]
                keep_value = type(metadata_value)(keep_value)
                if metadata_value != keep_value:
                    keep = False
                    break
            if keep:
                file_list_.append(file_list[i])
                file_metadata_.append(md)
        assert len(file_list_) > 0
        new_filepaths.append(file_list_)
        new_attrs = attrs.copy()
        new_attrs["file_metadata"] = file_metadata_
        new_fileattrs.append(new_attrs)
    return new_filepaths, new_fileattrs
