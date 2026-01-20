"""Find Matching Landmarks Across Images.

Authors:
    - The SCALLOPS development team.
"""

import logging
from typing import Literal

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from scipy.ndimage import center_of_mass
from skimage.feature import match_template as sk_match_template
from skimage.transform import resize
from sklearn.linear_model import RANSACRegressor

from scallops.io import get_image_spacing, pluralize

logger = logging.getLogger("scallops")


def _match_template_result(result):
    indices = np.argpartition(result.flatten(), -2)[-2:]
    ij = np.unravel_index(indices, result.shape)
    i, j = ij

    v1 = result[i[0], j[0]]
    v2 = result[i[1], j[1]]
    idx = i[0], j[0]
    ratio = 0
    if v2 > v1:
        idx = i[1], j[1]
        tmp = v2
        v2 = v1
        v1 = tmp
    dist = np.sqrt((i[0] - i[1]) ** 2 + (j[0] - j[1]) ** 2)
    if dist > 2:
        ratio = v2 / v1
    return idx, v1, ratio


def _match_template_physical_space_inputs(
    image: xr.DataArray,
    template: xr.DataArray,
    y_start: float,
    x_start: float,
    slice_size: float,
    template_padding: float,
    template_labels: xr.DataArray | None = None,
    image_labels: xr.DataArray | None = None,
    translation: tuple[float, float] | None | Literal["com", "none"] = "com",
    image_spacing: tuple[float, float] | None = None,
    template_spacing: tuple[float, float] | None = None,
) -> dict:
    """Template matching in physical space.

    :param image: The image to match
    :param template: The template to search
    :param y_start: The starting y position in physical coordinates
    :param x_start: The starting x position in physical coordinates
    :param slice_size: The slice size in physical coordinates
    :param template_padding: Template padding in physical coordinates.
    :param template_labels: Template labels.
    :param image_labels: Image labels.
    :param translation: Global translation between template and image in physical
        coordinates.
    :param image_spacing: Image spacing.
    :param template_spacing: Template spacing.
    """
    image_spacing = (
        get_image_spacing(image.attrs) if image_spacing is None else image_spacing
    )
    template_spacing = (
        get_image_spacing(template.attrs)
        if template_spacing is None
        else template_spacing
    )
    end = y_start + slice_size, x_start + slice_size
    image_start = max(0, y_start / image_spacing[0]), max(0, x_start / image_spacing[1])
    image_end = (
        min(image.shape[0], end[0] / image_spacing[0]),
        min(image.shape[1], end[1] / image_spacing[1]),
    )
    template_start = y_start / template_spacing[0], x_start / template_spacing[1]
    template_end = end[0] / template_spacing[0], end[1] / template_spacing[1]
    if translation is None:
        translation = np.array([0, 0])
    translation = np.asarray(translation) / template_spacing
    template_padding = (
        template_padding / template_spacing[0],
        template_padding / template_spacing[1],
    )
    template_y = (
        int(max(0, template_start[0] - template_padding[0] + translation[0])),
        int(
            min(
                template.data.shape[0],
                template_end[0] + template_padding[0] + translation[0],
            )
        ),
    )
    template_x = (
        int(max(0, template_start[1] - template_padding[1] + translation[1])),
        int(
            min(
                template.data.shape[1],
                template_end[1] + template_padding[1] + +translation[1],
            )
        ),
    )

    template = template.data[
        template_y[0] : template_y[1], template_x[0] : template_x[1]
    ]

    if template_labels is not None:
        template_labels = template_labels.data[
            template_y[0] : template_y[1], template_x[0] : template_x[1]
        ]

    image = image.isel(
        y=slice(int(image_start[0]), int(image_end[0])),
        x=slice(int(image_start[1]), int(image_end[1])),
    ).data
    if image_labels is not None:
        image_labels = image_labels.isel(
            y=slice(int(image_start[0]), int(image_end[0])),
            x=slice(int(image_start[1]), int(image_end[1])),
        ).data
    shape_ratio = np.array(template_spacing) / np.array(image_spacing)
    return dict(
        template=template,
        template_start=template_start,
        template_y=template_y,
        template_x=template_x,
        y_start=y_start,
        x_start=x_start,
        image=image,
        image_spacing=image_spacing,
        shape_ratio=shape_ratio,
        template_labels=template_labels,
        image_labels=image_labels,
    )


@delayed
def _match_template_slice_delayed(
    template: np.ndarray,
    template_start: tuple[float, float],
    template_y: tuple[int, int],
    template_x: tuple[int, int],
    image: np.ndarray,
    image_spacing: tuple[float, float],
    shape_ratio: np.ndarray,
    template_labels: np.ndarray | None,
    image_labels: np.ndarray | None,
    y_start: float,
    x_start: float,
) -> pd.DataFrame:
    result = _match_template_slice(
        template=template,
        template_start=template_start,
        template_y=template_y,
        template_x=template_x,
        image=image,
        image_spacing=image_spacing,
        shape_ratio=shape_ratio,
        template_labels=template_labels,
        image_labels=image_labels,
    )

    if result is not None:
        result.pop("template")
        result.pop("image")
        result.pop("template_labels")
        result.pop("image_labels")
        result["y_start_microns"] = y_start
        result["x_start_microns"] = x_start
        return pd.DataFrame.from_dict([result])
    return pd.DataFrame(
        {
            "i": pd.Series(dtype="int"),
            "j": pd.Series(dtype="int"),
            "score": pd.Series(dtype="float"),
            "ratio": pd.Series(dtype="float"),
            "shift_microns_y": pd.Series(dtype="float"),
            "shift_microns_x": pd.Series(dtype="float"),
            "n_template_labels": pd.Series(dtype="int"),
            "n_image_labels": pd.Series(dtype="int"),
            "jaccard": pd.Series(dtype="float"),
            "y_start_microns": pd.Series(dtype="float"),
            "x_start_microns": pd.Series(dtype="float"),
        }
    )


def _match_template_slice(
    template: np.ndarray,
    template_start: tuple[float, float],
    template_y: tuple[int, int],
    template_x: tuple[int, int],
    image: np.ndarray,
    image_spacing: tuple[float, float],
    shape_ratio: np.ndarray,
    template_labels: np.ndarray | None,
    image_labels: np.ndarray | None,
) -> dict | None:
    new_shape = (np.array(template.shape) * shape_ratio).astype(int)
    if new_shape[0] < image.shape[0] or new_shape[1] < image.shape[1]:
        return None
    template = resize(template, new_shape)
    n_template_labels_full = None
    if template_labels is not None:
        template_labels = resize(template_labels, new_shape, order=0)
        unique_labels = np.unique(template_labels)
        unique_labels = unique_labels[unique_labels != 0]
        n_template_labels_full = len(unique_labels)
    n_image_labels = None
    if image_labels is not None:
        unique_labels = np.unique(image_labels)
        unique_labels = unique_labels[unique_labels != 0]
        n_image_labels = len(unique_labels)

    if (
        template.max() - template.min() == 0
        or image.max() - image.min() == 0
        or n_image_labels == 0
        or n_template_labels_full == 0
    ):
        return None
    else:
        result = sk_match_template(
            template,
            image,
        )
        result = _match_template_result(result)
    ij = result[0]
    template_padding = np.array(
        [
            template_start[0] - template_y[0],
            template_start[1] - template_x[0],
        ]
    )
    shift_pixels = ij - (template_padding * shape_ratio)
    shift_microns = shift_pixels * image_spacing
    jaccard = None
    n_template_labels = None
    if template_labels is not None:
        matching_template_labels = template_labels[
            ij[0] : ij[0] + image.shape[0],
            ij[1] : ij[1] + image.shape[1],
        ]
        unique_labels = np.unique(matching_template_labels)
        unique_labels = unique_labels[unique_labels != 0]
        n_template_labels = len(unique_labels)

    if template_labels is not None and image_labels is not None:
        matching_template_labels_binary = matching_template_labels.astype(bool).astype(
            np.uint8
        )
        image_labels_binary = image_labels.astype(bool).astype(np.uint8)
        intersection = (matching_template_labels_binary & image_labels_binary).sum()
        union = (matching_template_labels_binary | image_labels_binary).sum()
        jaccard = intersection / union if union != 0 else 0

    return dict(
        i=result[0][0],
        j=result[0][1],
        score=result[1],
        ratio=result[2],
        shift_microns_y=shift_microns[0],
        shift_microns_x=shift_microns[1],
        template=template,
        image=image,
        n_template_labels=n_template_labels,
        n_image_labels=n_image_labels,
        template_labels=template_labels,
        image_labels=image_labels,
        jaccard=jaccard,
    )


def match_template(
    image: xr.DataArray,
    template: xr.DataArray,
    y_start: float,
    x_start: float,
    slice_size: float = 200,
    template_padding: float = 500,
    template_labels: xr.DataArray | None = None,
    image_labels: xr.DataArray | None = None,
    translation: tuple[float, float] | None | Literal["com", "none"] = "com",
    image_spacing: tuple[float, float] | None = None,
    template_spacing: tuple[float, float] | None = None,
) -> dict | None:
    """Match image to template.

    :param image: The image to match
    :param template: The template to search
    :param y_start: The y start in physical coordinates of the image
    :param x_start: The x start in physical coordinates of the image
    :param slice_size: The slice size of the image in physical coordinates
    :param template_padding: Template padding in physical coordinates.
    :param template_labels: Optional template labels.
    :param image_labels: Optional image labels.
    :param translation: Global translation between template and image in physical coordinates. Use
        "com" for center of mass.
    :param image_spacing: Optional image spacing. If not provided, determined from metadata.
    :param template_spacing: Optional template spacing. If not provided, determined from metadata.
    :return: Result dictionary or None.
    """
    inputs = _match_template_physical_space_inputs(
        image=image,
        template=template,
        y_start=y_start,
        x_start=x_start,
        slice_size=slice_size,
        template_padding=template_padding,
        template_labels=template_labels,
        image_labels=image_labels,
        translation=translation,
        image_spacing=image_spacing,
        template_spacing=template_spacing,
    )

    return _match_template_slice(
        template=(
            inputs["template"].compute()
            if isinstance(inputs["template"], da.Array)
            else inputs["template"]
        ),
        template_start=inputs["template_start"],
        template_y=inputs["template_y"],
        template_x=inputs["template_x"],
        image=(
            inputs["image"].compute()
            if isinstance(inputs["image"], da.Array)
            else inputs["image"]
        ),
        image_spacing=inputs["image_spacing"],
        shape_ratio=inputs["shape_ratio"],
        template_labels=(
            inputs["template_labels"].compute()
            if isinstance(inputs["template_labels"], da.Array)
            else inputs["template_labels"]
        ),
        image_labels=(
            inputs["image_labels"].compute()
            if isinstance(inputs["image_labels"], da.Array)
            else inputs["image_labels"]
        ),
    )


def _center_of_mass(
    image: xr.DataArray,
    image_spacing: tuple[float, float],
    min_quantile: float | None = 0.25,
    max_quantile: float | None = 0.75,
) -> np.ndarray:
    values = image.values
    if min_quantile is not None or max_quantile is not None:
        q = []
        if min_quantile is not None:
            q.append(min_quantile)
        if max_quantile is not None:
            q.append(max_quantile)
        q = np.quantile(values, q, axis=None)
        q_str = ", ".join([f"{val:,.1f}" for val in q])
        logger.debug(f"{pluralize('Quantile', len(q))} for center of mass: {q_str}")
        if min_quantile is not None and max_quantile is not None:
            labels = ((values <= q[1]) & (values >= q[0])).astype(np.uint8)
        elif min_quantile is not None:
            labels = (values >= q[0]).astype(np.uint8)
        else:
            labels = (values <= q[0]).astype(np.uint8)
        values = np.multiply(values, labels)
    return np.array(center_of_mass(values)) * np.array(image_spacing)


def _get_translation(
    image: xr.DataArray,
    template: xr.DataArray,
    translation: tuple[float, float] | None | Literal["com", "none"] = "com",
    image_spacing: tuple[float, float] | None = None,
    template_spacing: tuple[float, float] | None = None,
    com_min_quantile: float | None = 0.25,
    com_max_quantile: float | None = 0.75,
):
    if isinstance(translation, str):
        image_spacing = (
            get_image_spacing(image.attrs) if image_spacing is None else image_spacing
        )
        template_spacing = (
            get_image_spacing(template.attrs)
            if template_spacing is None
            else template_spacing
        )
        if translation == "com":
            image_com = _center_of_mass(
                image=image,
                image_spacing=image_spacing,
                min_quantile=com_min_quantile,
                max_quantile=com_max_quantile,
            )
            template_com = _center_of_mass(
                image=template,
                image_spacing=template_spacing,
                min_quantile=com_min_quantile,
                max_quantile=com_max_quantile,
            )

            translation = template_com - image_com
            logger.info(
                f"Center of mass shift microns: {translation[0]:,.1f}, {translation[1]:,.1f}"
            )
        elif translation == "none":
            translation = None
        else:
            raise ValueError(f"Unknown translation: {translation}")
    return translation


def find_landmarks(
    image: xr.DataArray,
    template: xr.DataArray,
    slice_size: float = 200,
    template_padding: float = 750,
    step_size: float = 1000,
    template_labels: xr.DataArray | None = None,
    image_labels: xr.DataArray | None = None,
    translation: tuple[float, float] | None | Literal["com", "none"] = "com",
    image_spacing: tuple[float, float] | None = None,
    template_spacing: tuple[float, float] | None = None,
    com_min_quantile: float | None = 0.25,
    com_max_quantile: float | None = 0.75,
) -> dd.DataFrame:
    """Match image to template across a grid of coordinates.

    :param image: The image to match
    :param template: The template to search
    :param slice_size: The slice size of the image in physical coordinates
    :param template_padding: Template padding in physical coordinates.
    :param step_size: Image step size.
    :param template_labels: Optional template labels.
    :param image_labels: Optional image labels.
    :param translation: Global translation between template and image in physical
        coordinates. Use "com" for center of mass.
    :param image_spacing: Optional image spacing. If not provided, determined from
        metadata.
    :param template_spacing: Optional template spacing. If not provided, determined
        from metadata.
    :param com_max_quantile: Include values <= specified quantile for center of mass
        computation.
    :param com_min_quantile: Include values >= specified quantile for center of mass
        computation.
    :return: Delayed object that resolves to a data frame.
    """

    image_spacing = (
        get_image_spacing(image.attrs) if image_spacing is None else image_spacing
    )
    template_spacing = (
        get_image_spacing(template.attrs)
        if template_spacing is None
        else template_spacing
    )
    translation = _get_translation(
        translation=translation,
        image=image,
        template=template,
        image_spacing=image_spacing,
        template_spacing=template_spacing,
        com_min_quantile=com_min_quantile,
        com_max_quantile=com_max_quantile,
    )
    results = []

    meta = dd.utils.make_meta(
        pd.DataFrame(
            {
                "i": pd.Series(dtype="int"),
                "j": pd.Series(dtype="int"),
                "score": pd.Series(dtype="float"),
                "ratio": pd.Series(dtype="float"),
                "shift_microns_y": pd.Series(dtype="float"),
                "shift_microns_x": pd.Series(dtype="float"),
                "n_template_labels": pd.Series(dtype="int"),
                "n_image_labels": pd.Series(dtype="int"),
                "jaccard": pd.Series(dtype="float"),
                "y_start_microns": pd.Series(dtype="float"),
                "x_start_microns": pd.Series(dtype="float"),
            }
        )
    )

    for y_start in np.arange(0, image.sizes["y"] * image_spacing[0], step_size):
        for x_start in np.arange(0, image.sizes["x"] * image_spacing[1], step_size):
            inputs = _match_template_physical_space_inputs(
                image=image,
                template=template,
                y_start=y_start,
                x_start=x_start,
                slice_size=slice_size,
                template_padding=template_padding,
                template_labels=template_labels,
                image_labels=image_labels,
                translation=translation,
                image_spacing=image_spacing,
                template_spacing=template_spacing,
            )
            result = _match_template_slice_delayed(
                template=inputs["template"],
                template_start=inputs["template_start"],
                template_y=inputs["template_y"],
                template_x=inputs["template_x"],
                image=inputs["image"],
                image_spacing=inputs["image_spacing"],
                shape_ratio=inputs["shape_ratio"],
                template_labels=inputs["template_labels"],
                image_labels=inputs["image_labels"],
                y_start=y_start,
                x_start=x_start,
            )
            results.append(result)
    df = dd.from_delayed(results, meta=meta, verify_meta=False)
    df["moving_y_microns"] = df["y_start_microns"] + df["shift_microns_y"]
    df["moving_x_microns"] = df["x_start_microns"] + df["shift_microns_x"]
    _find_landmarks_inliers_delayed = delayed(_find_landmarks_inliers)
    return _find_landmarks_inliers_delayed(df)


def _find_landmarks_inliers(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1:
        df["inlier"] = False
    else:
        y = df[["shift_microns_y", "shift_microns_x"]]
        residual_threshold = np.median(np.abs(y - np.median(y))) * 0.6744897501960817
        reg = RANSACRegressor(
            random_state=0, loss="squared_error", residual_threshold=residual_threshold
        )
        reg.fit(
            df[["y_start_microns", "x_start_microns"]],
            y,
        )
        df["inlier"] = reg.inlier_mask_
    return df
