"""Experiment Utility Module.

This module provides utility functions to manipulate `Experiment` objects, including applying functions
to images or labels, and concatenating images within an experiment.

Authors:
    - The SCALLOPS development team.
"""

from collections.abc import Callable, Sequence
from typing import Literal

import dask.bag as db
import xarray as xr

from scallops.experiment.elements import Experiment


def map_images(
    experiment: Experiment | Sequence[Experiment],
    func: Callable,
    keys: Sequence[str] = None,
    output_type: Literal["images", "labels"] = "images",
    **kwargs,
):
    """Apply a function to all images in one or more experiments.

    :param experiment: Experiment(s) to map.
    :param func: Function to apply.
    :param keys: Optional subset of image keys in experiment(s) to map.
    :param output_type: Whether the output of `func` is an image or label.
    :param kwargs: Keyword arguments to pass to `func`.
    :return: Experiment mapping keys to images or labels.
    """
    if isinstance(experiment, Experiment):
        experiment = [experiment]

    if keys is None:
        if len(experiment) == 1:
            keys = experiment[0].images.keys()
        else:
            # Take the intersection of all keys
            keys = dict.fromkeys(experiment[0].images.keys())
            for i in range(1, len(experiment)):
                experiment_keys = experiment[i].images.keys()
                keys = [key for key in keys if key in experiment_keys]

    images = [tuple([exp.images[key] for exp in experiment]) for key in keys]
    results = db.from_sequence(images).starmap(func, **kwargs).compute()

    if output_type == "images":
        return Experiment(dict(zip(keys, results)))
    elif output_type == "labels":
        return Experiment(labels=dict(zip(keys, results)))
    else:
        raise NotImplementedError("Experiment only takes images or label types")


def _concat_images(
    experiment: Experiment, image_keys: Sequence[str] | None = None
) -> xr.DataArray:
    """Concatenate images.

    :param experiment: The experiment
    :param image_keys: List of image keys or None to concatenate all images in the experiment.
    :return: DataArray with the dimension i (image) added
    """
    if image_keys is None:
        image_keys = experiment.images.keys()
    images = [experiment.images[key] for key in image_keys]
    image = xr.concat(images, dim="i")
    image = image.assign_coords(i=list(image_keys))
    return image
