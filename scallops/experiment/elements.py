"""SCALLOPS Spatial Experiment Module.

This module provides classes and methods for managing spatial experiments, including image and label
data. The experiments are saved in the Zarr format, leveraging Dask for handling large datasets.
"""

from collections.abc import Mapping

import numpy as np
import xarray as xr
from zarr.storage import StoreLike

from scallops.experiment.abc import _LazyLoadData


class _DataAccessorDict:
    """A dictionary-like class for managing and lazy-loading data.

    This class is designed to store data, which may be loaded lazily, meaning that the data is only
    loaded into memory when it is accessed.
    """

    def __init__(self):
        """Initialize an empty DataAccessorDict."""
        self._d = {}

    def __getitem__(self, key):
        """Retrieve an item from the data accessor dictionary.

        :param key: The key associated with the data.
        :return: The data associated with the key, or None if not found.
        """
        value = self._d.get(key)
        if value is None:
            return None
        return value.data if isinstance(value, _LazyLoadData) else value

    def __contains__(self, key):
        """Check if a key is in the data accessor dictionary.

        :param key: The key to check for.
        :return: True if the key exists, False otherwise.
        """
        return key in self._d

    def __iter__(self):
        """Return an iterator over the dictionary keys.

        :return: An iterator over the keys.
        """
        return iter(self._d)

    def keys(self):
        """Return the keys of the dictionary.

        :return: A set-like object containing the keys of the dictionary.
        """
        return self._d.keys()

    def __len__(self):
        """Return the number of items in the dictionary.

        :return: The number of items in the dictionary.
        """
        return len(self._d)

    def __setitem__(self, key, value):
        """Set an item in the dictionary.

        :param key: The key associated with the data.
        :param value: The data to be stored.
        """
        self._d[key] = value

    def __delitem__(self, key):
        """Delete an item from the dictionary.

        :param key: The key associated with the data to be deleted.
        """
        del self._d[key]

    def __repr__(self):
        """Return a string representation of the dictionary.

        :return: A string representation of the dictionary's keys.
        """
        return ", ".join(self._d.keys())


class Experiment:
    """Spatial experiment containing images and labels.

    Maps keys (e.g. A01-102) to images and labels.
    """

    def __init__(
        self,
        images: Mapping[str, xr.DataArray] = None,
        labels: Mapping[str, np.ndarray | xr.DataArray] = None,
    ):
        """Create an experiment.

        :param images: Maps image key to image.
        :param labels: Maps label key to label.
        """
        self._images = _DataAccessorDict()
        self._labels = _DataAccessorDict()
        if images:
            for key in images:
                self._images[key] = images[key]
        if labels:
            for key in labels:
                self._labels[key] = labels[key]

    @property
    def images(self) -> Mapping[str, xr.DataArray]:
        """The image dictionary."""
        return self._images

    @property
    def labels(self) -> Mapping[str, np.ndarray | xr.DataArray]:
        """The labels dictionary."""
        return self._labels

    def save(self, store: StoreLike):
        """Save this experiment in Zarr format.

        :param store: Either a Zarr store or a string.
        """
        from scallops.zarr_io import (
            _write_zarr_image,
            _write_zarr_labels,
            open_ome_zarr,
        )

        root = open_ome_zarr(store, mode="w")

        for name in self._images:
            _write_zarr_image(
                name=name,
                root=root,
                image=self._images[name],
            )
        for name in self._labels:
            labels = self._labels[name]
            _write_zarr_labels(
                name=name,
                root=root,
                metadata=labels.attrs if isinstance(labels, xr.DataArray) else None,
                labels=labels,
            )
        if len(self._labels) > 0:
            labels_grp = root["labels"]
            labels_grp.attrs["labels"] = list(self._labels.keys())

    def __repr__(self):
        """Return a string representation of the experiment.

        :return: A string describing the number of images and labels in the experiment.
        """
        st = ["Experiment with "]
        nimages = len(self._images)
        nlabels = len(self._labels)
        st.append("{} image".format(nimages) + ("s" if nimages != 1 else ""))
        st.append(" and {} label".format(nlabels) + ("s" if nlabels != 1 else ""))
        return "".join(st)
