Custom Features
----------------

Defining Custom Features


To define a custom feature, create a function that takes a `RegionProperties` object as an input and returns a computed
value. Custom feature functions can optionally accept additional arguments if needed.

Example:

.. code-block:: python

    from skimage.measure import regionprops

    def custom_feature(region: RegionProperties) -> float:
        """
        Calculate a custom feature for a region.

        :param region: Instance of `RegionProperties` representing the region.
        :return: Computed feature value as a float.
        """
        # Example: Return the ratio of the area to the perimeter
        return region.area / region.perimeter

Using Custom Features
^^^^^^^^^^^^^^^^^^^^^^

Once the custom feature functions are defined, they can be passed to the `label_features` or
`compute_features` functions via the `custom_features` parameter.

Example:

.. code-block:: python

    import xarray as xr
    import numpy as np
    from skimage.measure import regionprops_table
    from your_package.features import custom_feature

    # Define custom features
    custom_features = {
        'custom_feature': custom_feature
    }

    # Example image and labels
    image = xr.DataArray(np.random.random((100, 100, 3)), dims=["y", "x", "c"])
    labels = np.random.randint(0, 2, size=(100, 100))

    # Define the features to be extracted
    features = {
        'cell': ['area', 'perimeter', 'custom_feature']
    }

    # Compute features
    df = compute_features(
        image=image,
        labels={'cell': labels},
        features=features,
        custom_features=custom_features
    )

    print(df)

Example Custom Feature Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def mean_intensity(region: RegionProperties, channel: int = 0) -> float:
        """
        Calculate the mean intensity of a specific channel for a region.

        :param region: Instance of `RegionProperties` representing the region.
        :param channel: Channel index to compute the mean intensity for.
        :return: Mean intensity value as a float.
        """
        intensity_image = region.intensity_image[..., channel]
        return intensity_image.mean()

Registering Custom Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using custom features, it's important to ensure they are properly registered and passed to the feature extraction
functions. Here's how you can include multiple custom features:

.. code-block:: python

    # Define additional custom features
    def max_intensity(region: RegionProperties, channel: int = 0) -> float:
        intensity_image = region.intensity_image[..., channel]
        return intensity_image.max()

    custom_features = {
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity
    }

    # Compute features including custom features
    df = compute_features(
        image=image,
        labels={'cell': labels},
        features=features,
        custom_features=custom_features
    )

    print(df)

By defining and registering custom features, users can extend the functionality of the feature extraction pipeline to
meet their specific needs.
