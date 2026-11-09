Predefined Registration parameters
----------------------------------------

SCALLOPS provides a set of predefined parameters for registration. The default is ``affine nl-100``.

Available Options
^^^^^^^^^^^^^^^^^^

- rigid

   * Description: Rigid registration using 1 resolution and a small step size.
   * Transformations: Translation, Rotation

- affine

   * Description: Affine registration using 1 resolution and a small step size.
   * Transformations: Translation, Rotation, Scaling, Shearing

- nl-100

   * Description: Non-linear registration using B-splines using 1 resolution and a final grid spacing of 100 physical units (microns when the image spacing is in microns).
   * Transformations: Non-linear (B-spline)


These parameters use mutual information as the image similarity measure; advanced mean squares and advanced normalized correlation versions of these options are available with
the suffixes `ams` and `anc` respectively. For example, `rigid-anc`.

Note that parameters can be composed in any manner. For example `rigid affine nl-100`.

To use custom registration parameters, pass JSON or elastix text parameter files to ``--itk-parameters``.
Please refer to the `Elastix <ElastixWebsite_>`_ manual for more information.



.. _ElastixWebsite: https://elastix.dev/
