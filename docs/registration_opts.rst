Predefined Registration parameters
----------------------------------------

Scallops provides a set of predefined parameters for registration.

Available Options
^^^^^^^^^^^^^^^^^^

- rigid

   * Description: Rigid registration using 1 resolution and a small step size.
   * Transformations: Translation, Rotation

- affine

   * Description: Affine registration using 1 resolution and a small step size.
   * Transformations: Translation, Rotation, Scaling, Shearing

- nl-100

   * Description: Non-linear registration using B-splines using 1 resolution and a final grid spacing of 100 microns..
   * Transformations: Non-linear (B-spline)


These parameters use mutual information as the image similarity measure; advanced mean squares and advanced normalized correlation versions of these options are available with
the suffixes `ams` and `anc` respectively. For example, `rigid-anc`.

Note that parameters can be composed in any manner. For example `rigid affine nl-100`.

In order to use custom registration parameters, pass a set of JSON files to the `itk-parameters` argument.
Please refer to the `Elastix <ElastixWebsite_>`_ manual for more information.



.. _ElastixWebsite: https://elastix.dev/
