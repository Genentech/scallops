Predefined Registration parameters
----------------------------------------

Scallops provides a set of predefined parameters for registration. Options that end in wsireg where adopted from WSIreg_.

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


- rigid-wsireg

   * Description: Rigid registration using 10 resolutions.
   * Transformations: Translation, Rotation

- affine-wsireg

   * Description: Affine registration using 10 resolutions.
   * Transformations: Translation, Rotation, Scaling, Shearing

- similarity-wsireg

   * Description: Similarity registration using 10 resolutions.
   * Transformations: Translation, Rotation, Uniform Scaling

- nl-wsireg

   * Description: Non-linear registration using B-splines using 10 resolutions and a final grid spacing of 100 microns.
   * Transformations: Non-linear (B-spline)

- nl2-wsireg

   * Description: Non-linear registration using B-splines using 10 resolutions and a final grid spacing of 75 microns.
   * Transformations: Non-linear (B-spline)

- nl3-wsireg

   * Description: Non-linear registration using B-splines using 1 resolution and a final grid spacing of 200 microns.
   * Transformations: Non-linear (B-spline)

- fi_correction-wsireg

   * Description: Rigid registration using 4 resolutions.
   * Transformations: Translation, Rotation



These parameters use mutual information as the image similarity measure; advanced mean squares and advanced normalized correlation versions of these options are available with
the suffixes `ams` and `anc` respectively. For example, `rigid-anc`.

Note that parameters can be composed in any manner. For example `rigid affine nl-100`.

In order to use custom registration parameters, pass a set of JSON files to the `itk-parameters` argument.
Please refer to the `Elastix <ElastixWebsite_>`_ manual for more information.


.. _WSIreg: https://github.com/NHPatterson/wsireg
.. _ElastixWebsite: https://elastix.dev/
