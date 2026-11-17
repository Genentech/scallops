************
FAQ
************


#. How do I stitch images stored as separate files for each channel or z-index?
    Use "c" and "z" to indicate where in the file name the channel and z-index are located. Note that
    "tile" is also included in the image pattern in order to distinguish each individual image tile.

    Example::

        scallops stitch --image-pattern "e100_1_t0_{well}_s{tile}_w{c}_z{z}.tif" \
        --images my-images/ --groupby well --image-output stitch.zarr --report-output reports/

#. Why do I get an ImportError when installing SCALLOPS?
    Please ensure TMPDIR has execution permissions or temporarily set TMPDIR to a directory that does::

        TMPDIR=. pip install scallops

#. How do I use StarDist or other deep learning models when the workflow can't download them?
    Some workflow environments (e.g. AWS HealthOmics) prohibit accessing public websites.
    You need to download model files to an accessible location, and set the workflow input, ``model_dir``
    to this location.

    Example::

        wget https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip
        aws s3 cp python_2D_versatile_fluo.zip s3://my-bucket/model/2D_versatile_fluo.zip

    SCALLOPS looks for the StarDist model as ``2D_versatile_fluo.zip``, which is why the ``aws s3 cp``
    command above gives the file that name.
    In your OPS workflow input JSON, set ``model_dir`` to ``s3://my-bucket/model/``

#. How do I pass configuration parameters when reading or writing to S3, GCP, or other file systems?
    SCALLOPS uses fsspec_ for I/O operations. Please see `fsspec configuration documentation`_ for more details.

    Example::

        export FSSPEC_s3='{"anon":true}'



#. How long does it take to install SCALLOPS?
    This depends on your machine, network speed, and package manager. On a MacBook Air laptop, using
    ``pip install`` in a virtual environment:

    ::

        real	2m32.125s
        user	0m38.533s
        sys	0m15.782s

    and using ``uv pip install``:
    ::

        real	0m26.296s
        user	0m4.659s
        sys	0m6.972s

#. Why are my channels not aligned with each other in the stitched image?
    Chromatic aberration can make each channel slightly magnified and shifted relative to the others. Use
    ``--alignment-correction-channels`` to align channels to the reference channel when stitching. See
    :ref:`channel_alignment` for the parameter file format and how missing values are estimated.

.. _fsspec: https://filesystem-spec.readthedocs.io/
.. _`fsspec configuration documentation`: https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration
