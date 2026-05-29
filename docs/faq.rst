************
FAQ
************


#. I have separate images for each channel or z-index.
    Use "c" and "z" to indicate where in the file name the channel and z-index are located. Note that
    "tile" is also included in the image pattern in order to distinguish each individual image tile.

    Example::

        scallops stitch --image-pattern "e100_1_t0_{well}_s{tile}_w{c}_z{z}.tif" \
        --images my-images/ --groupby well --image-output stitch.zarr --report-output reports/

#. ImportError installing SCALLOPS
    Please ensure TMPDIR has execution permissions or temporarily set TMPDIR to a directory that does::

        TMPDIR=. pip install scallops

#. Unable to download model for Stardist or other deep learning models
    Some workflow environments (e.g. AWS HealthOmics) prohibit accesssing public websites.
    You need to download model files to an accessible location, and set the workflow input, `model_dir`
    to this location.

    Example::

        wget https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip
        aws s3 cp python_2D_versatile_fluo.zip s3://my-bucket/model/

    In your OPS workflow input JSON, set `model_dir` to `s3://my-bucket/model/`

#. How do I pass configuration parameters when reading or writing to S3, GCP, or other file systems?
    SCALLOPS uses fsspec_ for I/O operations. Please see `fsspec configuration documentation`_ for more details.

    Example::

        export FSSPEC_s3='{"anon":true}'



#. How long it takes SCALLOPS to install?
    This is highly dependant on your mahcine, internet speed, network speed, and resource manege used.
    On a Macbook air laptop, using regular pip install command on a virtual environment on a comercial network:

    ::

        real	2m32.125s
        user	0m38.533s
        sys	0m15.782s

    but using uv engine:
    ::

        real	0m26.296s
        user	0m4.659s
        sys	0m6.972s

.. _fsspec: https://filesystem-spec.readthedocs.io/
.. _`fsspec configuration documentation`: https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration
