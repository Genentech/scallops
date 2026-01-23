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
