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

#. My channels are not aligned with each other in the stitched image.
    Chromatic aberration can make each channel slightly magnified and shifted relative to the others. Use
    ``--alignment-correction-channels`` to align channels to the reference channel (``--align-channel``) when writing
    the stitched image, and ``--alignment-correction-channel-order`` to name every input channel, in input order (use
    ``-`` for a channel without values). Names are matched to the parameter file as text, ignoring case.

    Values are read from a JSON parameter file (``--alignment-correction-params``), usually one per microscope.
    ``m`` is the magnification about the tile centre and ``ty``, ``tx`` the translation in pixels, such that
    ``corrected(y, x) = raw(c + (1 + m) * ((y, x) - c) + (ty, tx))`` with ``c`` the tile centre. ``same_as`` reuses the
    values of another channel.

    Example::

        {
          "format": "scallops-channel-alignment",
          "version": 1,
          "tile_shape": [2304, 2304],
          "reference_channel": "DAPI",
          "channels": {
            "FITC": {"m": 0.000751, "ty": 0.341, "tx": -0.162},
            "CY7": {"m": -0.001493, "ty": 0.950, "tx": 0.300}
          }
        }

    Channels without values (or all channels, if no parameter file is given) are estimated from up to 100 interior
    tiles of each well, using the nuclei of the reference channel, which must be a nuclear stain such as DAPI. If an
    estimate fails its quality checks, stitching stops with an error; give values for that channel in the parameter
    file or leave it out of ``--alignment-correction-channels``. The values applied to each image are saved to
    ``<image>-channel-alignment.json`` in the report output and to the stitched image metadata.

    When channel alignment is requested, every output channel is resampled once with a quintic B-spline (alignment and
    radial correction combined) instead of the bilinear radial correction, to keep fine detail.
    ``--alignment-correction-channels`` cannot be combined with ``--channel-reference``.

    Correcting a channel's shift leaves a band of missing pixels along some tile edges (pixels that would come from
    outside the tile). The crop applied when writing the stitched image is widened to remove these bands. If a band is
    wider than half the typical overlap between neighbouring tiles, cropping it would open gaps between tiles, so
    stitching stops with an error: leave that channel out of ``--alignment-correction-channels`` or give smaller
    parameters for it (which may correct it only partly). If the tiles could not be aligned and were placed at their
    stage positions, nothing is cropped, so stitching stops with an error when a corrected channel would leave a band;
    run without ``--alignment-correction-channels``. The crop used is saved in the per-image record (``fusion_crop``).

.. _fsspec: https://filesystem-spec.readthedocs.io/
.. _`fsspec configuration documentation`: https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration
