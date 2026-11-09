*****************
Release Notes
*****************

* v1.0.2, November 9, 2026
    * The ``norm-features`` and ``rank-features`` commands are replaced by the new ``pert-map``
      command, with subcommands ``normalize``, ``filter``, ``rank``, ``aggregate``, ``pca``,
      ``tvn``, ``similarity-matrix``, ``enrichment`` and ``recall``.
    * ``imcomposite`` now takes separate row and column facets.
    * Add the ``puncta`` feature, which detects small bright spots (vesicles, granules, Golgi
      fragments) in each label and measures their number, size, intensity and spatial
      distribution, using a threshold relative to each cell's local background.
    * Add ``segment overlap`` (also added to the WDL workflow), which computes the overlap between nuclei and cells and renumbers cells to match nuclei for non watershed-based cell segmentation algorithms.
    * Segmentation can use a separate channel and smoothing for the threshold mask
      (``--mask-channel``, ``--mask-sigma``).
    * Stitching can correct alignment between channels
      (``--alignment-correction-channels``, ``--alignment-correction-params``,
      ``--alignment-correction-channel-order``). Parameters are estimated per well when not
      given.
    * Stitching detects Araceli JSON metadata automatically.
    * ``--labels`` accepts more than one directory. When a label appears in several, the last
      one wins.
    * Faster read calling and barcode assignment.
    * Stitching releases the GIL, doesn't lock when writing the fused image, and no longer
      needs shared memory.
    * ITK registration loads images it reuses into memory only once.
    * Processes writing to the same zarr store can now create shared groups at the same time
      without failing.
    * When registering timepoints of a moving image with different numbers of channels without
      ``--unroll-channels``, pad timepoints with leading empty channels to match the timepoint
      with the most channels instead of unrolling channels. ``--moving-channel`` indexes the
      padded channels.



* v1.0.1, August 4, 2026
    * Remove support for zarr <3.
    * Only compute features for cells with a valid barcode by default in WDL workflow.
    * Add option to align across channels in stitching within a tile when outputting stitched image.
    * Add option to select time index for illumination correction and stitching (e.g. for live imaging based phenotype data)

* v1.0.0, March 19, 2026
    * Initial release.
