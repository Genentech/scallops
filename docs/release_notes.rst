*****************
Release Notes
*****************

* Unreleased
    * When registering timepoints of a moving image with different numbers of channels without
      ``--unroll-channels``, pad timepoints with leading empty channels to match the timepoint with
      the most channels instead of unrolling channels.
      ``--moving-channel`` indexes the padded channels.
    * Wheels no longer include the test suite. The ``miniwdl_local`` miniwdl container backend
      moved to ``scallops._miniwdl_local``.

* v1.0.1, August 4, 2026
    * Remove support for zarr <3.
    * Only compute features for cells with a valid barcode by default in WDL workflow.
    * Add option to align across channels in stitching within a tile when outputting stitched image.
    * Add option to select time index for illumination correction and stitching (e.g. for live imaging based phenotype data)

* v1.0.0, March 19, 2026
    * Initial release.
