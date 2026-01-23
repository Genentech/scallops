************
API
************

SCALLOPS provides an extensive API that supports data reading, writing, processing, and visualization,
enabling users to work with OPS data efficiently, and be able to customize it to their needs.

Experiment Data Structure
=========================
The core data structure used to organize and handle experiments.

.. autosummary::
    :toctree: .

    scallops.experiment.elements.Experiment


Reading
=======
Modules for reading and handling input data, including image data, barcode data, and spatial information.

.. autosummary::
    :toctree: .

    scallops.io.get_image_spacing
    scallops.io.get_pixel_positions
    scallops.io.physical_coords_to_px
    scallops.io.read_barcodes
    scallops.io.read_experiment
    scallops.io.read_image
    scallops.io.read_in_situ_reads_and_barcodes
    scallops.zarr_io.read_ome_zarr_array


Writing
=======
Functions for saving images and metadata in various formats.

.. autosummary::
    :toctree: .

    scallops.experiment.elements.Experiment.save
    scallops.io.create_multiscene_hyperstack
    scallops.io.save_ome_tiff
    scallops.io.save_stack_imagej
    scallops.io.to_label_crops
    scallops.zarr_io.write_zarr


Features
===================
Tools to extract and analyze features from regions of interest, such as nuclei .

.. autosummary::
    :toctree: .

    scallops.features.find_objects.find_objects
    scallops.features.generate.label_features
    scallops.features.normalize.normalize_features
    scallops.features.rank.rank_features
    scallops.features.agg.agg_features



Registration
============
Modules for image alignment and registration.

Cross-Correlation
-------------------------------
    Alignment based on `SKimage' phase cross correlation`_

        .. autosummary::
            :toctree: .

            scallops.registration.crosscorrelation.align_image
            scallops.registration.crosscorrelation.align_images


ITK
-----------------
    Alignment based on ITK_

        .. autosummary::
            :toctree: .

            scallops.registration.itk.itk_align
            scallops.registration.itk.itk_align_to_reference_time
            scallops.registration.itk.itk_transform_image
            scallops.registration.itk.itk_transform_labels


Landmarks
-------------------------------
    Find matching landmarks between two images

        .. autosummary::
            :toctree: .

            scallops.registration.landmarks.find_landmarks
            scallops.registration.landmarks.match_template


Pixel-Based Decoding
====================
Functions for decoding images at the pixel level using codebooks.

.. autosummary::
    :toctree: .

    scallops.codebook.decode_metric
    scallops.codebook.estimate_scale_factors
    scallops.codebook.image_to_codes
    scallops.codebook.unit_norm

Spot-Based Decoding
===================
Tools for decoding sequencing spots and assigning them to barcodes and labels.

.. autosummary::
    :toctree: .


    scallops.reads.apply_channel_crosstalk_matrix
    scallops.reads.assign_barcodes_to_labels
    scallops.reads.barcode_to_prefix
    scallops.reads.base_counts
    scallops.reads.channel_crosstalk_matrix
    scallops.reads.correct_mismatches
    scallops.reads.decode_max
    scallops.reads.merge_sbs_phenotype
    scallops.reads.peaks_to_bases
    scallops.reads.quality_softmax
    scallops.reads.read_statistics
    scallops.reads.summarize_base_call_mismatches
    scallops.spots.find_peaks
    scallops.spots.max_filter
    scallops.spots.normalize_base_intensities
    scallops.spots.peak_thresholds_from_bases
    scallops.spots.peak_thresholds_from_reads
    scallops.spots.std
    scallops.spots.transform_log




Segmentation
============

Nuclei Segmentation
-------------------
    Segmentation tools for identifying nuclei in images, using popular algorithms such as Stardist.

    .. autosummary::
        :toctree: .

        scallops.segmentation.cellpose.segment_nuclei_cellpose
        scallops.segmentation.stardist.segment_nuclei_stardist
        scallops.segmentation.watershed.segment_nuclei_watershed


Cell Segmentation
-------------------
    Tools for segmenting cells, using various algorithms, including watershed and propagation methods.

    .. autosummary::
        :toctree:

        scallops.segmentation.cellpose.segment_cells_cellpose
        scallops.segmentation.cyto_channel_summary
        scallops.segmentation.propagation.segment_cells_propagation
        scallops.segmentation.watershed.segment_cells_watershed

Utilities
==========
    Segmentation utility functions.

    .. autosummary::
        :toctree: .

        scallops.segmentation.util.area_overlap
        scallops.segmentation.util.close_labels
        scallops.segmentation.util.cyto_channel_summary
        scallops.segmentation.util.image2mask
        scallops.segmentation.util.remove_boundary_labels
        scallops.segmentation.util.remove_labels_by_area
        scallops.segmentation.util.remove_labels_region_props
        scallops.segmentation.util.remove_masked_regions
        scallops.segmentation.util.remove_small_objects_std
        scallops.segmentation.util.threshold_quantile



Xarray/Experiment Custom Operations
=====================================
Custom operations for working with Xarray and experiments.

.. autosummary::
    :toctree: .

    scallops.xr.apply_data_array
    scallops.xr.iter_data_array
    scallops.experiment.util.map_images



Visualization
=============
Visualization utilities for creating composite images, plots, and segmentation visuals.

Composites
-----------

    .. autosummary::
        :toctree: .

        scallops.visualize.composite.experiment_composite
        scallops.visualize.composite.imcomposite
        scallops.visualize.composite.label_montage
        scallops.visualize.composite.montage_plot

Cross-talk
-----------

    .. autosummary::
        :toctree: .

        scallops.visualize.crosstalk.pairwise_channel_scatter_plot

Distributions
-------------

    .. autosummary::
        :toctree: .

        scallops.visualize.distribution.cdf_plot
        scallops.visualize.distribution.comparative_effect_scatter
        scallops.visualize.distribution.ridge_plot
        scallops.visualize.distribution.volcano_plot

Heatmaps
-----------

    .. autosummary::
        :toctree: .

        scallops.visualize.heatmap.base_call_mismatches_heatmap
        scallops.visualize.heatmap.in_situ_identity_matrix_plot
        scallops.visualize.heatmap.plate_heatmap
        scallops.visualize.heatmap.plot_well_aggregated_heatmaps

Histograms
-----------

    .. autosummary::
        :toctree: .

        scallops.visualize.histogram.channel_hist_plot
        scallops.visualize.histogram.in_situ_barcode_hist_plot

Imshow
------

    .. autosummary::
        :toctree: .

        scallops.visualize.imshow.imshow_plane
        scallops.visualize.imshow.plot_percentile_montage
        scallops.visualize.imshow.plot_plate
        scallops.visualize.imshow.tiles_over_stitch

Napari
-----------

    .. autosummary::
        :toctree: .

        scallops.visualize.napari.imnapari
        scallops.visualize.napari.experiment_napari
        scallops.visualize.napari.add_bases
        scallops.visualize.napari.radial_distortion_estimation

Registration plots
-------------------

    .. autosummary::
        :toctree: .

        scallops.visualize.registration.diagnose_registration
        scallops.visualize.registration.plot_registration

Segmentation plots
-------------------

    .. autosummary::
        :toctree: .

        scallops.visualize.segmentation.plot_segmentation


Visualization Utilities
=======================
Additional utilities for visualizing multicolor labels and channels.

.. autosummary::
    :toctree: .

    scallops.visualize.utils.channel_thresholds
    scallops.visualize.utils.multicolor_labels


Stitching Utilities
===================
Utilities specifically for stitching operations.

.. autosummary::
    :toctree: .

    scallops.stitch.utils.tile_overlap_mask


Datasets
===================
Builtin datasets for testing

.. autosummary::
    :toctree: .

    scallops.datasets.example_feature_summary_stats
    scallops.datasets.feldman_2019_small

.. _`SKimage' phase cross correlation`: https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
.. _ITK: https://itk.org/Doxygen/html/RegistrationPage.html
