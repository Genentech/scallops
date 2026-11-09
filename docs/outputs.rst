.. _outputs:

Outputs
=======

This section describes the files written by the main SCALLOPS commands. Examples are
from processing a single image group, ``A1-102`` (well ``A1``, tile ``102``).

Each image group is identified by a key built from the ``--groupby`` fields (for
example ``A1-102`` for ``--groupby well tile``). Most commands write one output per
key, so they can be run on many groups in parallel and skip groups that already have
output unless ``--force`` is given.

Tables are written as Parquet. A path such as ``A1-102.parquet`` may be a directory of
Parquet part files; read it with ``pandas.read_parquet`` or ``dask.dataframe.read_parquet``.

Images and labels
-----------------

Images and segmentation labels are written as `OME-Zarr <https://ngff.openmicroscopy.org/0.4/>`__
(version 0.4, stored in Zarr format 2). Images are under ``images/`` and labels under
``labels/``, each named by the image key. Label names add a suffix such as ``-nuclei``
or ``-cell``:

.. code:: text

   segment.zarr/
   ├── images/
   │   └── A1-102/
   │       ├── .zattrs        # OME-Zarr multiscales metadata
   │       └── s0/            # full-resolution array
   └── labels/
       ├── A1-102-nuclei/
       │   └── s0/
       ├── A1-102-cell/
       │   └── s0/
       └── A1-102-cytosol/
           └── s0/

Read them with :func:`scallops.io.read_experiment`, or open a single array with
``scallops.zarr_io.open_ome_zarr`` and :func:`scallops.zarr_io.read_ome_zarr_array`.
The stores can also be viewed in napari with the ``napari-ome-zarr`` plugin (install the
``napari`` extra).

Unless ``--no-version`` is given, commands store their command-line arguments and the
SCALLOPS version in the output metadata.

Segmentation (segment)
----------------------

``scallops segment nuclei`` writes ``labels/{key}-nuclei`` to the output Zarr.
``scallops segment cell`` writes ``labels/{key}-cell`` and, when nuclei labels are
provided, ``labels/{key}-cytosol`` (cell minus nuclei). Label values are object IDs;
``0`` is background. With the watershed and propagation methods, each cell has the same
ID as its nucleus.

When objects are filtered by area or intensity (``--min-area``, ``--max-area``,
``--min-intensity``, ``--max-intensity``), the unfiltered labels are also written as
``{key}-nuclei.all`` or ``{key}-cell.all``.

In situ sequencing (pooled-sbs)
-------------------------------

Spot detection (``pooled-sbs spot-detect``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writes a Zarr store containing candidate spots and intermediate images:

.. code:: text

   spot-detect.zarr/
   ├── images/
   │   ├── A1-102-max/        # max filter of the LoG image, used for read calling
   │   ├── A1-102-log/        # Laplacian of Gaussian filtered image (--save log)
   │   └── A1-102-std/        # standard deviation over cycles (--save std)
   └── points/
       └── A1-102-peaks.parquet

``points/{key}-peaks.parquet`` has one row per candidate spot:

======== ==================================================
Column   Description
======== ==================================================
y, x     Spot position in pixels
sigma    LoG sigma at which the spot was detected
peak     Peak score; higher values indicate stronger spots
======== ==================================================

Read calling (``pooled-sbs reads``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writes a directory with one subdirectory per output type:

.. code:: text

   reads/
   ├── reads/A1-102.parquet       # one row per spot
   ├── labels/A1-102.parquet      # one row per cell (barcode assignment)
   ├── bases/A1-102.parquet       # one row per spot, cycle and base (--save-bases)
   ├── crosstalk/A1-102-w.zarr    # crosstalk correction matrix
   └── spots/                     # computed peak thresholds

The crosstalk matrix is written unless ``--crosstalk-correction-method none`` is used.
By default only spots inside a label (``--label-name``) are called; use ``--all-labels``
to also call spots outside labels (assigned label ``0``).

``reads/{key}.parquet``:

================== ==============================================================
Column             Description
================== ==============================================================
y, x, sigma, peak  Spot position, LoG sigma and peak score from spot detection
label              ID of the label containing the spot (``0`` = outside labels)
read               Spot ID
barcode            Called sequence, one base per cycle
Q                  Per-cycle base quality
Q_mean, Q_min      Mean and minimum of ``Q`` across cycles
barcode_match      Whether ``barcode`` is in the barcodes file
================== ==============================================================

``labels/{key}.parquet`` assigns barcodes to each label from the reads it contains.
Reads with the same barcode are grouped and ranked by their summed ``Q_mean`` (then
summed ``peak``); the top two barcodes are reported with suffixes ``_0`` and ``_1``:

========================================== ==================================================
Column                                     Description
========================================== ==================================================
label                                      Label ID
barcode_0, barcode_1                       Top two barcodes
barcode_count                              Total number of reads in the label
barcode_count_{i}                          Number of reads with ``barcode_{i}``
barcode_Q_mean_{i}, barcode_Q_min_{i}      Summed ``Q_mean`` and ``Q_min`` of the reads with
                                           ``barcode_{i}``
barcode_Q_{i}                              Summed per-cycle ``Q`` of the reads with
                                           ``barcode_{i}``
barcode_peak_{i}                           Summed peak score of the reads with ``barcode_{i}``
mismatch                                   ``True`` if no read in the label exactly matched a
                                           barcode in the barcodes file, so barcodes were
                                           assigned from reads with mismatches (see
                                           ``--mismatches``)
========================================== ==================================================

``bases/{key}.parquet`` (with ``--save-bases``) has the intensity of every base at every
cycle for each spot: ``read``, ``c`` (base), ``t`` (cycle), ``y``, ``x``, ``sigma``,
``peak``, ``label``, ``intensity``, and ``corrected_intensity`` (after crosstalk
correction).

Merging (``pooled-sbs merge``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Joins the barcode assignments (``reads/labels``) with the phenotype tables from
``scallops features`` and ``scallops find-objects`` and annotates each barcode using the
barcodes CSV. Writes one table per image key, ``{output}/{key}.parquet`` (or Zarr with
``--format zarr``), indexed by ``label``. Columns include the ``labels`` columns above,
all phenotype feature columns, and the barcodes CSV columns for ``barcode_0`` (for example
``sgRNA``, ``gene_symbol``) and for ``barcode_1`` (with suffix ``_1``).

Use ``--join-sbs`` and ``--join-phenotype`` to choose whether cells missing from the
SBS or phenotype tables are kept (``outer``) or dropped (``inner``).

Features (features, find-objects)
---------------------------------

``scallops features`` writes one directory per label type, with a features table and an
objects table for each image key:

.. code:: text

   features/
   ├── cell/
   │   ├── A1-102.parquet          # requested features
   │   └── A1-102-objects.parquet  # bounding box, centroid and area
   └── nuclei/
       ├── A1-102.parquet
       └── A1-102-objects.parquet

Rows are labels, indexed by label ID. Columns follow CellProfiler naming,
``{Compartment}_{Category}_{Feature}_{Channel}``, for example
``Nuclei_Intensity_MeanIntensity_Channel0``, ``Cells_AreaShape_Area``, or
``Nuclei_Correlation_Pearson_Channel0_Channel1``. Channels are numbered from 0.

``scallops find-objects`` writes only the objects table,
``{output}/{label_name}/{key}-objects.parquet``, with columns such as
``Cells_AreaShape_BoundingBoxMinimum_Y``, ``Cells_AreaShape_Center_X`` and
``Cells_AreaShape_Area``.

Illumination correction (illum-corr agg)
----------------------------------------

Writes one illumination correction (flat-field) image per image key, either as
OME-TIFF (``{output}/{key}.ome.tiff``, the default) or OME-Zarr
(``--output-image-format zarr``). With ``--z-index focus``, the z-index
chosen for each tile is also saved to ``{key}-zindex.parquet``.

Dialout analysis (dialout)
--------------------------

``scallops dialout analysis`` writes, for each pool (named after the FASTQ file prefix):

==================================== ===================================================
File                                 Contents
==================================== ===================================================
``{pool}.sam``                       Reads aligned to the reference with BWA
``{pool}.counts.csv``                Read count per guide sequence: ``sequence``,
                                     ``count``, ``count_fraction``, the design CSV
                                     columns, ``in_design``, ``mismatches`` and
                                     ``closest_match`` (nearest design sequence for
                                     sequences not in the design)
``{pool}.dropout.csv``               Design sequences with no reads
``{pool}.stats.csv``                 Summary statistics: ``n_mapped``,
                                     ``fraction_mapped``, ``average_read_count``,
                                     ``skew_ratio`` (90th / 10th percentile of counts),
                                     ``drop_out_ratio`` and ``n_drop_outs``
``{pool}-reference-unaligned.fa``,   Unaligned and unmatched reads
``{pool}-dialout-unaligned.fa``      (``--save-unaligned-fasta``)
==================================== ===================================================

Example ``{pool}.stats.csv`` values for three pools:

======== ======== =============== ================== ========== ============== ===========
index    n_mapped fraction_mapped average_read_count skew_ratio drop_out_ratio n_drop_outs
======== ======== =============== ================== ========== ============== ===========
T2-A06   643024   0.660           392.98             3.12       0.0            0
T2-A04   382737   0.661           235.98             3.19       0.0015         1
T2-A05   583195   0.657           354.96             3.10       0.0            0
======== ======== =============== ================== ========== ============== ===========

``scallops dialout report`` combines the per-pool results into plots (summary
statistics, matches, mismatches and guide ranks), written to a PDF or, if the output is a
directory, as individual PNG files.
