Outputs explained
=================


Scallop’s command line has a series of default and optional outputs that
will be described below.

In-Situ sequencing pipeline (pooled-sbs)
----------------------------------------

Let’s first touch a bit (for more info check the `CLI
documentation <https://gred-cumulus.pages.roche.com/scallops/command_line.html>`__.
Let’s say that we want to run in-situ sequencing pipeline from the
command line using stardist (the default) as nuclei segmentation,
followed by watershed cell segmentation with threshold defined by the
Li’s method. Let’s assume that you would like to use the test files and
that your current working directory is Scallop’s root directory. Then:

.. code:: bash

   scallops pooled-sbs pipeline scallops/tests/data/experimentC/input  --barcodes scallops/tests/data/experimentC/barcodes.csv --pheno=scallops/tests/data/experimentC/10X_c0-DAPI-p65ab

Will generate the following directories:

.. code:: bash

   bases       cells       combined    images.zarr phenos      reads

In ``bases`` you will find the dataframes in parquet format of the bases
information extracted from your groups. Since by default we group by
tile and well, and we only have one well and two tiles, you’ll find:

.. code:: bash

   ./bases
   ├── A1-102.parquet
   └── A1-103.parquet

   1 directory, 2 files

Both files contain the read id, cycle, channel, intensity, cell id,
coordinates y and x, well and tile information:

+---+-----+------+--------------------+------+---+---+--------------------+---------------------+------+------+
| y | x   | read | peak               | cell | t | c | intensity          | corrected_intensity | well | tile |
+===+=====+======+====================+======+===+===+====================+=====================+======+======+
| 5 | 705 | 0    | 364.59483811395535 | 0    | 1 | G | 538.2688477073593  | -48.44409230154324  | A1   | 102  |
| 5 | 705 | 0    | 364.59483811395535 | 0    | 1 | T | 2706.2792184281157 | 3510.5365471687214  | A1   | 102  |
+---+-----+------+--------------------+------+---+---+--------------------+---------------------+------+------+


Likewise, in the ``reads`` directory, you’ll find the dataframes containing the reads info:

.. code:: bash

   ./reads
   ├── A1-102.parquet
   └── A1-103.parquet

   1 directory, 2 files

Containing the identified reads information such as quantiles and peaks:

+---+-----+------+--------------------+------+-----------+--------------------+--------------------+---------------------+--------------------+--------------------+--------------------+--------------------+----------------------+--------------------+----------------------+------+------+
| y | x   | read | peak               | cell | barcode   | Q_0                | Q_1                | Q_2                 | Q_3                | Q_4                | Q_5                | Q_6                | Q_7                  | Q_8                | Q_min                | well | tile |
+===+=====+======+====================+======+===========+====================+====================+=====================+====================+====================+====================+====================+======================+====================+======================+======+======+
| 5 | 705 | 0    | 364.59483811395535 | 0    | TATTCTTCC | 0.8225319160232156 | 0.213521539841818  | 0.5315020019263625  | 1.0                | 0.1871177283289207 | 0.6452085822773499 | 1.0                | 0.008381143001541913 | 0.1944106438921418 | 0.008381143001541913 | A1   | 102  |
| 5 | 756 | 1    | 1162.3744344193526 | 0    | AAGCCAATT | 1.0                | 0.8525545481276788 | 0.41514790479643904 | 0.4095041269847144 | 1.0                | 0.5086887399634992 | 0.9443351500907629 | 0.5966046145349035   | 1.0                | 0.4095041269847144   | A1   | 102  |
+---+-----+------+--------------------+------+-----------+--------------------+--------------------+---------------------+--------------------+--------------------+--------------------+--------------------+----------------------+--------------------+----------------------+------+------+


Then,  the ``cells`` directory includes parquet files with cell information:

.. code:: bash

   ./cells
   ├── A1-102.parquet
   └── A1-103.parquet

   1 directory, 2 files

with the barcodes counts and their corresponding peaks and sequences:

+-------------------+------+----------------+----------------------+----------------+----------------------+---------------+------+------+
| peak              | cell | cell_barcode_0 | cell_barcode_count_0 | cell_barcode_1 | cell_barcode_count_1 | barcode_count | well | tile |
+===================+======+================+======================+================+======================+===============+======+======+
| 438.1286071891493 | 36   | GACCAATGG      | 4                    | ACCGGTTTA      | 1.0                  | 5             | A1   | 102  |
| 398.4562017293166 | 17   | CTTCGCACT      | 2                    |                | 0.0                  | 2             | A1   | 102  |
+-------------------+------+----------------+----------------------+----------------+----------------------+---------------+------+------+


Finally, we have the ``combined`` directory with all the information combined:

.. code:: bash

   ./combined/
   ├── A1-102.parquet
   └── A1-103.parquet

   1 directory, 2 files

+------+------+------+--------------------+----------------+----------------------+----------------+----------------------+---------------+-------------------+--------------------+------------+--------------+--------------------+--------------------+-------------------+-----------------+--------------+-------------+-------------------+-----------------+--------------------+----------------------+-------------+------------------+----------------------+---------------+--------------------+
| well | tile | cell | peak               | cell_barcode_0 | cell_barcode_count_0 | cell_barcode_1 | cell_barcode_count_1 | barcode_count | cells_x           | cells_y            | cells_area | nuclei_max_1 | nuclei_mean_1      | nuclei_corr_0_1    | nuclei_y          | nuclei_median_1 | nuclei_max_0 | nuclei_area | nuclei_x          | nuclei_median_0 | nuclei_mean_0      | sgRNA                | gene_symbol | duplicate_prefix | sgRNA_1              | gene_symbol_1 | duplicate_prefix_1 |
+======+======+======+====================+================+======================+================+======================+===============+===================+====================+============+==============+====================+====================+===================+=================+==============+=============+===================+=================+====================+======================+=============+==================+======================+===============+====================+
| A1   | 102  | 17   | 398.4562017293166  | CTTCGCACT      | 2.0                  |                | 0.0                  | 2.0           | 821.2             | 8.631578947368421  | 95.0       | 2819         | 2280.957746478873  | 0.87922644251275   | 7.788732394366197 | 2278.0          | 1646         | 71.0        | 821.7605633802817 | 1237.0          | 1239.338028169014  | CTTCGACACTGATGATCTGC | ATXN3L      | False            |                      |               |                    |
| A1   | 102  | 19   | 147.69660807883213 | GCTGCAGTC      | 1.0                  | CAAATCCCA      | 1.0                  | 2.0           | 890.7610062893082 | 10.754716981132075 | 159.0      | 2122         | 1663.0069444444443 | 0.7376070465901201 | 10.11111111111111 | 1682.0          | 1792         | 144.0       | 891.2569444444445 | 1411.0          | 1363.8402777777778 | GCTGCAAGTCTCCCACCGGA | SMAD1       | False            | CAAATCCCCAACTCATCTCG | RNF24         | False              |
+------+------+------+--------------------+----------------+----------------------+----------------+----------------------+---------------+-------------------+--------------------+------------+--------------+--------------------+--------------------+-------------------+-----------------+--------------+-------------+-------------------+-----------------+--------------------+----------------------+-------------+------------------+----------------------+---------------+--------------------+

I have purposely left the zarr directory for last.


Zarr output
------------

Note that you can choose to generate tiff instead of zarr images
(see documentation for more information) and also can control which images are saved.

We follow the OME-ZARR format, which according to `Open
microscopy <https://ngff.openmicroscopy.org/latest/>`__: >OME-Zarr is an
implementation of the OME-NGFF specification using the Zarr format.
Arrays MUST be defined and stored in a hierarchical organization as
defined by the version 2 of the Zarr specification . OME-NGFF metadata
MUST be stored as attributes in the corresponding Zarr groups.

In short is a hierarchical way to storing images that is very amenable
for cloud computing. Going back to our example above, our default
``images.zarr`` contains:

.. code:: bash

   ./images.zarr/
   ├── .zgroup
   ├── A1-102
   │   ├── .zattrs
   │   ├── .zgroup
   │   ├── 0
   │   └── labels
   ├── A1-102-phenotype
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   ├── A1-103
   │   ├── .zattrs
   │   ├── .zgroup
   │   ├── 0
   │   └── labels
   └── A1-103-phenotype
       ├── .zattrs
       ├── .zgroup
       └── 0

Which contain the images and labels of each of the groupings. Notice
that there are also metadata hidden files called ``.zattrs`` and
``.zgroup`` which contain metadata about each level group and attributes
(i.e. the way to organize the inner structure). Let’s zoom in to only
one of the groupings:

.. code:: bash

   /Users/hleaploj/Playground/testzarr/default/images.zarr
   ├── .zgroup
   ├── A1-102
   │   ├── .zattrs
   │   ├── .zgroup
   │   ├── 0
   │   │   ├── .zarray
   │   │   ├── 0
   │   │   │   ├── 0
   │   │   │   │   └── 0
   │   │   │   │       ├── 0
   │   │   │   │       │   ├── 0
   │   │   │   │       │   ├── 1
   │   │   │   │       │   ├── 2
   │   │   │   │       │   └── 3
   │   │   │   │       ├── 1
   │   │   │   │       │   ├── 0
   │   │   │   │       │   ├── 1
   │   │   │   │       │   ├── 2
   │   │   │   │       │   └── 3
   │   │   │   │       ├── 2
   │   │   │   │       │   ├── 0
   │   │   │   │       │   ├── 1
   │   │   │   │       │   ├── 2
   │   │   │   │       │   └── 3
   │   │   │   │       └── 3
   │   │   │   │           ├── 0
   │   │   │   │           ├── 1
   │   │   │   │           ├── 2
   │   │   │   │           └── 3
   │   │   │   ├── 1
   │   │   │   │   └── 0
   .   .   .   .   .
   .   .   .   .   .
   .   .   .   .   .
   │   │       └── 2
   │   │           └── 0
   │   │               ├── 0
   │   │               │   ├── 0
   │   │               │   ├── 1
   │   │               │   ├── 2
   │   │               │   └── 3
   │   │               ├── 1
   .   .               .   .
   .   .               .   .
   .   .               .   .
   │   │               └── 3
   │   │                   ├── 0
   │   │                   ├── 1
   │   │                   ├── 2
   │   │                   └── 3
   │   └── labels
   │       ├── .zattrs
   │       ├── .zgroup
   │       ├── cell
   │       │   ├── .zattrs
   │       │   ├── .zgroup
   │       │   └── 0
   │       │       ├── .zarray
   │       │       ├── 0
   │       │       │   ├── 0
   │       │       │   └── 1
   │       │       ├── 1
   .       .       .   .
   .       .       .   .
   .       .       .   .
   │       │       └── 3
   │       │           ├── 0
   │       │           └── 1
   │       ├── cytosol
   │       │   ├── .zattrs
   │       │   ├── .zgroup
   │       │   └── 0
   │       │       ├── .zarray
   │       │       ├── 0
   │       │       │   ├── 0
   │       │       │   └── 1
   │       │       ├── 1
   .       .       .   .
   .       .       .   .
   .       .       .   .
   │       │       └── 3
   │       │           ├── 0
   │       │           └── 1
   │       ├── iss-spots
   │       │   ├── .zattrs
   │       │   ├── .zgroup
   │       │   └── 0
   │       │       ├── .zarray
   │       │       ├── 0
   │       │       │   ├── 0
   │       │       │   ├── 1
   │       │       │   ├── 2
   │       │       │   └── 3
   │       │       ├── 1
   .       .       .   .
   .       .       .   .
   .       .       .   .
   │       │       └── 3
   │       │           ├── 0
   │       │           ├── 1
   │       │           ├── 2
   │       │           └── 3
   │       └── nuclei
   │           ├── .zattrs
   │           ├── .zgroup
   │           └── 0
   │               ├── .zarray
   │               ├── 0
   │               │   ├── 0
   │               │   └── 1
   │               ├── 1
   .               .   .
   .               .   .
   .               .   .

   │               └── 3
   │                   ├── 0
   │                   └── 1
   └── A1-102-phenotype
      ├── .zattrs
       ├── .zgroup
       └── 0
           ├── .zarray
           ├── 0
           │   ├── 0
           │   │   ├── 0
           │   │   └── 1
           │   └── 1
           │       ├── 0
           │       └── 1
           └── 1
               ├── 0
               │   ├── 0
               │   └── 1
               └── 1
                   ├── 0
                   └── 1

Here we see an extra hidden file, ``.zarray``, that informs where the
actual image starts (as opposed to groups). In the above case we can see
that the image of well A1 and tile 102, has a root called ``A1-102``,
with three groups (0, 1 and 2), which according to `the official
format <https://ngff.openmicroscopy.org/latest/>`__: >Each multiscale
level is stored as a separate Zarr array, which is a folder containing
chunk files which compose the array. > The name of the array is
arbitrary with the ordering defined by the “multiscales” metadata, but
is often a sequence starting at 0.

Therein are all the chunked data: >Chunks are stored with the nested
directory layout. All but the last chunk element are stored as
directories. The terminal > chunk is a file. Together the directory and
file names provide the “chunk coordinate” (t, c, z, y, x), where the
maximum > coordinate will be dimension_size / chunk_size.

We then see the ``labels`` group. We store here all the segmentation
information including nuclei, cells and cytosol. All these follow the
OME-ZARR schema: >All labels will be listed in .zattrs. Each dimension
of the label (t, c, z, y, x) should be either the same as the >
corresponding dimension of the image, or 1 if that dimension of the
label is irrelevant.

Storing intermediate outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CLI, through the option ``--save`` by including the things to save
(`see CLI
documentation <https://gred-cumulus.pages.roche.com/scallops/command_line.html#scallops-pooled-if-sbs>`__):
>–save Outputs to save. Choose from
cell-labels,nuclei-labels,cytosol-labels,spot-labels,aligned,cell-mask,max,log,std,peaks,
> phenotype-aligned,bases,reads,cells,phenotype,combined,crosstalk >
Default:
cell-labels,nuclei-labels,cytosol-labels,spot-labels,aligned,phenotype-aligned,bases,reads,cells,phenotype,combined

If you use them all, you’ll find (just showing first 3 levels):

.. code:: bash

   ./images.zarr
   ├── .zgroup
   ├── A1-102
   │   ├── .zattrs
   │   ├── .zgroup
   │   ├── 0
   │   │   ├── .zarray
   │   │   ├── 0
   │   │   ├── 1
   │   │   └── 2
   │   └── labels
   │       ├── .zattrs
   │       ├── .zgroup
   │       ├── cell
   │       ├── cytosol
   │       ├── iss-spots
   │       └── nuclei
   ├── A1-102-cell-mask
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       ├── 2
   │       └── 3
   ├── A1-102-log
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       └── 2
   ├── A1-102-max
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       └── 2
   ├── A1-102-peaks
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       ├── 2
   │       └── 3
   ├── A1-102-phenotype
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       └── 1
   ├── A1-102-std
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       ├── 2
   │       └── 3
   ├── A1-103
   │   ├── .zattrs
   │   ├── .zgroup
   │   ├── 0
   │   │   ├── .zarray
   │   │   ├── 0
   │   │   ├── 1
   │   │   └── 2
   │   └── labels
   │       ├── .zattrs
   │       ├── .zgroup
   │       ├── cell
   │       ├── cytosol
   │       ├── iss-spots
   │       └── nuclei
   ├── A1-103-cell-mask
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       ├── 2
   │       └── 3
   ├── A1-103-log
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       └── 2
   ├── A1-103-max
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       └── 2
   ├── A1-103-peaks
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       ├── 1
   │       ├── 2
   │       └── 3
   ├── A1-103-phenotype
   │   ├── .zattrs
   │   ├── .zgroup
   │   └── 0
   │       ├── .zarray
   │       ├── 0
   │       └── 1
   └── A1-103-std
       ├── .zattrs
       ├── .zgroup
       └── 0
           ├── .zarray
           ├── 0
           ├── 1
           ├── 2
           └── 3


Illumination correction (illum-corr)
-------------------------------------

When using
`BaSiCPy <https://basicpy.readthedocs.io/en/latest/index.html>`__ to do
illumination correction, the output will include a directory,
``model`` of models, with one subdirectory per channel:

.. code:: bash

   model/
   ├── c0
   │   ├── profiles.npy
   │   └── settings.json
   ├── c1
   │   ├── profiles.npy
   │   └── settings.json
   └── c2
       ├── profiles.npy
       └── settings.json

The ``profiles.npy`` files contain the models store in numpy binary,
while the json files contain the settings for the correction.

Additionally, it will generate one or two tiff files with the flatfield
and, optionally, the darkfield.

If the ``--plot-fit`` is used, a multipage pdf would be generated
following the training of the model.

Dialout analysis (dialout)
--------------------------

Outputs:
Reads per spacer_20mer per pool. Example:

+----------------------+--------------+-----------------------+--------------+-----------------------+--------------+-----------------------+---------------------+----------------------+-------------+---------+------------+---------------+
| sequence             | count_T2-A06 | count_fraction_T2-A06 | count_T2-A04 | count_fraction_T2-A04 | count_T2-A05 | count_fraction_T2-A05 | ID                  | gene_id              | gene_symbol | dialout | mismatches | closest_match |
+======================+==============+=======================+==============+=======================+==============+=======================+=====================+======================+=============+=========+============+===============+
| ATTCACAGTGCTGGTCCCAA | 1138.0       | 0.0017697628704371    | 592.0        | 0.0015467540373676    | 1035.0       | 0.001774706573273     | ENSG00000170142_61  | ENSG00000170142      | UBE2E1      | 5.0     | 0          |               |
| GGAGTCCTCGGAGAGCAGGA | 1125.0       | 0.001749545895643     | 477.0        | 0.0012462866145682    | 849.0        | 0.0014557737977863    | ENSG00000263001_74  | ENSG00000263001      | GTF2I       | 5.0     | 0          |               |
| TATGCTTGTAAACACCTTGG | 1067.0       | 0.0016593470850232    | 607.0        | 0.0015859454403415    | 857.0        | 0.0014694913365169    | ENSG00000165699_523 | ENSG00000165699      | TSC1        | 5.0     | 0          |               |
| AAACTCCCTCATCCGCCCGA | 496.0        | 0.0007713553459901    | 257.0        | 0.0006714793709518    | 354.0        | 0.0006070010888296    | 1                   | AAACTCCCTCAGCCGCCCGA |             |         |            |               |
| GTTGCCCTCGAGGTCAATGT | 496.0        | 0.0007713553459901    | 378.0        | 0.0009876233549408    | 500.0        | 0.0008573461706633    | ENSG00000130725_52  | ENSG00000130725      | UBE2M       | 5.0     | 0          |               |
+----------------------+--------------+-----------------------+--------------+-----------------------+--------------+-----------------------+---------------------+----------------------+-------------+---------+------------+---------------+


Summarized stats per pool. Example:

+--------+----------+--------------------+--------------------+--------------------+-----------------------+-------------+
| index  | n_mapped | fraction_mapped    | average_read_count | skew_ratio         | drop_out_ratio        | n_drop_outs |
+========+==========+====================+====================+====================+=======================+=============+
| T2-A06 | 643024   | 0.6599747925724300 | 392.9778761061950  | 3.115920398009950  | 0.0                   | 0           |
| T2-A04 | 382737   | 0.66117268637843   | 235.98227474150700 | 3.187183811129850  | 0.0014749262536873200 | 1           |
| T2-A05 | 583195   | 0.6565736737818610 | 354.9631268436580  | 3.104017611447440  | 0.0                   | 0           |
+--------+----------+--------------------+--------------------+--------------------+-----------------------+-------------+
