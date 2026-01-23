********************
Workflow Reference
********************

Scallops provides two primary end-to-end pipelines written in **WDL 1.0** (Workflow Description Language). These workflows are designed for scalability and reproducibility across various environments, including local machines, cloud infrastructure, and high-performance computing (HPC) clusters.

.. contents:: Table of Contents
   :local:
   :depth: 2

--------------------------------------------------------------------------------

Stitching Workflow
==================
**File:** ``stitch_workflow.wdl``

This workflow performs illumination correction (flatfield estimation) followed by image stitching. It takes raw microscopy images (e.g., `.nd2`, `.tiff`) and converts them into OME-Zarr format.

Workflow Steps
--------------

1.  **Grouping**: The workflow scans the input directories (`urls`) using the `image_pattern`. It groups images based on the `groupby` parameter (default: plate, well, timepoint).
2.  **Illumination Correction**: (Optional) For each group, it calculates a flatfield image (mean or median projection). This step is parallelized across groups.
3.  **Stitching**:

    * Applies the calculated flatfield to the raw tiles.
    * Corrects for radial distortion.
    * Aligns tiles using stage positions and cross-correlation.
    * Stitches tiles into an OME-Zarr image.

Inputs
------

Minimal Configuration (Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the absolute minimum parameters required to run the stitching workflow.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **urls**
     - Array[String]
     - List of directories containing raw images (e.g S3 URLs).
   * - **image_pattern**
     - String
     - Regex-like pattern to parse filenames (e.g., ``"Well{well}_Point{point}.nd2"``).
   * - **output_directory**
     - String
     - Base path for outputs.
   * - **docker**
     - String
     - Workflow docker image.

.. code-block:: json
   :caption: Minimal Stitching JSON

   {
      "urls": ["s3://your-bucket/experiment_data/"],
      "image_pattern": "20231010_10x_6W_SBS_c{t}/plate{plate}/Well{well}_Point{skip}_{skip}_Channel{skip}_Seq{skip}.nd2",
      "output_directory": "s3://your-bucket/experiment_data/stitch/iss/",
      "docker":"772311241819.dkr.ecr.us-west-2.amazonaws.com/scallops:1.0.0"
   }

Full Parameter Reference (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the complete list of exposed options, including optional settings for grouping, distortion correction, and resource allocation.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **groupby**
     - Array[String]
     - Metadata keys to group tiles by. Default: ``["plate", "well", "t"]``.
   * - **subset**
     - Array[String]
     - Filter to process only specific groups (e.g., ``["A-1", "B-2"]``).
   * - **z_index**
     - String
     - Specific Z-plane to stitch, or ``"focus"`` for auto-focus.
   * - **stitch_channel**
     - Int
     - The reference channel index used for calculating stitching offsets. Default: ``0``.
   * - **stitch_radial_correction_k**
     - String
     - Coefficient for barrel distortion correction.
   * - **stitch_max_shift**
     - Float
     - Maximum allowed shift between tiles.
   * - **stitch_blend**
     - String
     - Blending method for overlapping regions.
   * - **stitch_crop**
     - Int
     - Pixels to crop from edges before stitching.
   * - **stitch_min_overlap_fraction**
     - Float
     - Minimum overlap required between tiles.
   * - **run_illumination_correction**
     - Boolean
     - Default ``true``. Set to ``false`` if images are pre-corrected.
   * - **illumination_agg_method**
     - String
     - Method for flatfield calculation. Default: ``"mean"``.
   * - **expected_images**
     - Int
     - Expected number of images per group (useful for QC).
   * - **rename**
     - String
     - Path to a 2-column CSV mapping image IDs to new IDs.
   * - **force_stitch**
     - Boolean
     - Force re-run of stitching even if output exists.
   * - **Resources**
     - Various
     - ``stitch_cpu``, ``stitch_memory``, etc. can be set to override defaults.

Outputs
-------

The workflow generates the following directory structure in `output_directory`:

* ``illumination_correction/``: Contains calculated flatfield (and optionally darkfield) images in TIFF format.
* ``stitch/``: Contains the stitched images in OME-Zarr format.

--------------------------------------------------------------------------------

OPS Workflow
============
**File:** ``ops_workflow.wdl``

The Optical Pooled Screens (OPS) workflow is a comprehensive pipeline that integrates Phenotypic imaging (IF) with In-Situ Sequencing (ISS).

Workflow Steps
--------------

**Phase 1: Phenotype Pre-processing**
    1.  **Registration**: Aligns multiple phenotypic rounds (if applicable) to a reference timepoint (e.g., "IF").
    2.  **Segmentation**: Segments Nuclei and Cells using the registered images.
    3.  **Object Discovery**: Creates labeled object maps for Nuclei, Cells, and Cytosol.

**Phase 2: ISS Pre-processing**
    1.  **Registration**: Aligns the ISS anchor round (t0) to the rest of cycles to prepare the coordinate space.

**Phase 3: Integration & Analysis**
    1.  **Cross-Modality Registration**: Aligns the Phenotype images to the ISS coordinate space.
    2.  **Feature Extraction**: Calculates morphological and intensity features for Nuclei, Cells, and Cytosol.
    3.  **Sequencing Analysis**: Detects spots in ISS channels and decodes the sequence (read calling).
    4.  **Merge**: Combines phenotypic features with decoded barcodes into a single dataset.

Inputs
------

Minimal Configuration (Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the absolute minimum parameters required to run the OPS workflow.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **output_directory**
     - String
     - Base path for outputs.
   * - **iss_url**
     - String
     - Path to stitched ISS Zarr (Required if running ISS analysis).
   * - **phenotype_url**
     - String
     - Path to stitched Phenotype Zarr (Required if running Phenotype analysis).
   * - **phenotype_dapi_channel**
     - Int
     - Channel index for DAPI in phenotype images.
   * - **phenotype_cyto_channel**
     - Array[Int]
     - Channel indices for Cytoplasm segmentation.
   * - **reads_labels**
     - String
     - Which segmentation label to assign reads to (e.g., ``"cell"`` or ``"nuclei"``).
   * - **docker**
     - String
     - Workflow docker image.

.. code-block:: json
   :caption: Minimal OPS JSON

   {
      "output_directory": "s3://your-bucket/experiment/ops_results/",
      "iss_url": "s3://your-bucket/experiment/stitch/iss/stitch/stitch.zarr/",
      "phenotype_url": "s3://your-bucket/experiment/stitch/pheno/stitch/stitch.zarr/",
      "phenotype_dapi_channel": 4,
      "phenotype_cyto_channel": [6],
      "reads_labels": "cell",
      "docker":"772311241819.dkr.ecr.us-west-2.amazonaws.com/scallops:1.0.0"
   }

Full Parameter Reference (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the complete list of exposed options covering registration, feature extraction, spot detection, and library configuration.

**Data & Grouping**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **iss_image_pattern**
     - String
     - Default: ``"{plate}-{well}-{t}"``.
   * - **phenotype_image_pattern**
     - String
     - Default: ``"{plate}-{well}-{t}"``.
   * - **groupby**
     - Array[String]
     - Default: ``["plate", "well"]``.
   * - **subset**
     - Array[String]
     - Filter specific wells/plates.

**Segmentation & Registration**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **reference_phenotype_time**
     - String
     - Timepoint to use as reference (e.g., ``"IF"``).
   * - **phenotype_dapi_channel_before_registration**
     - Int
     - DAPI index before registration (for pheno-pheno alignment).
   * - **iss_dapi_channel**
     - Int
     - DAPI index in ISS images.
   * - **nuclei_segmentation**
     - String
     - Method (e.g., ``"stardist"``, ``"cellpose"``).
   * - **cell_segmentation_method**
     - String
     - Method (e.g., ``"watershed"``).
   * - **cell_segmentation_extra_arguments**
     - String
     - Extra flags (e.g., ``"--closing-radius 5"``).
   * - **register_across_channels**
     - Boolean
     - Enable cross-channel registration logic.

**Feature Extraction**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **phenotype_nuclei_features**
     - Array[String]
     - List of features (e.g., ``["intensity_*"]``).
   * - **phenotype_cell_features**
     - Array[String]
     - List of features.
   * - **phenotype_cytosol_features**
     - Array[String]
     - List of features.
   * - **features_cell_min_area**
     - Int
     - Minimum area filter for cells.
   * - **features_nuclei_min_area**
     - Int
     - Minimum area filter for nuclei.

**Sequencing (ISS)**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **barcodes**
     - String
     - Path to CSV containing the library design.
   * - **barcode_column**
     - String
     - Column name in the barcode CSV.
   * - **iss_expected_cycles**
     - Int
     - Number of sequencing cycles.
   * - **iss_channels**
     - Array[Int]
     - Channels to use for spot detection. Default: ``[1,2,3,4]``.
   * - **reads_bases**
     - String
     - Bases order (e.g., ``"GTAC"``).
   * - **spot_detection_sigma_log**
     - Array[Float]
     - Sigma for Laplacian of Gaussian spot detection.

**Additional Parameters**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - **model_dir**
     - String
     - Path containing deep learning model resouces (See :doc:`FAQ <faq>` for more details.)
   * - **run_``task``**
     - Boolean
     - Set to ``false``, (e.g. run_nuclei_segmentation) to skip task
   * - **force_``task``**
     - Boolean
     - Set to ``true``, to re-run task (e.g. force_segment_cell) even if output exists.
   * - **Resources**
     - Various
     - ``segment_nuclei_cpu``, ``segment_nuclei_memory``, etc. can be set to override defaults.
   * - **batch_size**
     - Int
     - Number of groups to process in one batch.
Outputs
-------

The `output_directory` will contain subdirectories for every major step:

* ``segment.zarr``: Nuclei and Cell labels.
* ``pheno-to-iss-registered.zarr``: Phenotype images transformed to align with ISS.
* ``features-nuclei-<index>/``, ``features-cell-<index>/``, ``features-cytosol-<index>/``: Parquet files containing calculated features. The ``<index>`` refers to different splits of the data that had been run in parallel.
* ``spot-detect.zarr``: Raw spot locations.
* ``reads/``: Decoded reads per cell.
* ``merge/``: **Final Output.** A merged Parquet dataset linking Cell IDs, Barcodes, and Phenotypic Features.

--------------------------------------------------------------------------------

Running on AWS HealthOmics
==========================

AWS HealthOmics provides a managed service for running bioinformatics workflows at scale. Scallops workflows (WDL) are fully compatible with HealthOmics. We recommend using the `miniwdl-omics-run` tool to simplify the submission process.

Prerequisites
-------------

1.  **S3 Buckets:** You must have S3 buckets for inputs (images) and outputs.
2.  **IAM Role:** An IAM role with permissions to read/write to your S3 buckets and execution permissions for HealthOmics.
3.  **Docker Images:** The Scallops Docker image must be in Amazon ECR (Elastic Container Registry).

Step 1: Configure Input JSON
----------------------------
Create a JSON file (e.g., ``ops_input.json``) defining your inputs. Below is a minimal example for the **OPS Workflow**.

**Note:** Ensure all S3 paths end with a trailing slash ``/`` if they refer to directories.

.. code-block:: json

    {
      "iss_url": "s3://your-bucket/experiment/ISS/stitch.zarr/",
      "iss_image_pattern": "{plate}-{well}-{t}",
      "phenotype_url": "s3://your-bucket/experiment/Pheno/stitch.zarr/",
      "phenotype_image_pattern": "{plate}-{well}-{t}",

      "subset": ["A-1", "A-2"],
      "groupby": ["plate", "well"],
      "output_directory": "s3://your-output-bucket/results/experiment_name/",

      "reference_phenotype_time": "IF",
      "phenotype_dapi_channel": 4,
      "phenotype_cyto_channel": [6],

      "phenotype_nuclei_features": ["intensity_*", "sizeshape", "colocalization_*_*", "spots_1,2,3"],
      "phenotype_cell_features": ["intensity_*", "sizeshape", "colocalization_*_*", "spots_1,2,3"],
      "phenotype_cytosol_features": ["intensity_*", "sizeshape", "colocalization_*_*", "spots_1,2,3"],

      "barcodes": "s3://your-bucket/library/barcodes.csv",
      "barcode_column": "opsBarcode",
      "reads_labels": "cell",
      "iss_expected_cycles": 7,
      "reads_bases": "GTAC",

      "segment_cell_threshold_correction_factor": 1.0,
      "cell_segmentation_extra_arguments": "--closing-radius 5",

      "docker": "123456789012.dkr.ecr.us-region-1.amazonaws.com/scallops:latest"
    }

Step 2: Run with miniwdl-omics-run
----------------------------------
Use the `miniwdl-omics-run` utility to submit the workflow. This tool zips your local WDL files, uploads them to S3, and triggers the HealthOmics run.

.. code-block:: bash

   miniwdl-omics-run \
     scallops/wdl/ops_workflow.wdl \
     -i ops_input.json \
     --role-arn arn:aws:iam::123456789012:role/YourHealthOmicsWorkflowRole \
     --output-uri s3://your-output-bucket/omics-logs/ \
     --name "OPS_Experiment_Run_01"

Arguments Explained:
^^^^^^^^^^^^^^^^^^^^
* **Workflow File**: Points to the local main WDL file (e.g., ``scallops/wdl/ops_workflow.wdl``). It will automatically bundle dependencies like ``ops_tasks.wdl``.
* **-i**: The input JSON file you created in Step 1.
* **--role-arn**: The AWS IAM role ARN that HealthOmics assumes to access S3 and CloudWatch.
* **--output-uri**: The S3 location where HealthOmics will store execution logs (different from the workflow `output_directory`).
* **--name**: A custom name for the run to identify it in the AWS Console.

--------------------------------------------------------------------------------

Customizing Workflows
=====================

Scallops' WDL architecture is modular. Key computational steps (such as stitching, registration, and segmentation) are defined as independent **Tasks** in files like ``ops_tasks.wdl`` and ``stitch_tasks.wdl``. This design allows you to construct your own custom workflows by importing these tasks, rather than relying solely on the pre-built end-to-end pipelines.

You can mix and match Scallops tasks with your own custom tasks (e.g., for QC or specific file conversions) to create tailored analysis solutions.

Example: Building a Custom Registration Workflow
------------------------------------------------

Suppose you only need to perform image registration without the full segmentation or sequencing analysis. You can create a simple WDL file that imports the Scallops tasks and calls only the registration step.

1.  **Create a new WDL file** (e.g., ``my_registration.wdl``).
2.  **Import the Scallops tasks** file.
3.  **Define a workflow** that calls the specific task.

.. code-block:: text

    version 1.0

    # Import the existing Scallops tasks
    import "scallops/wdl/ops_tasks.wdl" as tasks

    workflow my_custom_registration {
        input {
            String moving_image
            String fixed_image
            String output_dir
            String docker
        }

        # Call the existing Scallops registration task
        call tasks.register_elastix {
            input:
                moving = [moving_image],
                fixed = fixed_image,
                transform_output_directory = output_dir + "/transforms",
                moving_output_directory = output_dir + "/registered_images",
                # Pass through required runtime parameters
                docker = docker,
                cpu = 4,
                memory = "16 GiB",
                # ... (other required inputs like zones, disks, etc.)
        }
    }

Modifying Existing Tasks
------------------------

If the pre-built tasks do not perfectly fit your needs (e.g., you need to change the resource allocation or add a specific command-line flag not currently exposed), you can modify the task definitions directly:

1.  Copy the relevant task file (e.g., ``ops_tasks.wdl``) to your local directory.
2.  Edit the ``runtime`` block to adjust memory/CPU, or the ``command`` block to add new flags.
3.  Point your workflow to import your modified task file instead of the standard one.

.. code-block:: text

    # In your workflow file
    import "my_modified_tasks.wdl" as tasks
