***************************
Perturbation Map Building
***************************

SCALLOPS provides a step-by-step pipeline for building perturbation maps from
single-cell morphological feature profiles.  The pipeline is designed to be run
one step at a time, making it straightforward to wrap in a WDL workflow where
each task calls one ``scallops map-*`` command.

Every step reads and writes **AnnData Zarr** (``.zarr``) files — specifically
the `AnnData <https://anndata.readthedocs.io/>`_ format serialised with Zarr
v2 storage.  These files preserve ``obs`` (cell metadata), ``var`` (feature
names), ``uns`` (unstructured metadata including backprojection parameters),
and ``varm`` (per-variable arrays such as PCA loadings) through the entire
pipeline.  They can be read back with:

.. code-block:: python

    from scallops.io import read_anndata_zarr
    data = read_anndata_zarr("step_output.zarr", dask=True)

.. contents:: Pipeline overview
   :local:
   :depth: 2


Pipeline steps
==============

The recommended order is shown below.  The shape annotation on the right of
each arrow shows how the data dimensionality changes through the pipeline —
this is critical for understanding the memory cost of each step (see
:ref:`memory-requirements`).

.. code-block:: text

    Input (AnnData Zarr / Parquet from pooled-sbs merge)
      │                   shape: N × p_raw   (e.g. 10M × 10 000)
      ▼
    map-filter          ← remove low-variance, sparse, categorical, and batch-
      │                   correlated features; filter high-nan cells
      │                   shape: N × p       (e.g. 10M × 5 000)
      ▼
    map-transform-yj    ← Yeo-Johnson power transform (optional)
      │                   shape: N × p       (unchanged)
      ▼
    norm-features       ← well-level z-score  (--by plate well)
      │                   shape: N × p       (unchanged)
      ▼
    map-pca             ← *** DIMENSIONALITY REDUCTION ***
      │                   fit on NTC subset, project ALL cells
      │                   shape: N × K       (e.g. 10M × 128 PCs)
      │
      ├─ map-pca-select ← retain statistically significant PCs
      │                   shape: N × K'      (K' ≤ K)
      │
      └─ map-sphere     ← ZCA whitening (optional, between PCA and TVN)
                          shape: N × K'      (unchanged)
      ▼
    map-tvn             ← Typical Variation Normalization
      │                   input is already N × K', NOT N × p  ← this is why TVN is cheap
      │                   stores PCA + covariance-alignment parameters for backprojection
      │                   shape: N × K'      (unchanged)
      ▼
    norm-features       ← normalize to NTC reference (optional)
      │                   shape: N × K'      (unchanged)
      ▼
    map-agg             ← aggregate cells → perturbation profiles
      │                   shape: n_pert × K' (e.g. 5 000 × 128)  ← tiny!
      ▼
    map-center          ← subtract NTC mean (optional, before similarity)
      │                   shape: n_pert × K'
      ▼
    map-similarity      ← pairwise cosine / Pearson similarity
      │                   shape: n_pert × n_pert  (e.g. 5 000 × 5 000)  ← tiny!
      ▼
    map-cluster         ← cluster perturbations, reorder similarity matrix
      │                   shape: n_pert × n_pert  (reordered)
      ▼
    map-recall          ← Parquet recall metrics + optional AnnData injection


.. _memory-requirements:

Memory requirements and scaling
================================

Understanding how data dimensionality changes through the pipeline is essential
for estimating RAM requirements at production scale.

The key insight
---------------

**TVN operates on PCA-reduced data, not on raw features.**

``map-pca`` reduces N × p (e.g. 10M × 5 000) to N × K PCs (e.g. 10M × 128).
Everything from ``map-sphere`` onward — including ``map-tvn`` — sees the smaller
N × K representation.  This makes TVN much cheaper than it looks.

.. list-table:: Analytical RAM estimates at production scale (10M cells)
   :header-rows: 1
   :widths: 28 14 14 44

   * - Step
     - Shape
     - Peak RAM
     - Mode and formula
   * - **map-filter**
     - N × p_raw
     - **8 GB** / worker
     - Dask chunk-bounded.  ``chunk × p_raw × 4 B = 200K × 10K × 4``
   * - **map-transform-yj**
     - N × p
     - **8 GB** / worker
     - Dask chunk-bounded.  ``chunk × p × 8 B`` (float64 PowerTransformer)
   * - **norm-features** (scale)
     - N × p
     - **4 GB** / worker
     - Dask chunk-bounded.  ``chunk × p × 4 B``
   * - **map-pca** (fit, incremental)
     - NTC × p
     - **8 GB**
     - One batch in RAM at a time.  ``batch × p × 8 B = 200K × 5K × 8``
   * - **map-pca** (transform, chunked) ← peak
     - N × p → N × K
     - **9–10 GB**
     - Chunk input + full output.  ``(batch × p × 4) + (N × K × 4)``
       = ``(200K × 5K × 4) + (10M × 128 × 4)`` = 4 + 5.1 GB
   * - **map-sphere**
     - N × K
     - **10 GB**
     - Materialises all cells for SVD.  ``N × K × 8 B`` (float64)
   * - **map-tvn** ← surprisingly cheap!
     - N × K
     - **5.6 GB**
     - Materialises all cells + NTC for internal PCA.
       ``(N × K × 4) + (n_ntc × K × 8)``
       = ``(10M × 128 × 4) + (500K × 128 × 8)`` = 5.1 + 0.5 GB
   * - **map-agg → map-recall**
     - n_pert × K
     - **< 1 GB**
     - All perturbation-scale; independent of N

.. note::

   These estimates assume a PCA batch size of 200 000, 128 PCs, and a dask chunk
   size of 200 000 rows.  The ``map-pca`` transform peak (9–10 GB) is the
   practical bottleneck, not TVN.  All dask-bounded steps are per-worker; the
   total cluster RAM is ``n_workers × per_worker_peak``.

Why TVN is not the bottleneck
------------------------------

A common misconception is that ``map-tvn`` must materialise the entire N × p
feature matrix.  This is only true if TVN is run *without* a preceding
``map-pca`` step.

When the standard pipeline order is followed (``map-pca`` → ``map-pca-select``
→ ``map-sphere`` → ``map-tvn``), TVN's input is already the dimensionality-
reduced N × K representation.  For 10M cells and 128 PCs:

.. code-block:: text

   Without map-pca:   map-tvn sees  10M × 5 000 × float32  =  200 GB  ← infeasible
   With map-pca:      map-tvn sees  10M × 128   × float32  =    5 GB  ← manageable

TVN's internal PCA fitting is done on the NTC subset (500K × 128 × float64 = 0.5 GB),
and the covariance matrices are K × K = 128 × 128 = negligible.  The only step
that materialises the full N × K array is the matrix multiply that applies the
PCA transform and covariance alignment.

The real bottleneck: map-pca transform
---------------------------------------

``map-pca`` is the step where all N cells must be projected from feature space
(p columns) into PC space (K columns).  Scallops performs this in chunks
(controlled by ``--batch-size``) to avoid materialising the full N × p matrix,
but the *output* N × K must still be written into a contiguous array.

For 10M cells at 128 PCs:

.. code-block:: text

   Per-chunk cost (transient):  200K × 5 000 × float32 =  4.0 GB
   Output array (persistent):   10M × 128   × float32  =  5.1 GB
   Total peak:                  ≈ 9.1 GB

This output is then written to the AnnData Zarr so subsequent steps (sphere,
TVN, agg) read only the smaller N × K representation.

Practical recommendations
--------------------------

1. **Run dask-bounded steps on a distributed cluster.**
   Filter, YJ transform, and z-score normalisation each require roughly
   ``n_workers × chunk × p_raw × 4 bytes`` total cluster RAM.
   For 8 workers, 200K chunk, 10K features: 8 × 8 GB = 64 GB cluster RAM.

2. **Use ``--batch-size`` for map-pca fit.**
   Incremental PCA reads one batch of NTC cells at a time.  Each batch
   costs ``batch × p × 8 bytes`` (float64 for sklearn).
   At 200K batch × 5K features: 8 GB per batch.

3. **Let scallops chunk the map-pca transform.**
   Scallops already performs the projection in batches — the only
   unavoidable memory cost is the N × K output array (≈ 5 GB for 10M × 128).
   Consider reducing K (via ``--cluster-max-n-clusters`` in ``map-pca-select``)
   if contiguous RAM is limited.

4. **map-tvn through map-recall fit in a single moderately sized machine.**
   Once the data is in PC space, no step requires more than ~10 GB of RAM
   regardless of N.  These steps can be run on a workstation.

5. **Estimate requirements for your own data.**
   Run the memory profiler script on a small proxy dataset and use the
   analytical extrapolation table::

       python scallops/tests/memory_profile.py \
           --n-rows 30000 --n-features 300 --n-pcs 32 \
           --target-rows 10000000 \
           --target-features-raw 10000 \
           --target-features-filtered 5000 \
           --target-n-pcs 128

   The script reports measured tracemalloc peaks for the proxy (lower bound,
   ~50–70% of true RSS) and exact analytical estimates for the target scale.

Accuracy of the analytical estimates
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Step type
     - Accuracy
     - Notes
   * - Dask chunk-bounded (filter, YJ, scale, agg)
     - ±20 %
     - Formula ``batch × p × dtype_bytes`` is exact; tracemalloc underestimates
       by 30–50 % (misses C-level numpy allocations and dask worker buffers).
   * - Materialising (pca-transform, sphere, TVN)
     - ±5 %
     - Formula is deterministic; only overhead is Python object headers and
       allocator padding.
   * - Perturbation-scale (similarity, cluster, recall)
     - ±2 %
     - Matrix dimensions are known exactly before any computation.


Provenance tracking
===================

Every ``map-*`` command appends its metadata to a JSON list stored in
``uns["scallops"]`` of the output **AnnData Zarr**.  After *N* steps the chain
has *N* entries.  Read it back with::

    import json
    chain = json.loads(data.uns["scallops"])   # list of dicts, one per step


Backprojection parameters
=========================

``map-tvn`` stores the following keys in ``uns`` and ``varm`` of its output
**AnnData Zarr** so that any downstream step (including ``map-agg``,
``map-center``, and ``map-similarity``) can project profiles back to the
original z-score feature space.  All ``map-*`` steps propagate these keys
automatically, so you can call
:func:`~scallops.features.backprojection.top_features_from_backprojection`
on any downstream AnnData Zarr in the pipeline.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Contents
   * - ``uns["pca"]["PCs"]``
     - PCA components, shape ``(n_pcs, n_features)``
   * - ``uns["pca"]["mean"]``
     - PCA training mean, shape ``(n_features,)``
   * - ``uns["tvn_pre_scale_mean"]``
     - Reference mean before z-scoring, shape ``(n_features,)``
   * - ``uns["tvn_pre_scale_std"]``
     - Reference std before z-scoring, shape ``(n_features,)``
   * - ``uns["covariance_alignment_inv"]``
     - Dict mapping group key → inverse alignment matrix ``(n_pcs, n_pcs)``; only set when ``--by`` is used
   * - ``uns["normalization_arguments"]``
     - ``{"reference_query": ..., "by": ...}`` for audit purposes
   * - ``varm["PCs"]``
     - Transposed PCA components, shape ``(n_features, n_pcs)``

All ``map-*`` steps forward these keys through the pipeline.


Backprojecting to original features
=====================================

After running the pipeline, use
:func:`~scallops.features.backprojection.top_features_from_backprojection` to
find which original features best explain a cluster or perturbation set.  Load
any downstream **AnnData Zarr** that preserved ``uns`` (``tvn.zarr``,
``agg.zarr``, ``centered.zarr``, etc.):

.. code-block:: python

    from scallops.io import read_anndata_zarr
    from scallops.features.backprojection import (
        backproject_tvn,
        top_features_from_backprojection,
    )

    # Any AnnData Zarr that passed through map-tvn preserves the backprojection params
    data = read_anndata_zarr("agg.zarr")   # or tvn.zarr, centered.zarr, …

    # Full backprojection: TVN space → z-score space
    X_zscored = backproject_tvn(data)            # shape (n_obs, n_features)
    X_original = backproject_tvn(data, to_original_scale=True)

    # Top features discriminating gene_A from everything else
    result = top_features_from_backprojection(
        data,
        genes=["gene_A"],
        top_k=20,
    )
    print(result.head())
    #        feature     score  pvalue
    # 0  Cells_Intensity_f3  0.412     NaN
    # ...

    # Compare gene_A against gene_B only (specific reference)
    result_ab = top_features_from_backprojection(
        data,
        genes=["gene_A"],
        genes_ref=["gene_B"],
        top_k=20,
    )

    # With PC-level statistical filter (orthogonal, avoids correlated-feature issues)
    result_filt = top_features_from_backprojection(
        data,
        genes=["gene_A"],
        pc_stat_filter="ttest",
        pc_pvalue_threshold=0.05,
        top_k=20,
    )

    # Using cluster assignments from UMAP / k-means
    import numpy as np
    cluster_labels = np.array([...])    # one entry per obs
    result_clust = top_features_from_backprojection(
        data,
        cluster_labels=cluster_labels,
        cluster_query=2,               # cluster 2 vs all others
        cluster_ref=[0, 1],            # or: compare against clusters 0 and 1 specifically
        top_k=20,
    )

Feature importance is derived from the **centroid difference in PCA/TVN space**
(an orthogonal basis), which is projected back to z-score feature space.  This
avoids direct testing of correlated features.  The optional ``pc_stat_filter``
applies a t-test or Mann-Whitney U test to each *PC dimension* (which are
orthogonal by construction) before projecting, giving a principled way to
suppress noise components.


Example WDL workflow
=====================

Each pipeline step maps naturally to a WDL task.  The input and output are
always **AnnData Zarr** directories.  The workflow below follows the
recommended order, including the dimensionality-reducing ``map-pca`` step
before ``map-tvn`` so that TVN sees N × K PCs rather than N × p features.

.. code-block:: wdl

    workflow map_build {
        input {
            File   merged_zarr                  # from pooled-sbs merge
            String ntc_query = "gene_symbol=='NTC'"
            Int    pca_components = 128
            Int    pca_batch_size = 200000      # incremental PCA: 200K × p × 8B per batch
        }

        # ── Cell-level steps (dask, chunk-bounded) ─────────────────────────
        call map_filter   { input: zarr = merged_zarr }
        call map_transform_yj { input: zarr = map_filter.out }
        call norm_features_scale { input: zarr = map_transform_yj.out }

        # ── Dimensionality reduction: 10M × 5K → 10M × 128 ────────────────
        # After this point everything is cheap because shape = N × K, not N × p
        call map_pca {
            input:
                zarr = norm_features_scale.out,
                n_pcs = pca_components,
                batch_size = pca_batch_size,
                reference_query = ntc_query
        }
        call map_pca_select { input: zarr = map_pca.out }
        call map_sphere    { input: zarr = map_pca_select.out }

        # ── TVN: cheap because input is N × 128, not N × 5 000 ───────────
        call map_tvn {
            input:
                zarr = map_sphere.out,
                reference_query = ntc_query
        }

        # ── Profile-level steps (tiny: n_pert × 128) ──────────────────────
        call map_agg       { input: zarr = map_tvn.out }
        call map_center    { input: zarr = map_agg.out, reference_query = ntc_query }
        call map_similarity { input: zarr = map_center.out }
        call map_cluster   { input: zarr = map_similarity.out }
        call map_recall    { input: zarr = map_cluster.out }
    }

    # ── Representative task bodies ──────────────────────────────────────────

    task map_pca {
        input {
            File   zarr
            Int    n_pcs       = 128
            Int    batch_size  = 200000
            String reference_query = "gene_symbol=='NTC'"
        }
        command <<<
            scallops map-pca \
                --input ~{zarr} \
                --output pca.zarr \
                --components ~{n_pcs} \
                --batch-size ~{batch_size} \
                --reference-query "~{reference_query}"
        >>>
        # Peak RAM during this task:
        #   fit  : batch_size × n_features × 8 B  (incremental PCA)
        #   transform: (batch_size × n_features × 4 B) + (N × n_pcs × 4 B)
        output { File out = "pca.zarr" }
    }

    task map_tvn {
        input {
            File   zarr
            String reference_query = "gene_symbol=='NTC'"
            String? by
        }
        command <<<
            scallops map-tvn \
                --input ~{zarr} \
                --output tvn.zarr \
                --reference-query "~{reference_query}" \
                ~{if defined(by) then "--by " + by else ""}
        >>>
        # Peak RAM: N × n_pcs × 4 B  (e.g. 10M × 128 × 4 = 5 GB — not 200 GB!)
        # TVN is cheap because map-pca already reduced N × p → N × K.
        output { File out = "tvn.zarr" }
    }

    task map_recall {
        input {
            File   zarr
            File   corum_file
            File?  gmt_file
            File?  string_file
            String inject_zarr = "similarity_with_recall.zarr"
        }
        command <<<
            scallops map-recall \
                --input ~{zarr} \
                --output recall.parquet \
                --corum ~{corum_file} \
                ~{if defined(gmt_file)    then "--gmt "    + gmt_file    else ""} \
                ~{if defined(string_file) then "--string " + string_file else ""} \
                --inject-zarr ~{inject_zarr} \
                --min-genes 5 --min-pairs 10
        >>>
        output {
            File parquet = "recall.parquet"
            File zarr_with_recall = inject_zarr
        }
    }


API reference
=============

Backprojection
--------------

.. autosummary::
   :toctree: .

   scallops.features.backprojection.backproject_tvn
   scallops.features.backprojection.top_features_from_backprojection

Normalization & decomposition
------------------------------

.. autosummary::
   :toctree: .

   scallops.features.normalize.typical_variation_normalization
   scallops.features.preprocessing.filter_data
   scallops.features.preprocessing.remove_correlated_features
   scallops.features.preprocessing.filter_zero_inflated
   scallops.features.preprocessing.filter_low_cardinality
   scallops.features.preprocessing.filter_batch_correlated
   scallops.features.preprocessing.transform_features_yj
   scallops.features.decomposition.pca
   scallops.features.decomposition.sphere
   scallops.features.decomposition.select_pca_components
   scallops.features.decomposition.largest_variance_from_random_matrix
   scallops.features.agg.agg_features

Clustering
----------

.. autosummary::
   :toctree: .

   scallops.features.map_cluster.cluster_similarity

Recall evaluation
-----------------

.. autosummary::
   :toctree: .

   scallops.features.map_eval.pairwise_similarities
   scallops.features.map_eval.set_benchmark
   scallops.features.map_eval.pairwise_benchmark
   scallops.features.map_eval.recall
   scallops.features.map_eval.read_corum
   scallops.features.map_eval.read_gmt
   scallops.features.map_eval.read_string
   scallops.features.map_eval.fetch_string
   scallops.features.map_eval.read_reactome_fi
   scallops.features.map_eval.gmt_to_gene_sets
