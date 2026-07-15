import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    no_version_arg,
)

def _reference_query_arg(
    parser: argparse.ArgumentParser,
    default: str | None = None,
    help_suffix: str = "",
) -> None:
    """Add ``--reference-query`` to *parser*.

    :param parser: Parser to add the argument to.
    :param default: Default query string (e.g. ``"gene_symbol=='NTC'"``).
    :param help_suffix: Extra text appended to the help string.
    """
    parser.add_argument(
        "--reference-query",
        help=(
            "Pandas query expression that selects the reference "
            "(negative-control) observations used to fit this step.  "
            "Examples: \"gene_symbol=='NTC'\", \"type=='intergenic'\", "
            "\"perturbation_class=='scramble'\".  "
            "The query runs against obs column names, so any metadata "
            "column can be used as the reference selector."
            + (f"  {help_suffix}" if help_suffix else "")
        ),
        default=default,
        dest="reference_query",
    )


# ---------------------------------------------------------------------------
# map-filter
# ---------------------------------------------------------------------------


def _run_map_filter(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_filter

    run_pipeline_map_filter(arguments)


def _create_map_filter_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-filter",
        help="Filter cells and features as the first step of the map-build pipeline",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save filtered data in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--features",
        help="Features to include. If not specified all features are used.",
        nargs="*",
    )
    parser.add_argument(
        "--label-filter",
        help="Pandas query expression to filter cells before feature filtering.",
    )
    parser.add_argument(
        "--min-variance",
        help="Minimum feature variance to retain a feature. Negative values disable the threshold.",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--max-variance",
        help="Maximum feature variance to retain a feature. Negative values disable the threshold.",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--max-fraction-not-finite",
        help="Maximum fraction of non-finite values allowed per cell.",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--by",
        help="Column(s) in obs to stratify variance computation (e.g. plate well).",
        nargs="*",
    )
    parser.add_argument(
        "--filter-batch-size",
        help="Parquet streaming batch size (rows). Smaller values allow PyArrow to "
             "read more source files concurrently on high-RAM machines; larger values "
             "reduce loop overhead on memory-constrained machines.",
        type=int,
        default=100_000,
        dest="filter_batch_size",
    )

    # --- Condition column ---
    cond = parser.add_argument_group(
        "condition column (add a derived obs column from a source column → label map)"
    )
    cond.add_argument(
        "--condition-column",
        default=None,
        dest="condition_column",
        help="Name of the new obs column to create (e.g. 'condition').  "
             "When omitted no condition column is added.",
    )
    cond.add_argument(
        "--condition-source-column",
        default="well",
        dest="condition_source_column",
        help="Existing obs column whose values are looked up in --condition-map "
             "(default: 'well').",
    )
    cond.add_argument(
        "--condition-map",
        default=None,
        dest="condition_map",
        help="JSON dict mapping source-column values to condition labels "
             "(e.g. '{\"1\":\"GIRED\",\"4\":\"DMSO\"}').  "
             "When omitted the column must already exist in the input.",
    )

    # --- Correlated-feature filter ---
    corr = parser.add_argument_group("correlated-feature filter")
    corr.add_argument(
        "--max-correlation",
        help="Remove features with absolute Pearson correlation above this threshold. "
        "When omitted the filter is disabled.",
        type=float,
        default=None,
    )
    corr.add_argument(
        "--correlation-reference",
        help="Restrict the correlation estimate to cells matching this query "
        "(e.g. \"gene_symbol=='NTC'\"), avoiding biological signal.",
        default=None,
    )
    corr.add_argument(
        "--correlation-chunk-size",
        help="Column block size for the blocked correlation computation. "
        "Larger values are faster but use more memory.",
        type=int,
        default=512,
    )

    # --- Zero-inflation filter ---
    zi = parser.add_argument_group("zero-inflation filter")
    zi.add_argument(
        "--max-zero-fraction",
        help="Remove features where more than this fraction of values is near-zero. "
        "When omitted the filter is disabled.",
        type=float,
        default=None,
    )
    zi.add_argument(
        "--near-zero-threshold",
        help="Values with |v| <= this are counted as zero.",
        type=float,
        default=0.0,
    )

    # --- Low-cardinality / categorical filter ---
    cat = parser.add_argument_group("categorical filter")
    cat.add_argument(
        "--min-unique",
        help="Remove features with fewer than this many distinct finite values. "
        "Catches binary / integer-coded columns. When omitted the filter is disabled.",
        type=int,
        default=None,
    )

    # --- Batch-correlation filter ---
    batch = parser.add_argument_group("batch-correlation filter")
    batch.add_argument(
        "--batch-column",
        help="Column in obs that identifies the batch (e.g. plate). "
        "Features significantly associated with batch are removed. "
        "When omitted the filter is disabled.",
        default=None,
    )
    batch.add_argument(
        "--batch-reference",
        help="Restrict the batch-correlation test to cells matching this query "
        "(e.g. \"gene_symbol=='NTC'\"). Recommended to isolate technical variation.",
        default=None,
    )
    batch.add_argument(
        "--batch-pvalue",
        help="Significance threshold for the batch-association test.",
        type=float,
        default=0.05,
    )
    batch.add_argument(
        "--batch-method",
        help="Statistical test for batch association.",
        choices=["kruskal", "anova"],
        default="kruskal",
    )

    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_filter)


# ---------------------------------------------------------------------------
# map-transform-yj
# ---------------------------------------------------------------------------


def _run_map_transform_yj(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_transform_yj

    run_pipeline_map_transform_yj(arguments)


def _create_map_transform_yj_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-transform-yj",
        help="Apply Yeo-Johnson power transform to features",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save transformed data in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--by",
        help="Column(s) in obs to stratify the transform (e.g. plate well).",
        nargs="*",
    )
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_transform_yj)


# ---------------------------------------------------------------------------
# map-scale
# ---------------------------------------------------------------------------


def _run_map_scale(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_scale

    run_pipeline_map_scale(arguments)


def _create_map_scale_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-scale",
        help="Well-level z-score (global) or spatial k-NN z-score (local) normalisation",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save scaled data in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--scale-method",
        choices=["global", "local"],
        default="global",
        dest="scale_method",
        help="'global' for well-level z-score; 'local' for spatial k-NN z-score.",
    )
    parser.add_argument(
        "--plate-column",
        default="plate",
        dest="plate_column",
        help="obs column identifying the plate.",
    )
    parser.add_argument(
        "--well-column",
        default="well",
        dest="well_column",
        help="obs column identifying the well.",
    )
    local = parser.add_argument_group("local z-score options (--scale-method local)")
    local.add_argument(
        "--localz-neighbors",
        type=int,
        default=75,
        dest="localz_neighbors",
        help="Number of spatial nearest neighbours per cell.",
    )
    local.add_argument(
        "--localz-batch-size",
        type=int,
        default=50_000,
        dest="localz_batch_size",
        help="Cells processed per batch (caps peak memory).",
    )
    local.add_argument(
        "--localz-centroid-y",
        default="Nuclei_AreaShape_Center_Y",
        dest="localz_centroid_y",
        help="obs or var column with the y spatial coordinate.",
    )
    local.add_argument(
        "--localz-centroid-x",
        default="Nuclei_AreaShape_Center_X",
        dest="localz_centroid_x",
        help="obs or var column with the x spatial coordinate.",
    )
    local.add_argument(
        "--localz-max-value",
        type=float,
        default=5.0,
        dest="localz_max_value",
        help="Clip z-scores to ±this value after normalisation.",
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_scale)


# ---------------------------------------------------------------------------
# map-pca
# ---------------------------------------------------------------------------


def _run_map_pca(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_pca

    run_pipeline_map_pca(arguments)


def _create_map_pca_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-pca",
        help="Embed data with PCA; optionally fit on a reference subset only",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save PCA-embedded data in Zarr format",
        required=True,
    )
    _reference_query_arg(
        parser,
        help_suffix="When given, PCA is fitted on the reference subset only; "
        "all observations are then projected into the fitted space.",
    )
    parser.add_argument(
        "--components",
        help="Number of PCA components to retain.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for incremental PCA. Use 0 or negative to disable.",
        default=200_000,
        type=int,
    )
    parser.add_argument(
        "--whiten",
        help="Apply PCA whitening (divide by sqrt of explained variance).",
        action="store_true",
    )
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_pca)


# ---------------------------------------------------------------------------
# map-tvn
# ---------------------------------------------------------------------------


def _run_map_tvn(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_tvn

    run_pipeline_map_tvn(arguments)


def _create_map_tvn_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-tvn",
        help="Apply Typical Variation Normalization (TVN). "
        "Stores PCA components and covariance-alignment matrices in the output "
        "for downstream backprojection.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save TVN-normalized data in Zarr format",
        required=True,
    )
    _reference_query_arg(
        parser,
        default="gene_symbol=='NTC'",
        help_suffix="TVN is fitted on these observations (z-score, PCA, and "
        "covariance alignment all use this as the reference population).  "
        "Examples: \"gene_symbol=='NTC'\", \"type=='intergenic'\", "
        "\"perturbation_class=='scramble'\".",
    )
    parser.add_argument(
        "--by",
        help="Column(s) in obs to apply per-group covariance alignment "
        "(e.g. plate). When provided, each group is aligned to the global "
        "reference covariance.",
        nargs="*",
    )
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_tvn)


# ---------------------------------------------------------------------------
# map-agg
# ---------------------------------------------------------------------------


def _run_map_agg(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_agg

    run_pipeline_map_agg(arguments)


def _create_map_agg_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-agg",
        help="Aggregate single-cell profiles to perturbation-level profiles",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save aggregated profiles in Zarr format",
        required=True,
    )
    required.add_argument(
        "--by",
        help="Column(s) in obs to aggregate by (e.g. gene_symbol plate well).",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--perturbation",
        help="Column in obs identifying the perturbation, used for min-cells filtering.",
        default="gene_symbol",
    )
    parser.add_argument(
        "--method",
        help="Aggregation function.",
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument(
        "--min-cells",
        help="Minimum number of cells required per perturbation. "
        "Perturbations below this threshold are excluded before aggregation.",
        type=int,
    )
    parser.add_argument(
        "--barcode",
        help="Column in obs containing the guide barcode identifier, "
        "used when --agg-by-barcode is set.",
        default="barcode_0",
    )
    parser.add_argument(
        "--agg-by-barcode",
        help="Aggregate by barcode first, then by the columns in --by. "
        "Useful when multiple barcodes target the same gene.",
        action="store_true",
    )
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_agg)


# ---------------------------------------------------------------------------
# map-center
# ---------------------------------------------------------------------------


def _run_map_center(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_center

    run_pipeline_map_center(arguments)


def _create_map_center_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-center",
        help="Center profiles by subtracting the mean of a reference set "
        "(e.g. NTC controls). Typically applied after aggregation and before "
        "similarity-matrix computation.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save centered data in Zarr format",
        required=True,
    )
    _reference_query_arg(
        parser,
        default="gene_symbol=='NTC'",
        help_suffix="The mean of these profiles is subtracted from all profiles.  "
        "After centering, the reference profiles become the zero vector — exclude "
        "them in map-similarity using --exclude-reference-query.",
    )
    parser.add_argument(
        "--by",
        help="Column(s) in obs to stratify centering by groups.",
        nargs="*",
    )
    parser.add_argument(
        "--robust",
        help="Use median instead of mean for centering.",
        action="store_true",
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_center)


# ---------------------------------------------------------------------------
# map-similarity
# ---------------------------------------------------------------------------


def _run_map_similarity(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_similarity

    run_pipeline_map_similarity(arguments)


def _create_map_similarity_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-similarity",
        help="Compute pairwise similarity matrix between perturbation profiles",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s) containing aggregated profiles",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save the pairwise similarity matrix in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--metric",
        help="Similarity metric.",
        choices=["cosine", "pearson"],
        default="cosine",
    )
    parser.add_argument(
        "--perturbation",
        help="Column in obs used as row/column labels in the similarity matrix.",
        default="gene_symbol",
    )
    parser.add_argument(
        "--exclude-reference-query",
        help="Pandas query expression identifying profiles to exclude before "
        "computing similarities.  Recommended when map-center has been applied "
        "(the reference becomes the zero vector and cosine similarity is "
        "undefined).  Examples: \"gene_symbol=='NTC'\", \"type=='intergenic'\".",
        default=None,
        dest="exclude_reference_query",
    )
    parser.add_argument(
        "--output-format",
        help=(
            "Format of the output AnnData Zarr.  "
            "'matrix' (default): obs and var are both indexed by perturbation "
            "labels; X is the square (n × n) similarity matrix.  Compatible "
            "with map-recall as-is.  "
            "'anndata': profiles are stored in X (n × n_features) and the "
            "similarity matrix is stored in obsp[\"similarity\"].  obs retains "
            "all upstream metadata columns.  More suitable for downstream "
            "analysis in Python / scanpy."
        ),
        choices=["matrix", "anndata"],
        default="matrix",
        dest="output_format",
    )
    _cluster_args(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_similarity)


# ---------------------------------------------------------------------------
# map-recall
# ---------------------------------------------------------------------------


def _run_map_recall(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_recall

    run_pipeline_map_recall(arguments)


def _create_map_recall_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-recall",
        help="Evaluate the similarity matrix against one or more reference databases. "
        "Supports CORUM, GMT (Reactome/KEGG/GO/MSigDB), STRING, and Reactome FI.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to pairwise similarity matrix Zarr produced by map-similarity",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save recall results in Parquet format",
        required=True,
    )

    # --- Set-based reference sources (KS test) ---
    set_grp = parser.add_argument_group(
        "set-based references (KS test: within-set vs between-set)"
    )
    set_grp.add_argument(
        "--corum",
        help="Path(s) to CORUM complex files (tab-separated, columns: "
        "complex_name, subunits_gene_name with semicolon-separated genes). "
        "Download from mips.helmholtz-muenchen.de/corum.",
        nargs="+",
        default=None,
    )
    set_grp.add_argument(
        "--gmt",
        help="Path(s) to gene-set files in GMT format "
        "(Reactome pathways, KEGG, GO, MSigDB Hallmarks, etc.). "
        "Download from MSigDB (software.broadinstitute.org/gsea/msigdb) or "
        "ReactomeFI web interface.",
        nargs="+",
        default=None,
    )
    set_grp.add_argument(
        "--min-genes",
        help="Minimum number of genes a set must have in the similarity matrix "
        "to be evaluated.",
        default=10,
        type=int,
    )

    # --- Pairwise reference sources (recall of interacting pairs) ---
    pw_grp = parser.add_argument_group(
        "pairwise references (recall of known-interacting pairs)"
    )
    pw_grp.add_argument(
        "--string",
        help="Path(s) to STRING interaction TSV file(s). "
        "Expected columns: preferredName_A, preferredName_B, score (0–1000). "
        "Export from the STRING web interface or via: "
        "scallops map-recall --string-fetch (fetches at run time).",
        nargs="+",
        default=None,
    )
    pw_grp.add_argument(
        "--string-fetch",
        help="Query the STRING REST API for all genes in the similarity matrix. "
        "Requires an internet connection. Results are not cached.",
        action="store_true",
        dest="string_fetch",
    )
    pw_grp.add_argument(
        "--string-threshold",
        help="Minimum STRING combined score (0–1000) to include an interaction.",
        type=int,
        default=400,
        dest="string_threshold",
    )
    pw_grp.add_argument(
        "--string-species",
        help="NCBI taxonomy ID for the STRING query (default 9606 = Homo sapiens).",
        type=int,
        default=9606,
        dest="string_species",
    )
    pw_grp.add_argument(
        "--string-network-type",
        help="STRING network type: 'full' (all evidence) or 'physical' "
        "(physical interactions only).",
        choices=["full", "physical"],
        default="full",
        dest="string_network_type",
    )
    pw_grp.add_argument(
        "--reactome",
        help="Path(s) to Reactome Functional Interaction files (tab-separated). "
        "Download from reactomefip.wustl.edu.",
        nargs="+",
        default=None,
    )
    pw_grp.add_argument(
        "--min-pairs",
        help="Minimum number of reference pairs present in the similarity matrix "
        "to run the pairwise benchmark.",
        type=int,
        default=10,
        dest="min_pairs",
    )
    parser.add_argument(
        "--inject-zarr",
        help="Path for an AnnData Zarr copy of the input similarity matrix with "
        "recall results injected into uns['recall'].  When supplied, a new Zarr "
        "is written alongside the --output Parquet.  "
        "The uns['recall'] dict is keyed by source name; each value is a list of "
        "row records that can be reconstructed with "
        "pd.DataFrame(data.uns['recall']['source_name']).",
        default=None,
        dest="inject_zarr",
    )

    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_recall)


# ---------------------------------------------------------------------------
# map-sphere
# ---------------------------------------------------------------------------


def _run_map_sphere(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_sphere

    run_pipeline_map_sphere(arguments)


def _create_map_sphere_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-sphere",
        help="Apply ZCA sphering (whitening) to decorrelate features. "
        "Used in the pre-TVN pipeline after PCA.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save sphered data in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--by",
        help="Column(s) in obs to apply the transform per group "
        "(e.g. condition). When set the sphering matrix is fitted "
        "independently per group.",
        nargs="*",
    )
    parser.add_argument(
        "--epsilon",
        help="Regularisation constant added to singular values before inversion.",
        type=float,
        default=1e-5,
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_sphere)


# ---------------------------------------------------------------------------
# map-pca-select
# ---------------------------------------------------------------------------


def _run_map_pca_select(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_pca_select

    run_pipeline_map_pca_select(arguments)


def _create_map_pca_select_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-pca-select",
        help="Retain only statistically significant PCA components using the "
        "Tracy-Widom distribution (Johnstone 2001, Shekhar et al. 2022). "
        "Run after map-pca.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Path to input Zarr or Parquet file(s) from map-pca",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Path to save the selected PCA components in Zarr format",
        required=True,
    )
    parser.add_argument(
        "--method",
        help="Component selection strategy.  "
        "'variance' (default, recommended for morphological data): keep minimum "
        "PCs needed to explain --min-variance-fraction of total variance.  "
        "'permutation': column-independent permutation null, accounts for "
        "non-Gaussian marginals.  "
        "'tracy_widom': Tracy-Widom eigenvalue test — AVOID for correlated "
        "features (e.g. Cell Painting) because correlated features inflate all "
        "eigenvalues above the threshold, causing the test to retain everything.",
        choices=["variance", "permutation", "tracy_widom"],
        default="variance",
    )
    parser.add_argument(
        "--min-variance-fraction",
        help="(method=variance) Cumulative variance fraction to retain (0–1).",
        type=float,
        default=0.95,
        dest="min_variance_fraction",
    )
    parser.add_argument(
        "--pval",
        help="Significance level for method=permutation or method=tracy_widom.",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--n-perms",
        help="(method=permutation) Number of permutation replicates.",
        type=int,
        default=100,
        dest="n_perms",
    )
    parser.add_argument(
        "--max-components",
        help="Hard upper cap on retained components, applied after any method.",
        type=int,
        default=None,
        dest="max_components",
    )
    parser.add_argument(
        "--n-features",
        help="(method=tracy_widom) Number of original features used to fit PCA. "
        "Inferred from uns['pca']['features'] when omitted.",
        type=int,
        default=None,
        dest="n_features",
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_pca_select)


# ---------------------------------------------------------------------------
# Shared clustering argument group
# ---------------------------------------------------------------------------


def _cluster_args(parser: argparse.ArgumentParser) -> None:
    """Add all clustering arguments to *parser* as an optional argument group.

    Applies to both ``map-similarity`` (integrated clustering) and the
    standalone ``map-cluster`` command.
    """
    grp = parser.add_argument_group("clustering")
    grp.add_argument(
        "--cluster-method",
        help=(
            "Clustering algorithm applied to the similarity matrix.  "
            "Perturbations are reordered so same-cluster entries are adjacent.  "
            "``none`` disables clustering.  "
            "``hierarchical`` (default when clustering is enabled): "
            "agglomerative clustering via scipy; hyperparameter = n_clusters.  "
            "``hdbscan``: density-based hierarchical; hyperparameter = "
            "min_cluster_size (requires the ``hdbscan`` package).  "
            "``leiden``: graph community detection; hyperparameter = resolution "
            "(requires ``leidenalg`` and ``python-igraph``)."
        ),
        choices=["none", "hierarchical", "hdbscan", "leiden"],
        default=None,
        dest="cluster_method",
    )
    grp.add_argument(
        "--cluster-auto-params",
        help="Estimate the main hyperparameter of the chosen clustering method "
        "using an elbow criterion.  Enabled by default when --cluster-method is set.  "
        "Disable to use the explicitly supplied hyperparameter value.",
        action="store_true",
        default=True,
        dest="cluster_auto_params",
    )

    # Hierarchical
    hier = parser.add_argument_group("hierarchical clustering options")
    hier.add_argument(
        "--cluster-n-clusters",
        help="Target number of flat clusters for hierarchical clustering.  "
        "When omitted and --cluster-auto-params is set, the number is "
        "estimated from the largest gap in the dendrogram merge heights.",
        type=int,
        default=None,
        dest="cluster_n_clusters",
    )
    hier.add_argument(
        "--cluster-linkage",
        help="Linkage criterion for hierarchical clustering.",
        choices=["ward", "complete", "average", "single"],
        default="ward",
        dest="cluster_linkage",
    )
    hier.add_argument(
        "--cluster-max-n-clusters",
        help="Upper bound on the auto-estimated number of clusters.",
        type=int,
        default=50,
        dest="cluster_max_n_clusters",
    )

    # HDBSCAN
    hdb = parser.add_argument_group("HDBSCAN options")
    hdb.add_argument(
        "--cluster-min-cluster-size",
        help="Minimum cluster size for HDBSCAN.  Estimated via elbow when "
        "--cluster-auto-params is set.",
        type=int,
        default=None,
        dest="cluster_min_cluster_size",
    )
    hdb.add_argument(
        "--cluster-min-samples",
        help="HDBSCAN min_samples parameter (defaults to min_cluster_size).",
        type=int,
        default=None,
        dest="cluster_min_samples",
    )

    # Leiden
    lei = parser.add_argument_group("Leiden options")
    lei.add_argument(
        "--cluster-resolution",
        help="Leiden resolution parameter.  Estimated via elbow when "
        "--cluster-auto-params is set.",
        type=float,
        default=None,
        dest="cluster_resolution",
    )
    lei.add_argument(
        "--cluster-similarity-threshold",
        help="Minimum similarity to include an edge in the Leiden graph.",
        type=float,
        default=0.3,
        dest="cluster_similarity_threshold",
    )
    lei.add_argument(
        "--cluster-leiden-res-min",
        help="Lower bound of the resolution search range for Leiden elbow.",
        type=float,
        default=0.05,
        dest="cluster_leiden_res_min",
    )
    lei.add_argument(
        "--cluster-leiden-res-max",
        help="Upper bound of the resolution search range for Leiden elbow.",
        type=float,
        default=2.0,
        dest="cluster_leiden_res_max",
    )

    # Shared
    shared = parser.add_argument_group("clustering shared options")
    shared.add_argument(
        "--cluster-elbow-n-range",
        help="Number of candidate hyperparameter values to evaluate during "
        "the elbow search (HDBSCAN and Leiden).",
        type=int,
        default=20,
        dest="cluster_elbow_n_range",
    )
    shared.add_argument(
        "--cluster-random-state",
        help="Random seed for Leiden clustering.",
        type=int,
        default=0,
        dest="cluster_random_state",
    )


# ---------------------------------------------------------------------------
# map-cluster (standalone)
# ---------------------------------------------------------------------------


def _run_map_cluster(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_cluster

    run_pipeline_map_cluster(arguments)


def _create_map_cluster_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-cluster",
        help="Cluster perturbation profiles from a similarity AnnData Zarr and "
        "reorder the matrix so same-cluster entries are adjacent.  "
        "Supports hierarchical (default), HDBSCAN, and Leiden, each with "
        "automatic hyperparameter estimation via an elbow criterion.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Similarity AnnData Zarr from map-similarity (matrix or anndata format)",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Output AnnData Zarr path with clustering applied",
        required=True,
    )
    required.add_argument(
        "--method",
        help="Clustering algorithm: 'hierarchical' (default), 'hdbscan', or 'leiden'.",
        choices=["hierarchical", "hdbscan", "leiden"],
        default="hierarchical",
    )

    # Hierarchical
    hier = parser.add_argument_group("hierarchical clustering options")
    hier.add_argument(
        "--n-clusters",
        help="Target number of clusters.  Estimated from dendrogram elbow "
        "when --auto-params is set.",
        type=int,
        default=None,
        dest="n_clusters",
    )
    hier.add_argument(
        "--linkage",
        help="Linkage criterion.",
        choices=["ward", "complete", "average", "single"],
        default="ward",
    )
    hier.add_argument(
        "--max-n-clusters",
        help="Upper bound for auto-estimated n_clusters.",
        type=int,
        default=50,
        dest="max_n_clusters",
    )

    # HDBSCAN
    hdb = parser.add_argument_group("HDBSCAN options")
    hdb.add_argument(
        "--min-cluster-size",
        help="HDBSCAN min_cluster_size.  Estimated via elbow when --auto-params.",
        type=int,
        default=None,
        dest="min_cluster_size",
    )
    hdb.add_argument(
        "--min-samples",
        help="HDBSCAN min_samples.",
        type=int,
        default=None,
        dest="min_samples",
    )

    # Leiden
    lei = parser.add_argument_group("Leiden options")
    lei.add_argument(
        "--resolution",
        help="Leiden resolution.  Estimated via elbow when --auto-params.",
        type=float,
        default=None,
    )
    lei.add_argument(
        "--similarity-threshold",
        help="Minimum similarity to include an edge in the Leiden graph.",
        type=float,
        default=0.3,
        dest="similarity_threshold",
    )
    lei.add_argument(
        "--leiden-res-min",
        type=float,
        default=0.05,
        dest="leiden_res_min",
    )
    lei.add_argument(
        "--leiden-res-max",
        type=float,
        default=2.0,
        dest="leiden_res_max",
    )

    # Shared
    shared = parser.add_argument_group("shared options")
    shared.add_argument(
        "--auto-params",
        help="Estimate the main hyperparameter via elbow criterion (default: True).",
        action="store_true",
        default=True,
        dest="auto_params",
    )
    shared.add_argument(
        "--elbow-n-range",
        help="Number of candidate values to evaluate in the elbow search.",
        type=int,
        default=20,
        dest="elbow_n_range",
    )
    shared.add_argument(
        "--random-state",
        type=int,
        default=0,
        dest="random_state",
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_cluster)


# ---------------------------------------------------------------------------
# map-backproject
# ---------------------------------------------------------------------------


def _run_map_backproject(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_backproject

    run_pipeline_map_backproject(arguments)


def _create_map_backproject_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "map-backproject",
        help="Rank original features by their contribution to a query vs. reference "
             "centroid difference, backprojected through the TVN/PCA chain.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--input",
        help="Path to TVN/aggregated AnnData Zarr (e.g. profiles.zarr from map agg).",
        required=True,
        nargs="+",
    )
    required.add_argument(
        "--output",
        help="Output Parquet path for the ranked feature table.",
        required=True,
    )

    query_grp = parser.add_argument_group(
        "query selector (provide --query OR --cluster-query + --cluster-labels-zarr)"
    )
    query_grp.add_argument(
        "--query",
        nargs="+",
        dest="query",
        metavar="PERTURBATION",
        help="Perturbation name(s) in --perturbation-column that form the query set.",
    )
    query_grp.add_argument(
        "--cluster-query",
        dest="cluster_query",
        metavar="LABEL",
        help="Cluster label value to select as the query set "
             "(requires --cluster-labels-zarr).",
    )
    query_grp.add_argument(
        "--cluster-labels-zarr",
        dest="cluster_labels_zarr",
        metavar="PATH",
        help="Similarity AnnData Zarr containing obs['cluster'] labels "
             "(written by map cluster / map similarity).",
    )

    ref_grp = parser.add_argument_group(
        "reference selector (default: all non-query observations)"
    )
    ref_grp.add_argument(
        "--reference",
        nargs="+",
        dest="reference",
        metavar="PERTURBATION",
        help="Perturbation name(s) that form the reference set.",
    )
    ref_grp.add_argument(
        "--cluster-ref",
        dest="cluster_ref",
        metavar="LABEL",
        help="Cluster label value(s) to use as the reference set.",
    )

    parser.add_argument(
        "--perturbation-column",
        default="gene_symbol",
        dest="perturbation_column",
        help="obs column that identifies perturbations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Return only the top-k features by |score|.  None = all.",
    )
    parser.add_argument(
        "--pc-stat-filter",
        choices=["ttest", "mannwhitney"],
        default=None,
        dest="pc_stat_filter",
        help="Statistical test to prune non-significant PC components before "
             "backprojection.  None skips filtering.",
    )
    parser.add_argument(
        "--pc-pvalue-threshold",
        type=float,
        default=0.05,
        dest="pc_pvalue_threshold",
        help="p-value cutoff for --pc-stat-filter.",
    )
    parser.add_argument(
        "--group",
        default=None,
        dest="group",
        help="Covariance-alignment group key (required when TVN was run with "
             "--tvn-by and multiple groups exist).",
    )
    parser.add_argument(
        "--to-original-scale",
        action="store_true",
        dest="to_original_scale",
        help="Backproject past the z-scoring step to recover original measurement units.",
    )
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_map_backproject)


# ---------------------------------------------------------------------------
# map subcommand registry
# ---------------------------------------------------------------------------


class _Renaming:
    """Minimal subparsers proxy that replaces the command name on add_parser.

    Used by register_map_subcommands so that existing _create_map_*_parser
    functions (which hard-code names like "map-filter") can be reused
    unchanged — only the command name seen by the user changes.
    """

    def __init__(self, sp, short_name: str):
        self._sp = sp
        self._short_name = short_name

    def add_parser(self, _ignored_name: str, **kwargs):
        return self._sp.add_parser(self._short_name, **kwargs)


def register_map_subcommands(
    map_subparsers: argparse.ArgumentParser,
    default_help: bool,
) -> None:
    """Register all map subcommands under the ``map`` parent parser.

    Each step gets a short name (e.g. ``filter``, ``tvn``) instead of the
    flat top-level name (e.g. ``map-filter``, ``map-tvn``).  The
    implementation delegates to the existing ``_create_map_*_parser``
    functions unchanged; only the subparser name is overridden via the
    :class:`_Renaming` proxy.

    :param map_subparsers: The subparsers action created on the ``map`` parser.
    :param default_help: Passed through to each parser creator.
    """
    _create_map_filter_parser(_Renaming(map_subparsers, "filter"), default_help)
    _create_map_transform_yj_parser(_Renaming(map_subparsers, "transform-yj"), default_help)
    _create_map_scale_parser(_Renaming(map_subparsers, "scale"), default_help)
    _create_map_pca_parser(_Renaming(map_subparsers, "pca"), default_help)
    _create_map_pca_select_parser(_Renaming(map_subparsers, "pca-select"), default_help)
    _create_map_sphere_parser(_Renaming(map_subparsers, "sphere"), default_help)
    _create_map_tvn_parser(_Renaming(map_subparsers, "tvn"), default_help)
    _create_map_agg_parser(_Renaming(map_subparsers, "agg"), default_help)
    _create_map_center_parser(_Renaming(map_subparsers, "center"), default_help)
    _create_map_similarity_parser(_Renaming(map_subparsers, "similarity"), default_help)
    _create_map_cluster_parser(_Renaming(map_subparsers, "cluster"), default_help)
    _create_map_recall_parser(_Renaming(map_subparsers, "recall"), default_help)
    _create_map_backproject_parser(_Renaming(map_subparsers, "backproject"), default_help)
    _create_run_parser(map_subparsers, default_help)


# ---------------------------------------------------------------------------
# map run  (full pipeline)
# ---------------------------------------------------------------------------


def _run_map_run(arguments: argparse.Namespace):
    from scallops.cli.map_build import run_pipeline_map_run

    run_pipeline_map_run(arguments)


def _create_run_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    """Register ``scallops map run`` — the single-machine full pipeline runner.

    Chains all individual ``map`` steps in order, saving each result as an
    AnnData Zarr in ``--output-dir``.  Steps whose output already exists are
    skipped unless ``--force`` is set, so interrupted runs can be resumed.

    Step order::

        filter → transform-yj → scale → pca → pca-select → sphere
        → tvn → agg → center → similarity → cluster → recall

    Intermediate files are named ``{output_dir}/NN_stepname.zarr``.
    """
    parser = subparsers.add_parser(
        "run",
        help="Run the full map-building pipeline on a single machine",
        description=_create_run_parser.__doc__,
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )

    # ── Required ─────────────────────────────────────────────────────────────
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "-i", "--input",
        help="Input AnnData Zarr or Parquet file (output of pooled-sbs merge).",
        required=True, nargs="+",
    )
    req.add_argument(
        "-o", "--output-dir",
        help="Directory to store all intermediate and final AnnData Zarr outputs.",
        required=True, dest="output_dir",
    )

    # ── Pipeline control ──────────────────────────────────────────────────────
    pipe = parser.add_argument_group("pipeline control")
    pipe.add_argument(
        "--steps",
        help=(
            "Comma-separated list of steps to run, or 'all' (default).  "
            "Available: filter, transform-yj, scale, pca, pca-select, sphere, "
            "tvn, agg, center, similarity, recall.  "
            "Steps not listed are skipped; their expected output must already "
            "exist in --output-dir (resume after partial failure)."
        ),
        default="all",
    )
    pipe.add_argument("--force", action="store_true",
                      help="Re-run all steps even if their output exists.")
    pipe.add_argument("--no-version", action="store_true", dest="no_version",
                      help="Do not record scallops version in step provenance.")

    # ── Condition column (derived from a well / obs mapping) ─────────────────
    cond = parser.add_argument_group(
        "condition column (create a derived obs column for TVN --by grouping)"
    )
    cond.add_argument(
        "--condition-column",
        help="Name of the new obs column to create.  "
        "Must be used together with --condition-map.  "
        "Example: --condition-column condition",
        default=None,
        dest="condition_column",
    )
    cond.add_argument(
        "--condition-source-column",
        help="Existing obs column whose values are looked up in --condition-map.  "
        "Default: 'well'.",
        default="well",
        dest="condition_source_column",
    )
    cond.add_argument(
        "--condition-map",
        help="JSON dict mapping source-column values → condition labels.  "
        "Example: '{\"1\":\"GIRED\",\"2\":\"GIRED\",\"3\":\"GIRED\","
        "\"4\":\"DMSO\",\"5\":\"DMSO\",\"6\":\"DMSO\"}'.  "
        "All source values present in the data must appear in the map.",
        default=None,
        dest="condition_map",
    )

    # ── Shared / reference ────────────────────────────────────────────────────
    shared = parser.add_argument_group("shared reference and grouping")
    shared.add_argument(
        "--reference-query",
        help="Pandas query identifying negative-control cells used in pca "
             "(fit), tvn, and center steps.",
        default="gene_symbol=='NTC'", dest="reference_query",
    )
    shared.add_argument(
        "--perturbation",
        help="obs column that identifies perturbations (used by agg and similarity).",
        default="gene_symbol",
    )
    shared.add_argument(
        "--plate-column", default="plate", dest="plate_column",
        help="obs column identifying the plate.  Used as part of the experimental-unit "
             "grouping for filter, transform-yj, and scale (all three stratify by "
             "plate × well).  Set to the same value as --well-column if you have no "
             "plate structure.",
    )
    shared.add_argument(
        "--well-column", default="well", dest="well_column",
        help="obs column identifying the well.  Together with --plate-column this "
             "defines the experimental unit used by filter, transform-yj, and scale.",
    )
    shared.add_argument(
        "--tvn-by",
        help=(
            "Column(s) in obs for TVN per-group covariance alignment.  "
            "This is a SEPARATE question from the plate/well grouping used by "
            "filter / transform-yj / scale.  Those steps always group by "
            "plate × well (the experimental unit).  --tvn-by asks which *biological* "
            "group should receive its own alignment matrix: e.g. 'condition' aligns "
            "GIRED and DMSO cells separately; 'plate' corrects plate-to-plate "
            "covariance shifts; omit for a single global alignment.  "
            "Requires a column that already exists in obs or is created by "
            "--condition-column / --condition-map."
        ),
        nargs="*", default=None,
        dest="tvn_by",
    )

    # ── Filtering ─────────────────────────────────────────────────────────────
    filt = parser.add_argument_group("feature / cell filtering (map filter step)")
    filt.add_argument(
        "--label-filter",
        help="Pandas query expression to filter cells before any processing step "
        "(e.g. \"barcode_count_0 / barcode_count > 0.5\").  Applied at data load time.",
        default=None, dest="label_filter",
    )
    filt.add_argument("--min-variance", type=float, default=0.1, dest="min_variance")
    filt.add_argument("--max-variance", type=float, default=5.0, dest="max_variance")
    filt.add_argument("--max-fraction-not-finite", type=float, default=0.25,
                      dest="max_fraction_not_finite")
    filt.add_argument(
        "--filter-batch-size", type=int, default=500_000, dest="filter_batch_size",
        help="Rows per streaming batch during parquet filter (default 500 000). "
             "Larger = more RAM used but fewer batches; auto-scales readahead "
             "to stay within available memory.",
    )
    filt.add_argument("--max-correlation", type=float, default=None, dest="max_correlation",
                      help="Remove pairs of features with |r| above this threshold "
                           "(disabled by default).")
    filt.add_argument(
        "--batch-column", default=None, dest="batch_column",
        help="obs column for batch-correlation filter (disabled when omitted).",
    )
    filt.add_argument(
        "--batch-pvalue", type=float, default=0.05, dest="batch_pvalue",
        help="Significance threshold for batch-association test.",
    )
    filt.add_argument(
        "--batch-method", choices=["kruskal", "anova"], default="kruskal",
        dest="batch_method",
    )
    filt.add_argument(
        "--batch-reference", default=None, dest="batch_reference",
        help="Query restricting the batch-correlation test to reference cells "
             "(e.g. \"gene_symbol=='NTC'\").",
    )

    # ── Scale method (global or local z-score, always by plate × well) ────────
    scale = parser.add_argument_group(
        "scale method (global or local z-score, always grouped by plate × well)"
    )
    scale.add_argument(
        "--scale-method",
        help=(
            "How to z-score features within each plate × well group.  "
            "'global' (default): subtract the well mean and divide by the well std "
            "computed across all cells in that well.  "
            "'local': spatial k-NN z-score — each cell is normalised relative to "
            "its --localz-neighbors nearest spatial neighbours within the same well, "
            "removing both global well bias and local spatial gradients.  "
            "The two modes are mutually exclusive alternatives; do not combine them."
        ),
        choices=["global", "local"],
        default="global",
        dest="scale_method",
    )
    scale.add_argument(
        "--localz-neighbors",
        help="k for spatial k-NN z-score (used when --scale-method local).",
        type=int, default=75, dest="localz_neighbors",
    )
    scale.add_argument(
        "--localz-max-value",
        help="Clip local z-scores to ±this value (used when --scale-method local).",
        type=float, default=5.0, dest="localz_max_value",
    )
    scale.add_argument(
        "--localz-centroid-y",
        help="obs column for cell centroid Y coordinate.",
        default="Nuclei_AreaShape_Center_Y", dest="localz_centroid_y",
    )
    scale.add_argument(
        "--localz-centroid-x",
        help="obs column for cell centroid X coordinate.",
        default="Nuclei_AreaShape_Center_X", dest="localz_centroid_x",
    )
    scale.add_argument(
        "--localz-batch-size",
        help="Number of cells processed per batch in local z-score.  Controls "
             "the size of the intermediate (batch × k_neighbors × features) array.  "
             "Lower values use less RAM; higher values run faster.  "
             "Default 50 000 → ~75 GB peak on 5 000 features / 75 neighbours.",
        type=int, default=50_000, dest="localz_batch_size",
    )

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca = parser.add_argument_group("PCA (map pca + pca-select steps)")
    pca.add_argument("--pca-components", type=int, default=128, dest="pca_components",
                     help="Number of PCA components to fit.")
    pca.add_argument("--pca-batch-size", type=int, default=200_000, dest="pca_batch_size",
                     help="Batch size for incremental PCA.")
    pca.add_argument("--pca-select-method", default="variance",
                     choices=["variance", "permutation", "tracy_widom"],
                     dest="pca_select_method",
                     help="Component selection method after PCA.")
    pca.add_argument("--pca-variance-fraction", type=float, default=0.95,
                     dest="pca_variance_fraction",
                     help="Minimum cumulative variance fraction to retain "
                          "(method=variance only).")

    # ── Aggregation ───────────────────────────────────────────────────────────
    agg = parser.add_argument_group("profile aggregation (map agg step)")
    agg.add_argument("--agg-by", nargs="+", default=None, dest="agg_by",
                     help="obs column(s) to aggregate by.  Defaults to --perturbation.")
    agg.add_argument("--agg-method", choices=["mean", "median"], default="mean",
                     dest="agg_method")
    agg.add_argument("--min-cells", type=int, default=None, dest="min_cells",
                     help="Exclude perturbations with fewer cells before aggregation.")

    # ── Similarity ────────────────────────────────────────────────────────────
    sim = parser.add_argument_group("similarity (map similarity step)")
    sim.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")

    # ── Clustering (all options from map cluster, incl. auto-param tuning) ───
    _cluster_args(parser)

    # ── Recall ────────────────────────────────────────────────────────────────
    rec = parser.add_argument_group("recall benchmarks (map recall step)")
    rec.add_argument("--corum", nargs="+", default=None,
                     help="CORUM complex file(s).")
    rec.add_argument("--gmt", nargs="+", default=None,
                     help="GMT gene-set file(s) (Reactome, KEGG, GO, MSigDB).")
    rec.add_argument("--string-fetch", action="store_true", dest="string_fetch",
                     help="Query the STRING REST API for all perturbations.")
    rec.add_argument("--string-threshold", type=int, default=400,
                     dest="string_threshold")
    rec.add_argument("--string-species", type=int, default=9606,
                     dest="string_species",
                     help="NCBI taxonomy ID for STRING queries (default 9606 = human).")
    rec.add_argument("--string-network-type",
                     choices=["full", "physical"], default="full",
                     dest="string_network_type")
    rec.add_argument("--min-genes", type=int, default=5, dest="min_genes")
    rec.add_argument("--min-pairs", type=int, default=10, dest="min_pairs")

    parser.set_defaults(func=_run_map_run)
