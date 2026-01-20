"""CLI Module for pooled dialout library sequencing data analysis and reporting.

This module provides functionality for analyzing pooled dialout library sequencing data
and generating reports through command-line interface (CLI). It includes functions for
running the dialout pipeline, creating analysis reports, and combining statistics.

Authors:
    - The SCALLOPS development team

Note: This module requires the BWA tool for indexing and mapping.
"""

import argparse
import glob
import os
import shutil
import tempfile
from collections import defaultdict
from operator import itemgetter
from subprocess import check_call
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import seaborn as sns
from dask.bag import from_sequence
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import pairwise_distances_argmin_min

from scallops.cli.util import _get_cli_logger
from scallops.io import _get_fs_protocol

logger = _get_cli_logger()


def bwa_index(fasta: str):
    """Indexes a FASTA file using BWA.

    :param fasta: Path to the FASTA file to be indexed.
    :return: None
    """
    # e.g. TCTTGTGGAAAGGACGAAACACCGNNNNNNNNNNNNNNNNNNNNGTTTTAGAGCTAGAAATAGCAAGTTAAAATA
    if not os.path.exists(fasta + ".bwt") or os.path.getmtime(
        fasta + ".bwt"
    ) < os.path.getmtime(fasta):
        check_call(["bwa", "index", fasta])


def hamming_distance(
    query_df: pd.DataFrame, design_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Hamming distance between sequences in the query DataFrame and the design
    DataFrame.

    :param query_df: DataFrame containing query sequences.
    :param design_df: DataFrame containing reference sequences.
    :return: Tuple of two NumPy arrays, the first containing indices of the closest
        matches, and the second containing the Hamming distances.
    """
    unmapped_reads = np.array([[ord(b) for b in read] for read in query_df["sequence"]])
    whitelist = np.array([[ord(b) for b in read] for read in design_df.index])

    argmin, distances = pairwise_distances_argmin_min(
        unmapped_reads, whitelist, metric="hamming"
    )
    return argmin, (distances * len(whitelist[0])).astype(int)


def bwa(fastq: str, output_dir: str, fasta: str, bwa_threads: int):
    """Run BWA on a FASTQ.

    :param fastq: Path to the input FASTQ file containing sequencing reads.
    :param output_dir: Path to the output directory where analysis results will be
        stored.
    :param fasta: Path to the FASTA file representing the reference genome.
    :param bwa_threads: Number of threads to use for BWA alignment.
    :return: Path to output sam file
    """
    basename = os.path.splitext(os.path.basename(fastq))[0].split("_")[0]
    bwa_output = os.path.join(output_dir, f"{basename}.sam")
    logger.info(f"Running dialout on {fastq}")
    fastq_fs, _ = fsspec.core.url_to_fs(fastq)
    local_path = None
    if _get_fs_protocol(fastq_fs) != "file":
        tmp_dir = tempfile.mkdtemp()
        local_path = os.path.join(tmp_dir, os.path.basename(fastq))
        fastq_fs.get(fastq, local_path)
        fastq = local_path

    check_call(
        [
            "bwa",
            "mem",
            "-v",
            "1",
            "-o",
            bwa_output,
            "-t",
            str(bwa_threads),
            fasta,
            fastq,
        ]
    )
    if local_path is not None:
        os.remove(local_path)
    return bwa_output


def single_pool(
    fastq: str,
    output_dir: str,
    fasta: str,
    bwa_threads: int,
    gRNA_start: int,
    gRNA_end: int,
    design_df: pd.DataFrame,
    min_mismatches: int = 3,
    min_counts: int = 10,
    save_unaligned_fasta: bool = True,
) -> str:
    """Process a single pool of sequencing reads from a dialout library.

    This function takes a set of sequencing reads (a 'pool') obtained from a pooled
    dialout library, aligns the reads to a reference genome, identifies and counts
    occurrences of specific genetic sequences (gRNAs), and generates output files with
    relevant information.

    :param fastq: Path to the input FASTQ file containing sequencing reads.
    :param output_dir: Path to the output directory where analysis results will be
        stored.
    :param fasta: Path to the FASTA file representing the reference genome.
    :param bwa_threads: Number of threads to use for BWA alignment.
    :param gRNA_start: Start position of the region of interest (gRNA) in the reference
        genome.
    :param gRNA_end: End position of the region of interest (gRNA) in the reference
        genome.
    :param design_df: DataFrame containing design information, including 'dialout'
        values and spacer sequences.
    :param min_mismatches: Minimum number of mismatches allowed for a sequence to be
        considered a match.
    :param min_counts: Minimum read count threshold for a sequence to be included in
        the analysis.
    :param save_unaligned_fasta: Whether to save unaligned sequences to fasta file
    :return: A string representing the basename of the processed pool.
    """

    # e.g. T2-A04_S100_L001_R1_001.fastq
    from pysam import AlignmentFile

    expected_len = gRNA_end - gRNA_start + 1
    n_unmapped = 0
    bwa_output = bwa(
        fastq=fastq, output_dir=output_dir, fasta=fasta, bwa_threads=bwa_threads
    )
    basename = os.path.splitext(os.path.basename(fastq))[0].split("_")[0]
    with AlignmentFile(bwa_output, "rb") as aligned:
        seq_2_counts = defaultdict(lambda: 0)
        unaligned_seq_2_counts = defaultdict(lambda: 0)
        for read in aligned.fetch(until_eof=True):
            seq = read.seq
            if read.is_mapped:
                # Remove sequence before and after gRNA region
                seq = seq[gRNA_start : gRNA_end + 1]
                # reference_name = read.reference_name
                # seq_2_counts = reference_2_seq_2_counts[reference_name]
                seq_2_counts[seq] += 1
            else:
                n_unmapped += 1
                unaligned_seq_2_counts[seq] += 1
    if len(unaligned_seq_2_counts) > 0 and save_unaligned_fasta:
        with open(
            os.path.join(output_dir, f"{basename}-reference-unaligned.fa"), "wt"
        ) as fasta:
            for counter, (seq, count) in enumerate(
                sorted(unaligned_seq_2_counts.items(), key=itemgetter(1), reverse=True)
            ):
                line = f">seq{counter};count={count}\n{seq}\n"
                fasta.write(line)

    # convert to dataframe
    sequences = []
    counts = []
    for seq in seq_2_counts:
        sequences.append(seq)
        counts.append(seq_2_counts[seq])
    mapped_counts_df = pd.DataFrame(dict(sequence=sequences, count=counts))
    design_df["in_design"] = True  # indicate whether sequence found in design.csv
    mapped_counts_df = mapped_counts_df.join(design_df, on="sequence", how="outer")
    mapped_counts_df["in_design"] = mapped_counts_df["in_design"].fillna(False)
    mapped_counts_df["count"] = mapped_counts_df["count"].fillna(0)
    mapped_counts_df["index"] = basename
    mapped_counts_df["mismatches"] = 0
    mapped_counts_df["count_fraction"] = (
        mapped_counts_df["count"] / mapped_counts_df["count"].sum()
    )
    mapped_counts_df["closest_match"] = ""
    mapped_counts_df = mapped_counts_df.sort_values("count", ascending=False)

    mismatched_counts_df = mapped_counts_df[~mapped_counts_df["in_design"]]
    mapped_counts_df.loc[
        mismatched_counts_df.index, "mismatches"
    ] = -1  # mismatch reads get default value of `mismatches == -1`

    mismatched_counts_df = mismatched_counts_df[
        mismatched_counts_df["sequence"].str.len() == expected_len
    ]  # don't find the closest match for reads that are not the expected length

    if len(mismatched_counts_df) > 0:
        argmin, distances = hamming_distance(
            mismatched_counts_df.drop_duplicates(subset="sequence", ignore_index=True),
            design_df,
        )
        mapped_counts_df.loc[mismatched_counts_df.index, "mismatches"] = distances
        mapped_counts_df.loc[mismatched_counts_df.index, "closest_match"] = (
            design_df.index[argmin]
        )

        if save_unaligned_fasta:
            mismatched_counts_df = mapped_counts_df.query(
                f"count >= {min_counts} & mismatches >= {min_mismatches}"
            )
            lines = [
                f">seq{i};count={item['count']};mismatches={item['mismatches']};closest={item['closest_match']}"
                f"\n{item['sequence']}"
                for i in range(len(mismatched_counts_df))
                if (item := mismatched_counts_df.iloc[i])
            ]
            with open(
                os.path.join(output_dir, f"{basename}-dialout-unaligned.fa"), "wt"
            ) as fasta:
                fasta.write("\n".join(lines))
    mapped_counts_df.to_csv(
        os.path.join(output_dir, f"{basename}.counts.csv"), index=False
    )
    mapped_counts_df = mapped_counts_df[mapped_counts_df["in_design"]]
    dropout_df = mapped_counts_df[mapped_counts_df["count"] == 0]
    dropout_df.to_csv(os.path.join(output_dir, f"{basename}.dropout.csv"), index=False)

    n_mapped = mapped_counts_df["count"].sum()
    fraction_mapped = n_mapped / (n_mapped + n_unmapped)
    n_drop_outs = len(dropout_df)
    drop_out_ratio = n_drop_outs / len(design_df)
    average_read_count = mapped_counts_df["count"].mean()
    quantiles = mapped_counts_df["count"].quantile([0.9, 0.1]).values
    skew_ratio = quantiles[0] / quantiles[1]
    stats_data = [
        [
            basename,
            n_mapped,
            fraction_mapped,
            average_read_count,
            skew_ratio,
            drop_out_ratio,
            n_drop_outs,
        ]
    ]
    stats_df = pd.DataFrame(
        stats_data,
        columns=[
            "index",
            "n_mapped",
            "fraction_mapped",
            "average_read_count",
            "skew_ratio",
            "drop_out_ratio",
            "n_drop_outs",
        ],
    )

    stats_df.to_csv(os.path.join(output_dir, f"{basename}.stats.csv"), index=False)
    return basename


def get_pools(output_dir: str) -> list[str]:
    """Retrieve a list of pool names based on the SAM files in the specified directory.

    Pools are identified by the names of SAM files present in the given directory,
    where each SAM file corresponds to the output of processing sequencing reads for a
    specific pool in the dialout library.

    :param output_dir: Path to the directory containing SAM files from the analysis
        pipeline.
    :return: A list of pool names.
    """

    pools = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.iglob(os.path.join(output_dir, "*.sam"))
    ]
    if len(pools) == 0:
        pools = [
            os.path.basename(path)[:-10]
            for path in glob.iglob(os.path.join(output_dir, "*.stats.csv"))
        ]
    assert len(pools) > 0, "No pools found"
    return pools


def report(
    analysis_dir: str,
    output: str,
    design_df: pd.DataFrame,
    gene_id_col: str = "gene_id",
    pools: Optional[list[str]] = None,
    min_total_reads: int = 10,
    min_sample_reads: int = 2,
    sample_names: Optional[str] = None,
) -> None:
    """Create a report from the analysis pipeline output.

    This function generates a comprehensive report based on the results of the dialout
    library sequencing data analysis. The report includes summary statistics, scatter
    plots, and other visualizations to help interpret the sequencing data.

    :param analysis_dir: Path to the analysis output directory.
    :param design_df: Design dataframe
    :param gene_id_col: Column in design data frame containing gene id
    :param output: Path to the output PDF file or directory where PNG images will be
        created.
    :param pools: Optional list of specific pools to include in the report.
    :param min_total_reads: Minimum total reads to include a guide in the pairwise
        sample plot.
    :param min_sample_reads: Minimum reads to include a guide in the individual sample
        guide rank plot.
    :param sample_names: Optional path to a CSV file containing `index` and `sample`
        columns.

    """
    if pools is None:
        pools = get_pools(analysis_dir)
    index_rename = None
    if sample_names is not None:
        index_rename = pd.read_csv(
            sample_names, index_col="index", usecols=["index", "sample"]
        ).to_dict()["sample"]
    stats_df = combine_stats(analysis_dir, pools, index_rename)
    counts_df = join_counts(
        analysis_dir, pools, design_df=design_df, index_rename=index_rename
    )
    counts_df = counts_df.copy()
    alpha = 0.6
    count_columns = counts_df.columns[
        counts_df.columns.str.startswith("count_")
        & ~counts_df.columns.str.startswith("count_fraction")
    ]
    for c in count_columns:
        counts_df[c] = counts_df[c].fillna(0)
    count_fraction_columns = counts_df.columns[
        counts_df.columns.str.startswith("count_fraction")
    ]
    total_reads = counts_df[count_columns].sum(axis=1)
    counts_df["total_reads"] = total_reads
    counts_df["match"] = counts_df["mismatches"] == 0
    output_format = "pdf" if output.lower().endswith("pdf") else "png"
    if output_format == "pdf":
        pdf = PdfPages(output)
    elif not os.path.exists(output):
        os.makedirs(output)

    g = sns.catplot(
        data=pd.melt(stats_df, id_vars="index"),
        col="variable",
        x="index",
        y="value",
        kind="bar",
        col_wrap=2,
        sharey=False,
    )
    g.set_xticklabels(g.axes.flat[-1].get_xticklabels(), rotation=90)
    g.tight_layout()
    plt.suptitle("Summary Statistics")
    if output_format == "pdf":
        pdf.savefig()
    else:
        plt.savefig(os.path.join(output, "summary-stats.png"))
    plt.close()

    exact_matches_df = counts_df.query("mismatches==0")
    sns.pairplot(
        exact_matches_df,
        corner=True,
        vars=count_columns,
        hue="total_reads",
        diag_kws=dict(color=".2", hue=None),
        kind="scatter",
        plot_kws={"alpha": alpha},
    )
    plt.suptitle("Matches")
    if output_format == "pdf":
        pdf.savefig()
    else:
        plt.savefig(os.path.join(output, "matches.png"))
    plt.close()

    if gene_id_col in counts_df.columns:
        agg_dict = {c: "sum" for c in count_columns}
        agg_dict["total_reads"] = "sum"
        sns.pairplot(
            exact_matches_df.groupby(gene_id_col).agg(agg_dict),
            corner=True,
            vars=count_columns,
            hue="total_reads",
            diag_kws=dict(color=".2", hue=None),
            kind="scatter",
            plot_kws={"alpha": alpha},
        )
        plt.suptitle("Matches By Gene")
        if output_format == "pdf":
            pdf.savefig()
        else:
            plt.savefig(os.path.join(output, "matches-gene.png"))
        plt.close()

    sns.pairplot(
        counts_df.query(f"mismatches > 0 & total_reads >= {min_total_reads}"),
        corner=True,
        vars=count_columns,
        hue="mismatches",
        diag_kws=dict(color=".2", hue=None),
        kind="scatter",
        plot_kws={"alpha": alpha},
    )
    plt.suptitle(f"Mismatches with at least {min_total_reads:,} total reads")
    if output_format == "pdf":
        pdf.savefig()
    else:
        plt.savefig(os.path.join(output, "mismatches.png"))
    plt.close()

    # rank vs. reads colored by mismatches
    ncols = len(count_fraction_columns)
    nrows = 1
    col_wrap = 3
    if ncols > col_wrap:
        nrows = int(np.ceil(ncols / col_wrap))
        ncols = col_wrap

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(20.96, 20))
    axes = axes.ravel()
    for ax in axes:
        ax.set_visible(False)
    for i in range(len(count_fraction_columns)):
        axes[i].set_visible(True)
        fraction_column = count_fraction_columns[i]
        count_column = count_columns[i]
        _counts_df = counts_df[counts_df[count_column] >= min_sample_reads]
        ranks = _counts_df[count_column].rank(ascending=False)
        # sns.ecdfplot(x=_counts_df[c], hue=_counts_df["match"])
        sns.scatterplot(
            x=ranks, y=_counts_df[fraction_column], hue=_counts_df["match"], ax=axes[i]
        )
        axes[i].set_xlabel("Rank")

    plt.suptitle("Ranks")
    if output_format == "pdf":
        pdf.savefig()
    else:
        plt.savefig(os.path.join(output, "ranks.png"))
    plt.close()


def combine_stats(
    output_dir: str, pools: list[str], index_rename: Optional[dict[str, str]] = None
) -> pd.DataFrame:
    """Combine statistics from multiple pools into a single DataFrame.

    :param output_dir: Path to the directory containing individual pool statistics.
    :param pools: List of pool names to include in the combined statistics.
    :param index_rename: Optional dictionary to rename pool indices in the combined
        DataFrame.
    :return: A pandas DataFrame containing the combined statistics.
    """

    df = None
    for pool in pools:
        path = os.path.join(output_dir, pool + ".stats.csv")
        _df = pd.read_csv(path)
        df = pd.concat((df, _df)) if df is not None else _df
    if index_rename is not None:
        df["index"] = df["index"].replace(index_rename)
    return df


def join_counts(
    output_dir: str,
    pools: list[str],
    design_df: pd.DataFrame,
    index_rename: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Join counts from multiple pools into a single DataFrame.

    :param output_dir: Path to the directory containing individual pool count files.
    :param pools: List of pool names to include in the combined counts.
    :param design_df: DataFrame containing the design information for the guide
        sequences.
    :param index_rename: Optional dictionary to rename pool indices in the combined
        DataFrame.
    :return: A pandas DataFrame containing the combined counts.
    """

    df = None
    mismatches_df = None
    if index_rename is None:
        index_rename = {}
    for pool in pools:
        path = os.path.join(output_dir, pool + ".counts.csv")
        _df = pd.read_csv(
            path,
            usecols=[
                "sequence",
                "count",
                "count_fraction",
                "mismatches",
                "closest_match",
            ],
        )
        _mismatches_df = _df[["sequence", "mismatches", "closest_match"]].set_index(
            ["sequence", "mismatches", "closest_match"]
        )
        mismatches_df = (
            mismatches_df.join(_mismatches_df, how="outer")
            if mismatches_df is not None
            else _mismatches_df
        )

        _df = _df.set_index("sequence").drop(["mismatches", "closest_match"], axis=1)
        _df.columns = _df.columns + "_" + index_rename.get(pool, pool)
        df = df.join(_df, how="outer") if df is not None else _df

    df = (
        df.reset_index()
        .join(design_df, on="sequence", how="outer")
        .join(mismatches_df.reset_index().set_index("sequence"), on="sequence")
    )

    return df


def read_design_df(
    path: str, query: str | None = None, spacer_col: str = "spacer_20mer"
) -> pd.DataFrame:
    """Read the design DataFrame from a CSV file.

    :param path: Path to the design CSV file.
    :param query: Query string to filter design data frame
    :param spacer_col: Index column
    :return: A pandas DataFrame containing the design information.
    """

    design_df = pd.read_csv(path, index_col=spacer_col)
    if query is not None:
        design_df = design_df.query(query)
    return design_df


def run_report(arguments: argparse.Namespace) -> None:
    """Create a report from the analysis pipeline output.

    :param arguments: Arguments parsed by argparse.
    """
    analysis_dir = arguments.analysis_dir
    output = arguments.output
    design_csv = arguments.design_csv
    design_query = arguments.design_query
    design_spacer_col = arguments.design_spacer_col
    min_total_reads = arguments.min_total_reads
    min_sample_reads = arguments.min_sample_reads
    sample_names = arguments.sample_names
    design_df = read_design_df(design_csv, design_query, design_spacer_col)
    report(
        analysis_dir=analysis_dir,
        design_df=design_df,
        output=output,
        pools=None,
        min_total_reads=min_total_reads,
        min_sample_reads=min_sample_reads,
        sample_names=sample_names,
    )


def run_dialout_pipeline(arguments: argparse.Namespace) -> None:
    """Run the pooled dialout library sequencing data analysis pipeline.

    :param arguments: Arguments parsed by argparse.
    """
    from pysam import Fastafile

    if shutil.which("bwa") is None:
        raise ValueError("Please install bwa")
    fastq_dir = arguments.fastq
    output_dir = arguments.output
    design_csv = arguments.design_csv
    design_query = arguments.design_query
    design_spacer_col = arguments.design_spacer_col
    fasta = arguments.fasta
    save_unaligned_fasta = arguments.save_unaligned_fasta
    fasta_fs, _ = fsspec.core.url_to_fs(fasta)
    fasta_protocol = _get_fs_protocol(fasta_fs)
    local_path = None
    if fasta_protocol != "file":
        fd, local_path = tempfile.mkstemp(suffix=".fa")
        os.close(fd)
        fasta_fs.get(fasta, local_path)
        fasta = local_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_start = None
    n_end = None
    with Fastafile(fasta) as fasta_file:
        for ref in fasta_file.references:
            seq = fasta_file.fetch(ref)
            _n_start = seq.index("N")
            _n_end = seq.rindex("N")
            if n_start is None:
                n_start = _n_start
                n_end = _n_end
            assert n_start == _n_start, "Start of N for all references must be the same"
            assert n_end == _n_end, "End of N for all references must be the same"

    design_df = read_design_df(design_csv, design_query, design_spacer_col)
    fastq_fs, _ = fsspec.core.url_to_fs(fastq_dir)
    fastq_dir = fastq_dir.rstrip(fastq_fs.sep)
    fastq_paths = [
        path
        for path in fastq_fs.glob(f"{fastq_dir}{fastq_fs.sep}*")
        if path.lower().endswith((".fastq", ".fastq.gz"))
    ]

    if len(fastq_paths) == 0:
        raise ValueError("No FASTQ files found")
    fastq_protocol = _get_fs_protocol(fastq_fs)
    if fastq_protocol != "file":
        fastq_paths = [f"{fastq_protocol}://{m}" for m in fastq_paths]
    bwa_index(fasta)
    bwa_threads = 1
    bag = from_sequence(fastq_paths).map(
        single_pool,
        output_dir=output_dir,
        fasta=fasta,
        bwa_threads=bwa_threads,
        gRNA_start=n_start,
        gRNA_end=n_end,
        design_df=design_df,
        save_unaligned_fasta=save_unaligned_fasta,
    )
    bag.compute()
    if local_path is not None:
        os.remove(local_path)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    """Sets up the argument parsers for the 'dialout' command and its subcommands.

    This internal function configures the command-line interface for the 'dialout' tool,
    adding subparsers for 'analysis' and 'report' commands with their respective
    arguments.

    :param subparsers: The subparsers object from the main parser to which new parsers
        are added.
    :param default_help: Determines whether to include default values in help messages.
    """
    parser = subparsers.add_parser(
        "dialout",
        description="Pooled dialout library sequencing data analysis and report",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # Analysis subcommand
    analysis_parser = subparsers.add_parser(
        "analysis",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
        description="Pooled dialout library sequencing data analysis",
    )
    analysis_parser.add_argument(
        "--fastq", help="Path to FASTQ directory", required=True
    )
    analysis_parser.add_argument(
        "-o", "--output", help="Path to output directory", default="dialout-analysis"
    )
    analysis_parser.add_argument(
        "--fasta",
        help=(
            "FASTA file (template for mapping, including NNNs). May include multiple "
            "templates but the position of NNNs need to match. Example: "
            "TCTTGTGGAAAGGACGAAACACCGNNNNNNNNNNNNNNNNNNNNGTTTTAGAGCTAGAAATAGCAAGTTAAAAT"
        ),
        required=True,
    )
    analysis_parser.add_argument(
        "--design-csv",
        help="Design CSV file.",
        required=True,
    )
    analysis_parser.add_argument(
        "--design-spacer-col",
        help="`Spacer` column in design CSV.",
        default="spacer_20mer",
    )
    analysis_parser.add_argument(
        "--design-query",
        help="Expression to filter rows of design CSV.",
    )
    analysis_parser.add_argument(
        "--save-unaligned-fasta",
        help="Whether to save unaligned reads to FASTA file.",
        action="store_true",
    )

    analysis_parser.set_defaults(func=run_dialout_pipeline)

    # Report subcommand
    report_parser = subparsers.add_parser(
        "report",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
        description="Pooled dialout library sequencing report",
    )
    report_parser.add_argument(
        "--analysis-dir",
        help="Path to analysis output directory.",
        required=True,
    )
    report_parser.add_argument(
        "--output",
        help=(
            "Path to output PDF file or a directory where individual PNG images will "
            "be created."
        ),
        required=True,
    )
    report_parser.add_argument(
        "--design-csv",
        help="Design CSV file.",
        required=True,
    )
    report_parser.add_argument(
        "--design-spacer-col",
        help="`Spacer` column in design CSV.",
        default="spacer_20mer",
    )
    report_parser.add_argument(
        "--design-query",
        help="Expression to filter rows of design CSV.",
    )
    report_parser.add_argument(
        "--min-total-reads",
        help=(
            "Minimum total reads across all samples to include guide in pairwise "
            "sample plot."
        ),
        default=10,
        type=int,
    )
    report_parser.add_argument(
        "--min-sample-reads",
        help="Minimum reads to include guide in individual sample guide rank plot.",
        default=2,
        type=int,
    )
    report_parser.add_argument(
        "--sample-names",
        help="Path to CSV file containing the columns `index` and `sample`.",
    )

    report_parser.set_defaults(func=run_report)
