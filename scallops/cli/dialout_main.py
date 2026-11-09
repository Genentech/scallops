from __future__ import annotations

from scallops.cli.args import SubParsers, add_command, lazy_handler


def _create_parser(subparsers: SubParsers, default_help: bool) -> None:
    """Sets up the argument parsers for the 'dialout' command and its subcommands.

    This internal function configures the command-line interface for the 'dialout' tool, adding
    subparsers for 'analysis' and 'report' commands with their respective arguments.

    :param subparsers: The subparsers object from the main parser to which new parsers are added.
    :param default_help: Determines whether to include default values in help messages.
    """
    parser = add_command(
        subparsers,
        "dialout",
        default_help,
        help="Pooled dialout library sequencing data analysis and report",
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # Analysis subcommand
    analysis_parser = add_command(
        subparsers,
        "analysis",
        default_help,
        help="Pooled dialout library sequencing data analysis",
    )
    required = analysis_parser.add_argument_group("required arguments")
    required.add_argument("--fastq", help="Path to FASTQ directory", required=True)
    required.add_argument(
        "--fasta",
        help=(
            "FASTA file (template for mapping, including NNNs). May include multiple templates but the position of "
            "NNNs need to match. Example: "
            "TCTTGTGGAAAGGACGAAACACCGNNNNNNNNNNNNNNNNNNNNGTTTTAGAGCTAGAAATAGCAAGTTAAAATA"
        ),
        required=True,
    )
    required.add_argument(
        "--design-csv",
        help="Design CSV file",
        required=True,
    )

    analysis_parser.add_argument(
        "-o", "--output", help="Path to output directory", default="dialout-analysis"
    )

    analysis_parser.add_argument(
        "--design-spacer-col",
        help="`Spacer` column in design CSV",
        default="spacer_20mer",
    )
    analysis_parser.add_argument(
        "--design-query",
        help="Expression to filter rows of design CSV",
    )
    analysis_parser.add_argument(
        "--save-unaligned-fasta",
        help="Whether to save unaligned reads to FASTA file",
        action="store_true",
    )
    analysis_parser.set_defaults(
        func=lazy_handler("scallops.cli.dialout:run_dialout_pipeline")
    )

    # Report subcommand
    report_parser = add_command(
        subparsers,
        "report",
        default_help,
        help="Pooled dialout library sequencing report",
    )
    required = report_parser.add_argument_group("required arguments")
    required.add_argument(
        "--analysis-dir",
        help="Path to analysis output directory",
        required=True,
    )
    required.add_argument(
        "-o",
        "--output",
        help=(
            "Path to output pdf file or a directory where individual png images will be created"
        ),
        required=True,
    )
    required.add_argument(
        "--design-csv",
        help="Design CSV file",
        required=True,
    )
    report_parser.add_argument(
        "--design-spacer-col",
        help="`Spacer` column in design CSV",
        default="spacer_20mer",
    )
    report_parser.add_argument(
        "--design-query",
        help="Expression to filter rows of design CSV",
    )
    report_parser.add_argument(
        "--min-total-reads",
        help="Minimum total reads across all samples to include guide in pairwise sample plot",
        default=10,
        type=int,
    )
    report_parser.add_argument(
        "--min-sample-reads",
        help="Minimum reads to include guide in individual sample guide rank plot",
        default=2,
        type=int,
    )
    report_parser.add_argument(
        "--sample-names",
        help="Path to CSV file containing the columns `index` and `sample`",
    )
    report_parser.set_defaults(func=lazy_handler("scallops.cli.dialout:run_report"))
