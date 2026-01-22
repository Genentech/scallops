.. image:: _static/scallopsLogo.png
   :width: 150px
   :align: center


########
SCALLOPS
########

.. raw:: html

    <p style="line-height: 1.6; margin-bottom: 1em;">
      SCALLOPS (<b>Scal</b>able <b>L</b>ibrary for <b>O</b>ptical <b>P</b>ooled <b>S</b>creens) is a comprehensive Python
      package designed to streamline and scale the analysis of Optical Pooled Screens (OPS) for biological data.
      With a focus on handling large-scale, high-throughput screening data, SCALLOPS provides tools for efficiently
      processing, analyzing, and interpreting OPS data, leveraging modern distributed computing frameworks like Dask.
    </p>
    <br>


Main Focus Areas:
------------------
High-Throughput Data Processing: SCALLOPS is built to manage massive datasets typical of OPS experiments, allowing users
to efficiently process and analyze data across multiple scales.

Scalability and Performance: The package is optimized for both local and cloud-based distributed
environments, making it ideal for scaling to large datasets without compromising performance.

Modular Workflows:
------------------
SCALLOPS provides an end-to-end WDL workflow that can be customized, allowing users to tailor analyses
to their specific experimental needs.

Key Features:
-------------
Efficient Data Handling:
^^^^^^^^^^^^^^^^^^^^^^^^

SCALLOPS utilizes advanced memory management and lazy evaluation techniques, which minimize resource usage
while handling large datasets.

Command-Line Interface:
^^^^^^^^^^^^^^^^^^^^^^^
A user-friendly command-line interface (CLI) enables automation and batch processing, making it easy to
integrate SCALLOPS into larger pipelines.

Customizable Outputs:
^^^^^^^^^^^^^^^^^^^^^
The package can generate versatile outputs, including data visualizations and summary statistics, which
can be easily integrated into downstream analyses.

Notebook Examples:
^^^^^^^^^^^^^^^^^^
SCALLOPS includes practical Jupyter notebooks that walk users through typical workflows, making it easy to
get started with real-world datasets.

Custom Features:
^^^^^^^^^^^^^^^^
Advanced users can extend SCALLOPS with their own custom functions and workflows, ensuring the package can
grow with the complexity of the data.

Comprehensive API:
^^^^^^^^^^^^^^^^^^
SCALLOPS provides a rich API that exposes all the package functionalities, allowing users to integrate it
directly into their own Python scripts and workflows. This makes SCALLOPS highly adaptable, enabling
users to build fully customized data pipelines and analyses tailored to their unique experimental needs.

Typical Use Cases:
------------------
Large-Scale Screening Projects:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SCALLOPS is designed for handling the immense data loads of genome-wide OPS projects, helping users to
efficiently identify and quantify biological perturbations.

Data-Driven Insights:
^^^^^^^^^^^^^^^^^^^^^
SCALLOPS facilitates the discovery of patterns and trends in OPS data, helping users extract and interpret
complex biological systems data.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


   install
   command_line
   example_commands
   workflows
   example_notebooks
   faq
   api
