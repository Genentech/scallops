# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import scallops  # noqa


project = "scallops"
copyright = "2026, Genentech"
author = "scallops team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "myst_parser",  # allow md files
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "sphinx.ext.autosectionlabel",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

suppress_warnings = [
    "nbsphinx",
]
autodoc_default_options = {"members": True, "member-order": "bysource"}
autodoc_typehints = "description"
autosummary_generate = True
todo_include_todos = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# intersphinx_mapping = dict(
#     matplotlib=("https://matplotlib.org/stable/", None),
#     numpy=("https://numpy.org/doc/stable/", None),
#     pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
#     pytest=("https://docs.pytest.org/en/latest/", None),
#     python=("https://docs.python.org/3", None),
#     scipy=("https://docs.scipy.org/doc/scipy/", None),
#     seaborn=("https://seaborn.pydata.org/", None),
#     skimage=("https://scikit-image.org/docs/stable/api/", None),
#     sklearn=("https://scikit-learn.org/dev/", None),
#     xarray=("https://docs.xarray.dev/", None),
# )
intersphinx_disabled_reftypes = ["*"]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# Add custom CSS files
html_css_files = [
    "css/custom.css",
]


def skip_private_members(app, what, name, obj, skip, options):
    if name.startswith("_"):
        return True  # Skip this member
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_private_members)


html_logo = "_static/scallopsLogo.png"
