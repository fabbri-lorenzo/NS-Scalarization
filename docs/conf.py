"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options.  For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

The current configuration assumes that this repository has been checked out
locally with a typical Python package layout, and that you have installed
the required Sphinx dependencies (see the ``README`` in this project for
installation instructions).  If you run the documentation build on an
environment without network access, make sure all required packages are
available.
"""

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root directory to ``sys.path`` so that autodoc can find
# the Python modules without requiring installation.  Adjust the path if
# your source code lives in a different location.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "NS‑Scalarization"
copyright = "2025, Lorenzo Fabbri"
author = "Lorenzo Fabbri"

# The full version, including alpha/beta/rc tags.  You may update this
# manually or import the package to obtain ``__version__``.
release = "0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.  They can be
# extensions coming with Sphinx (named ``'sphinx.ext.*'``) or your custom
# ones.
extensions = [
    "myst_parser",  # Markdown support via MyST
    "sphinx.ext.autodoc",  # Pull docstrings into pages
    "sphinxcontrib.mermaid",  # Render Mermaid diagrams directly
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.napoleon",  # Parse Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx_autodoc_typehints",  # Render type hints nicely
    # Uncomment the following line if you have installed sphinxcontrib-mermaid
    # 'sphinxcontrib.mermaid',    # Render Mermaid diagrams directly
]

# Generate autosummary pages automatically
autosummary_generate = True

# Napoleon settings to handle Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# List of patterns to ignore when looking for source files.  This pattern
# also affects the ``exclude_patterns`` option in ``sphinx-apidoc``.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Exclude the incomplete or work-in-progress modules
    "Utils/scan.py",
]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.  The Read the Docs theme is a popular choice
# that produces clean, responsive pages.  You can switch to another theme
# such as ``'furo'`` or ``'alabaster'`` by changing this variable.
html_theme = "sphinx_rtd_theme"

# -- Options for HTML output -------------------------------------------------

# Add paths that contain custom static files (such as style sheets) here,
# relative to this directory.  They are copied after the builtin static
# files, so a file named ``default.css`` will override the builtin theme's
# CSS.
html_static_path = ['_static']

# If you choose to enable the ``sphinxcontrib.mermaid`` extension, you can
# configure it here.  For example, you can set the output format and enable
# live editing in the browser.  See the extension documentation for details.
mermaid_output_format = "png"
