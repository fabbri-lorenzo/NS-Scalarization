# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NS Scalarization'
copyright = '2025, Fabbri L.'
author = 'Fabbri L.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os, sys
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../tests"))

extensions = [
    "myst_parser",                # Markdown support
    "sphinx.ext.autodoc",         # pull docstrings
    "sphinx.ext.autosummary",     # summary pages
    "sphinx.ext.napoleon",        # Google/NumPy docstrings
    "sphinx_autodoc_typehints",   # show type hints nicely
    "sphinx.ext.viewcode",        # link to highlighted source
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring  = True

# For Mermaid Workflow Diagram
mermaid_output_format = "png"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']
