# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add parent directory to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "LMMs Engine"
copyright = "2024, LMMs Engine Contributors"
author = "LMMs Engine Contributors"
version = "1.0"
release = "1.0.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
# Use Google style docstrings instead of NumPy docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# MyST parser configuration for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Add markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Theme configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "style_external_links": False,
    "vcs_pageview_mode": "edit",
    "style_nav_header_background": "#2980B9",
}

# Output options
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Internationalization
language = "en"

# LaTeX output
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "12pt",
}

# The master toctree document
master_doc = "index"
