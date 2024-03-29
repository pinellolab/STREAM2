# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "stream2"
copyright = "2021, huidong chen"
author = "huidong chen"

# The full version, including alpha/beta/rc tags
release = "v0.1.0"

# -- Retrieve notebooks -------------------------------

from urllib.request import urlretrieve  # noqa: E402

notebooks_url = "https://github.com/pinellolab/STREAM2_tutorials/raw/main/tutorial_notebooks/"  # noqa
notebooks_v1_0 = [
    "complex_structure.ipynb",
    "supervision_ordinal.ipynb",
    "supervision_categorical.ipynb",
    "multiomics.ipynb",
    "stream_plots.ipynb",
]

for nb in notebooks_v1_0:
    try:
        urlretrieve(notebooks_url + nb, nb)
    except Exception:
        pass

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

github_repo = 'stream2'
github_nb_repo = 'stream2_tutorials'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
