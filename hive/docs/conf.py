# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_redactor_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# -*- coding: utf-8 -*-
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("../app"))
sys.path.append(os.path.abspath("../app/utils"))
sys.path.append(os.path.abspath("../app/domain"))
sys.path.append(os.path.abspath("../app/domain/helpers"))

# -- Project information -----------------------------------------------------

project = "Hives"
author = "Francisco Barros"
copyright = "2020, Francisco Barros"
version = "1.6"
release = "1.6.0rc1"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_future_annotations",
]

intersphinx_mapping = {
    "pd": ("https://pandas.pydata.org/docs/", None),
    "np": ("https://numpy.org/doc/stable/", None),
}
# show type hints in doc body instead of signature
autodoc_typehints = "description"  # signature, description, none
autoclass_content = "class"  # class, init, both

autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',  # alphabetical, groupwise, bysource
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = [".rst", ".md"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {
#     "stickysidebar": True
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names. The default sidebars (for documents that don't match any
# pattern) are # defined by theme itself.
#
# html_sidebars = [
#     "localtoc.html",
#     "relations.html",
#     "sourcelink.html",
#     "searchbox.html"
# ]

# -- Sphinx Themes -----------------------------------------------------------

html_theme = "sphinx_redactor_theme"
html_theme_path = [sphinx_redactor_theme.get_html_theme_path()]
html_show_sourcelink = False
html_title = "Hives - P2P Stochastic Swarm Guidance Simulator"
html_short_title = "Hives"
html_favicon = "_static/img/bee32.ico"
html_add_permalinks = ""
