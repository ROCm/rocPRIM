# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'rocPRIM'
# copyright = ''
author = ''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# furo is not the theme chosen by other ROCM project
# but it works fine too
# html_theme = "furo"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']
html_css_files = [
    "cpp_sig.css"
]

primary_domain = "cpp"
highlight_language = "cpp"

pygments_style = "sphinx"
pygments_dark_style = "material"

# Make sphinx not choke on CUDA / HIP attributes
cpp_id_attributes = [
    "__device__",
    "__global__",
    "__host__"
]
cpp_index_common_prefix = ["rocprim::"]

breathe_projects = {
    "rocprim": "../xml",
}
breathe_default_project = "rocprim"

html_show_copyright = False
html_show_sphinx = False
html_copy_source = False
