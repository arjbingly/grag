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
import os
import sys

sys.path.insert(0, os.path.abspath('../grag'))

# -- Project information -----------------------------------------------------

project = 'GRAG'
copyright = '2024, Arjun Bingly, Sanchit Vijay, Erika Pham, Kunal Inglunkar'
author = 'Arjun Bingly, Sanchit Vijay, Erika Pham, Kunal Inglunkar'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "arjbingly",  # Username
    "github_repo": "Capstone_5",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/src/",  # Path in the checkout to the docs root
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    # "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    # "sphinx_github_style"
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autosectionlabel"
]

sphinx_gallery_conf = {
    'examples_dirs': ['../../cookbook/Basic-RAG', '../../cookbook/Retriver-GUI'],  # path to your example scripts
    # 'examples_dirs': '../../cookbook',
    'gallery_dirs': ['auto_examples/Basic-RAG', 'auto_examples/Retriver-GUI'],
    # path to where to save gallery generated output
    'filename_pattern': '.py',
    'plot_gallery': 'False',
}


# Linkcode settings
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    # return "https://somesite/sourcerepo/%s.py" % filename
    return f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}/blob/{html_context['github_version']}/{html_context['conf_py_path']}/{filename}.py"


# Napoleon settings
napoleon_google_docstring = True

# Autosectionlabel settings
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

