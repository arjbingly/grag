Deploying Sphinx Documentation
##############################

Making a documentation website for your python project can be daunting and usually requires some web-dev
skills. But Sphinx can be used to generate documentations for your project - it can automatically convert your docstrings
and in-code documentation into various output formats like HTML, PDF, LaTeX, man-pages, etc.

This guide will show you the framework to easily create and streamline documentation for your Python project on a Git repository, using Sphinx.

The tutorial result should look like `this <https://elpham6.github.io/sphinx_demo/>`_.

Tutorial
********

1. Install Sphinx
==================

- For Debian/Ubuntu/Windows Subsystem for Linux (WSL), run ``sudo apt install python3-sphinx``
- For Anaconda, open the Anaconda terminal and run ``conda install sphinx``

To ``pip`` install:

- For Linux, macOS or Windows, run ``pip install -U sphinx``

For installation options not mentioned, refer to the `documentation <https://www.sphinx-doc.org/en/master/usage/installation.html>`_ for more information.

2. Create a Git repository
==========================

1. Create a Git repository using this `template <https://github.com/new?template_name=sphinx_template&template_owner=elpham6>`_. Put whatever name you would like for the Repository name, and click Create repository.

2. The repo's folder structure should have a ``docs``, ``src`` and ``examples`` folder.

``docs`` contain your files for Sphinx documentation. ``src`` should contain your code.
``examples`` contain some example modules.

In this guide, the ``src`` folder should have a **calculator.py** and **helloworld.py** file, and the ``examples`` folder has a **calc_example.py** file for demo purposes.

3. Run ``git clone <repo link>`` to clone the repo you just made.

3. Setting Up Documentation Sources
====================================

- In the repo, change your directory location to the ``docs`` folder (run ``cd docs``).

- Run ``sphinx-quickstart``. You will be prompted with the following:

1. **Separate source and build directories (y/n) [n]**: Select `n`.

2. **Project name**: enter your project name here

3. **Author name(s)**: enter your name(s) here

4. **Project release**: enter the version number of your project here

5. **Project language [en]**: enter the language for your project here. Default is English (`en`).

This will create a default configuration ``conf.py`` file, the make files and ``_build``, ``_static``, ``_templates`` folders.

4. Creating Documentation From Modules
======================================

- Go back to the root directory (run ``cd ..``).

- Run ``sphinx-apidoc -o docs src``.

Here, ``docs`` is for the output directory where all your documentation goes, and ``src`` is the directory with all the modules you want to document.

This will create .rst files for each Python module in ``src``.

5. Config Specifications
========================

Open **conf.py** in the ``docs`` folder. You can see all the default configurations here.

5.1. OS Syspath Change
-------------------------

Add the following to the beginning of **conf.py**:

.. code-block:: python

    import os
    import sys
    sys.path.insert(0, os.path.abspath("../src"))

**Note**: This assumes the folder structure of the template repo. If you have a different structure, make sure to replace ``"../src"`` with the path to the code you would like to make documentation for.

5.2. Adding Extensions
-----------------------

Sphinx has many useful extensions, which you can check out `here <https://www.sphinx-doc.org/en/master/usage/extensions/index.html>`_.

For this tutorial, add the following extensions to the ``extensions`` list in **conf.py**:

.. code-block:: python

    extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.linkcode",
      ]

- ``sphinx.ext.autodoc``: automatically takes doc strings from your Python files

- ``sphinx.ext.napoleon``:  understand NumPy or Google doc string standards and format them nicely. If you write your doc strings using Numpy or Google standard, you need this extension.

  Since this example uses Google-style doc string, add:

    ``napoleon_google_docstring = True``

  to the **conf.py** file.

- ``sphinx.ext.linkcode``: provides a link to the source code on GitHub. Note that this requires more config specifications, which you can refer to `here <https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html>`_.

  This guide assumes we want to get HTML output. Add the following dictionary ``html_context`` to pass to ``linkcode`` config:

.. code-block:: python

    html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "user_name",  # Username of repo's owner
    "github_repo": "sphinx_demo",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/src/",  # Path in the checkout to the code's root
    }



Make sure to change "user_name" to your username or the name of the repo owner.

Then, add the ``linkcode`` settings:

.. code-block:: python

    def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    # return "https://somesite/sourcerepo/%s.py" % filename
    # link to the source module/code on github
    return f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}/blob/{html_context['github_version']}/{html_context['conf_py_path']}/{filename}.py"

Adjust the config according to your folder structure and names. If you would like an output other than HTML, refer to Sphinx's `configuration documentation <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_.


5.3 Theme (Optional)
--------------------

The default theme for the output is Alabaster.

This tutorial uses a Sphinx theme called `Read the Docs <https://sphinx-themes.org/sample-sites/sphinx-rtd-theme/>`_, which has a much better format than the default.

1. To install, run ``pip install sphinx-rtd-theme``.

2. In the **conf.py** file, change to ``html_theme = sphinx_rtd_theme``.

  You can find more themes at various sources like `www.sphinx-themes.org/`,
  `https://sphinxthemes.com`, etc.

6. Building Documentation
=========================

1. Change directory to the ``docs`` folder.

2. Run ``make html``. The result will be in **docs/_build/html**.

3. To preview your website, go to **docs/_build/html**. Open **index.html**, which shows you the homepage for your documentation.

4. If you make any changes to your code or documentation, simply run ``make html`` again from the **docs** folder to update your documentation.

7. Deploying to GitHub Pages
============================

To automatically update the documentation on the website whenever you update your work, one way to do it is set up GitHub Action to trigger every time you push changes to the **main** branch of your repo.
This streamlines the process of keeping your documentation up-to-date.

7.1. Enable GitHub Pages
-------------------------
1. In your GitHub repository, click on **Settings**.

2. On the menu, under "Code and automation", click on **Pages**.

3. In the "Source" drop down menu, choose "GitHub Actions".

7.2. Set Up GitHub Actions
--------------------------

1. Move to the root directory of the repo.

2. Create a folder called ``.github``. Then within the folder, create another folder called ``workflows``.

3. Move to ``.github/workflows/``.

4. Create a .yml file, name it "sphinx-gitpg.yml".

5. To set up the configuration for the GitHub Action, copy and paste the following into the .yml file:

.. code-block:: yaml

    name: Docs build and upload

    on:
      push:
        branches:
          - main

      workflow_dispatch:

    permissions:
      contents: read
      pages: write
      id-token: write

    concurrency:
      group: "pages"
      cancel-in-progress: false

    jobs:
      docs:
        environment:
          name: github-pages
          url: ${{ steps.deployment.outputs.page_url }}
        runs-on: ubuntu-latest
        steps:
          - name: Checkout
            uses: actions/checkout@v4

          - name: Setup Python
            uses: actions/setup-python@v5
            with:
              python-version: '3.11'
          - name: Setup Sphinx
            run: |
              pip install sphinx sphinx_rtd_theme sphinx_gallery
          - name: Sphinx Build
            run: |
              cd 'docs'
              make html

          - name: Setup Pages
            uses: actions/configure-pages@v5

          - name: Upload GitHub Pages Artifact
            uses: actions/upload-pages-artifact@v3
            with:
              path: "docs/_build/html"

          - name: Deploy GitHub Pages
            id: deployment
            uses: actions/deploy-pages@v4

This makes sure that the documentation will be built and updated onto the GitHub page url only when you push changes on to your **main** branch.
You no longer need a ``_build`` folder at this point, as the .yml script performs this action automatically every time you push to **main**, then uploads the content of ``_build/html`` to the website.

If you add any more Sphinx extensions that needs to be installed, simply add the dependency to the "Setup Sphinx" step in the .yml file.

For example, ``pip install sphinx sphinx_rtd_theme`` means that the action will install sphinx, and sphinx_rtd_theme.

7.3. Check the Documentation Results
-------------------------------------
To check the result, go to https://user_name.github.io/sphinx_demo/, replace user_name with your GitHub username.

Also, if something fails, you can click on the "Actions" tab from the repository, and check for the error.

Now, if you make any changes and then push to the **main** branch of the repository, the website will automatically update the documentation.

8. Adding Content (optional)
============================

The default options in Sphinx produce a nice template, but you want to add and adjust content in order to produce a better website.

To add other pages to your Sphinx website, simply create `.rst` files in ``docs``, then add them to the ``toctree`` of ``index.rst``, or to the ``toctree`` of a file listed/included in ``index.rst``.

Below are some examples of what you can add to the documentation.

8.1. Adding Content on Homepage
----------------------------------

By default, when you view your homepage, you will only see the index menu and not the content of your code.
To add more pages:

1. Open ``index.rst`` and manually add `.rst` file names to the Contents of ``toctree``:

.. code-block:: rst

    .. toctree::
      :maxdepth: 4
      :caption: Contents:

      calculator

      helloworld

2. In the ``docs`` folder, run ``make html`` again.

3. Go to ``docs/_build/html`` and view the results. You will see the homepage showing the ``calculator`` and ``helloworld`` modules' content. You can also move back and forth between the sections of the documentation using the "Next" or "Previous" buttons.

8.2. Adding Another Section
---------------------------

Let's add a section called Demo Modules Overview, where we can write more explanation on the code.

1. In ``docs``, create a file called **overview.rst**, **overview.calculator.rst** and **overview.helloworld.rst**. For this turorial, simply copy the files from the ``docs`` folder of `this <https://github.com/elpham6/sphinx_demo/tree/main/docs>`_ repo to your own repo.

2. To add this section to the website, open **index.rst**. Add **overview** to the top of the ``toctree``.

.. code-block:: rst

    .. toctree::
      :maxdepth: 4
      :caption: Contents:

      overview

      calculator

      helloworld

3. Run ``make html`` from ``docs`` again.

4. Go to ``docs/_build/html`` and view the results.

You will see a new section called "Demo Modules Overview" with an index, showing content from **overview.calculator.rst** and **overview.helloworld.rst**.
It is easy to add new pages and new sections to the website.

8.3. Adding Examples
---------------------

Let's create a section for some example codes. We will use ``sphinx_gallery`` extension here.

1. Install sphinx_gallery: ``pip install sphinx_gallery``.

2. Open **conf.py**. Add "sphinx_gallery.gen_gallery" to the ``extensions`` list.

3. Add the sphinx_gallery config to **conf.py**:

   .. code-block:: python

      sphinx_gallery_conf = {
          # path to your example scripts
          'examples_dirs': ['../examples'],
          # path to where to save gallery generated output
          'gallery_dirs': ['auto_examples'],
          'filename_pattern': '.py',
          'plot_gallery': 'False',
      }

4. In the ``examples`` folder, create a **README.rst** or **README.txt** file.
   A readme file is necessary for sphinx_gallery to generate documentation. Copy/paste this text:

   .. code-block:: rst

      Calculator Examples
      ###################

      This folder contains example code for the **calculator.py** module.

5. From ``docs``, run ``make html``. There is a new folder called ``auto_examples`` created in ``docs``, with all the generated
   documentation for modules in the ``examples`` folder.


6. In ``docs/index.rst``, add the new automatically created index file:

.. code-block:: rst

    .. toctree::
      :maxdepth: 4
      :caption: Contents:

      overview

      calculator

      helloworld

      auto_examples/index

7. From ``docs``, run ``make html`` again.

You can now see the example code, with links to download the module.

**Note**: the docstring at the top of **calc_example.py** is in .rst format. That is because Sphinx automatically generates a .rst file from the .py file.
You can see that this docstring is formatted into the page. This means you can add other things, such as diagrams here as well.

There are a lot of other things you can do with Sphinx to customize your documentation website.

* For more instructions on defining document structure, refer to
  `Defining Docuement Structure <https://www.sphinx-doc.org/en/master/usage/quickstart.html#defining-document-structure>`_.

* For instructions on how to format reStructuredText, refer to
  `reStructuredText Basics <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.


Notes
****************
To ensure a better result:

* Have proper documentation for your code. This includes doc strings.
* Make sure that your doc strings follow a standard, eg. PEP, Google, Numpy, etc. This guide followed `Google doc string conventions <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
* Highly recommended to use a linter for both your code and docs, like `Ruff <https://docs.astral.sh/ruff/#testimonials>`_.


