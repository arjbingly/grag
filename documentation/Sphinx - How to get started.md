# Sphinx - How to get started?

## Prereqs

- You have proper documentation for your code. This includes doc strings.
- Make sure that your doc strings follow a standard, eg. PEP, Google, Numpy, etc.
- Highly recommended to use a linter for both your code and docs. I use Ruff.

## 1. Install Sphinx

*refer: https://www.sphinx-doc.org/en/master/usage/installation.html*

- **for Ubuntu/Debian** : `sudo apt install python3-sphinx`

## 1.1 Folder Structure (Recommended)

I like my docs as a separate folder in `src`. Therefore I create a new directory `docs`. Move here for the next step.

```
├── src
│   ├── build
│   ├── package_dir
│   │   ├── __init__.py
│   │   ├── module_1.py
│   │   ├── module_2.py
│   │   ├── subpackage_dir
│   │   │   ├── __init__.py
│   │   │   ├── module_1.py
│   │   │   ├── module_2.py
├── other directories and files not documented
│   ├── css
│   │   ├── **/*.css
│   ├── images
│   ├── js
│   ├── index.html
├── pyproject.toml
├── requirements.txt
├── package-lock.json
└── .gitignore

```

## 2. Quickstart

Run:

```
sphinx-quickstart
```

This will ask you the following:

1. **Do you wanna separate your build and source?** N, Since I recommend a folder structure with a separate docs
   directory.
2. Name of Project
3. Author Name
4. Project Release
5. Project Language

If run successfully this will create a `conf.py` file, the make files and folders like `_build`, `_static`,
and `_template`.

## 3. Extract docs

Move to `src` dir.  
Run:

```
sphinx-apidoc -o docs package_dir
``` 

Note that: here `-o` flag is the output folder, in our structure it is the `docs` directory, and `package_dir` is the
directory with all the code you want to document.

This will create `.rst` files for each Python module.

## 4. Conf.py changes

The `conf.py` file is in `docs` folder.
You can see all the details you specified in the `quickstart` here.

## 4.1 OS Syspath Changes

Add the following to the beginning of `conf.py`.

```
import os
import sys
sys.path.insert(0, os.path.abspath("../package_dir"))
``` 

*Note that this assumes the above-mentioned folder structure, if you have a different structure, make sure to point it
to the source code. *

## 4.3 Extensions

Sphinx has a lot of useful extensions. These should be added to the `extensions` tag. Some of the extensions I use are:

1. `sphinx.ext.autodoc` - This extension automatically takes doc strings.
2. `sphinx.ext.linkcode` (Optional) - This extension provides a link to the GitHub code. *(Note that this extension
   requires other configs. Refer to Sphinx extension documentation for more details.)*
3. `sphinx.ext.viewcode` (Optional) - This extension is similar to the above extension, instead of linking the code to
   GitHub, it displays the code in a static webpage.
4. `sphinx.ext.napoleon` (Optional) - Since we use Google doc-string standard, this is an essential extension.

## 4.2 Theme (Optional)

I use a Sphinx theme, which can be installed by running,

``` 
pip install sphinx-rtd-theme 
```

Change the `html_theme` tag in the `conf.py` to `sphinx_rtd_theme`.

You can find more themes at various sources like `www.sphinx-themes.org/`,
`https://sphinxthemes.com`, etc.

## 5 Building the docs

To finally generate the docs run the following command from the `docs` directory.

```
make html
```

This will create a `build` directory, where you can find the html files. Opening the `index.html` shows you the homepage
of your docs.

## 6.1 Adding other pages (Optional)

To add other pages to your sphinx website, you just have to create `.rst` reStructuredText files in the appropriate
location and add them to your `index.rst` or to the `toctree` of a file already mentioned in `index.rst`.

For more instructions on defining document structure refer
[Defining Docuement Structure](https://www.sphinx-doc.org/en/master/usage/quickstart.html#defining-document-structure)

For instructions on how to format reStructuredText refer to
[reStructuredText Basics](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

## 6.2 Adding examples (Optional)

TODO

- [https://sphinx-gallery.github.io/stable/index.html]()

### References

- [https://www.sphinx-doc.org/en/master/usage/quickstart.html]()
- [https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/]()
