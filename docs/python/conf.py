"""Sphinx configuration of the documentation of the Python package ``comrs``.

Build with ``just docs-py`` (or ``sphinx-build -b html docs/python docs/python/_build/html``)
after ``maturin develop --release``: the API pages are generated from the docstrings of
the compiled extension, and the gallery runs every script of ``examples/``.
"""

import sys
from pathlib import Path

import comrs

# `reset_style` below is referenced by name by the gallery; the examples that read the
# specifications of the repository import `_gallery_utils`; the style and the palette
# are those of `comrs.plot`, the themes those of `tools/_common.py`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "tools"))

from _common import GALLERY_THEMES  # noqa: E402

project = "comrs"
author = "Vincent Choqueuse"
copyright = "2022-2026, Vincent Choqueuse"
release = comrs.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
]

# the docstrings of the extension write code and formulas between single backticks
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_member_order = "groupwise"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
exclude_patterns = ["_build", "examples", "tutorials"]
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# --- figures: one style for the whole gallery ---------------------------------------


def reset_style(gallery_conf, fname):
    """The style of :func:`comrs.plot.style` from the matplotlib defaults, before every
    example."""
    import matplotlib as mpl

    mpl.rcdefaults()
    comrs.plot.style()


sphinx_gallery_conf = {
    # two galleries: the tutorials (step by step) and the classical results, one
    # sub-gallery per theme (the reproductions of named figures in their theme)
    "examples_dirs": ["tutorials", "examples"],
    "gallery_dirs": ["auto_tutorials", "auto_examples"],
    "filename_pattern": r"/plot_",
    "subsection_order": [f"examples/{t}" for t in GALLERY_THEMES],
    "within_subsection_order": "FileNameSortKey",
    "reset_modules": ("matplotlib", "conf.reset_style"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "image_scrapers": ("matplotlib",),
}

# --- theme --------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"comrs {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {"text": "comrs"},
    "github_url": "https://github.com/com-rs/com-rs",
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_sidebars = {"index": [], "getting_started": [], "performance": []}
html_context = {"default_mode": "light"}
