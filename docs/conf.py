"""Sphinx configuration for the fronttrack documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

project = "fronttrack"
author = "N Surya"
copyright = "2026, N Surya"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = []
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Figures in the examples are produced by running their code at build time.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]
plot_rcparams = {"figure.figsize": (7.5, 3.4), "font.size": 9}
plot_pre_code = "import numpy as np\nimport matplotlib.pyplot as plt\nfrom fractions import Fraction"

html_theme = "furo"
html_title = "fronttrack"
html_theme_options = {
    "source_repository": "https://github.com/surya-aero23/front-tracking-burgers",
    "source_branch": "main",
    "source_directory": "docs/",
}
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
