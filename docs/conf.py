"""Sphinx configuration."""

project = "Piezo_Load_Voltage_Processing"
author = "Kailer Okura"
copyright = "2025, Kailer Okura"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
