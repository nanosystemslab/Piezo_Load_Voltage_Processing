"""Nox sessions."""

import os
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox


try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None


package = "Piezo_Load_Voltage_Processing"
python_versions = ["3.12", "3.11", "3.10"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = ("docs-build", "docs", "mypy")

@session(python=python_versions)
def mypy(session: Session) -> None:
    """Run mypy static type checks."""
    session.install("mypy")
    session.run("mypy", "src/")

@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install("sphinx", "sphinx-argparse", "furo", "myst-parser", "sphinx-click")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)

@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    session.install(
        "sphinx", "sphinx-autobuild", "sphinx-argparse", "furo", "myst-parser"
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)

@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Run Typeguard to check runtime type correctness."""
    session.install("typeguard")
    session.install(".")  # Ensure your package is installed for type checking
    session.run("pytest", "--typeguard-packages=AM_Creep_Analysis")
