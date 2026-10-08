"""The version string lives in pyproject.toml and in the package. Keep them equal."""

import tomllib
from pathlib import Path

import embedpipe

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_version_matches_pyproject():
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert embedpipe.__version__ == declared
