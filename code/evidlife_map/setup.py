"""Setup script for EvidLife-Map (P1 skeleton).

For the editable install used in W1 bring-up:

    pip install -e .

Dependencies are pinned in ``requirements.txt`` to keep the Docker image
reproducible. ``pyproject.toml`` carries the tool configuration (ruff, black,
mypy, pytest).
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

_THIS = Path(__file__).resolve().parent

with (_THIS / "requirements.txt").open(encoding="utf-8") as fh:
    _INSTALL_REQUIRES = [
        line.strip()
        for line in fh
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="evidlife_map",
    version="0.1.0a1",
    description=(
        "EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping "
        "with Voxel Decay (IROS 2027 submission, P1 skeleton)."
    ),
    long_description=(_THIS / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="EvidLife-Map authors",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*", "experiments", "scripts")),
    install_requires=_INSTALL_REQUIRES,
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
