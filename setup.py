#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install Bartolina
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from ez_setup import use_setuptools

use_setuptools()

from setuptools import setup  # noqa


# =============================================================================
# CONSTANTS
# =============================================================================

# Short for cluster_toolkit
CLTK = {
    "name": "cluster_toolkit",
    "url": "https://github.com/tmcclintock/cluster_toolkit/",
}

REQUIREMENTS = [
    "numpy",
    "astropy",
    "attrs",
    "camb",
    "sklearn",
    "halotools",
    "pandas",
    "cython",
    "mpsort",
    "mpi4py",
    "pfft-python",
    "pmesh",
    f"{CLTK['name']} @ git+{CLTK['url']}@master#egg={CLTK['name']}",
]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "bartolina" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="bartolina",
        version=VERSION,
        description="Corrections for the redshift distortion",
        author=["Noelia Rocío Perez", "Claudio Antonio Lopez Cortez"],
        url="https://github.com/exiliadadelsur/Bartolina",
        license="MIT",
        keywords=["space redshift", "kaiser", "finger of god", "fog"],
        packages=["bartolina"],
        py_modules=["ez_setup"],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":

    do_setup()
