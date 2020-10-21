#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Bartolina Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install Bartolina
"""


# =============================================================================
# IMPORTS
# =============================================================================

# import os
# import pathlib

# from ez_setup import use_setuptools
# use_setuptools()

from setuptools import setup


# =============================================================================
# CONSTANTS
# =============================================================================

# REQUIREMENTS = ['https://github.com/tmcclintock/cluster_toolkit.git']

# PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# with open(PATH / "README.md") as fp:
#     LONG_DESCRIPTION = fp.read()

# with open(PATH / "grispy" / "__init__.py") as fp:
#     for line in fp.readlines():
#         if line.startswith("__version__ = "):
#             VERSION = line.split("=", 1)[-1].replace('"', '').strip()
#             break


# DESCRIPTION = "Grid Search in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="bartolina",
        author=["Noelia Rocío Perez", "Claudio Antonio Lopez Cortez"],
        keywords=["space redshift"],
        dependency_links=['https://github.com/tmcclintock/cluster_toolkit.git']

    )

#        install_requires=['https://github.com/tmcclintock/cluster_toolkit.git']
# version="0.0.1",
# description=DESCRIPTION,
# long_description=LONG_DESCRIPTION,
# long_description_content_type='text/markdown',
# author_email="tinchochalela@gmail.com",
# url="https://github.com/mchalela/GriSPy",
# license="MIT",
# classifiers=[
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Education",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Operating System :: OS Independent",
#     "Programming Language :: Python",
#     "Programming Language :: Python :: 3.8",
#     "Programming Language :: Python :: Implementation :: CPython",
#     "Topic :: Scientific/Engineering"],

# packages=["grispy"],
# py_modules=["ez_setup"],

# install_requires=REQUIREMENTS)


if __name__ == "__main__":

    do_setup()
