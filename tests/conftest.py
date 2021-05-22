# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import functools
import os
import pathlib

from astropy.table import Table

import bartolina

import joblib

import numpy as np

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================


PATH = os.path.abspath(os.path.dirname(__file__))

RESOURCES_PATH = pathlib.Path(PATH) / "data"


# =============================================================================
# MARKERS
# =============================================================================

# esto sirve para que puedan hacer
# pytest -m ALL asi corre TODO, por que sino no hay forma de
# correr los unittest y los de integracion juntos
# esencialmente es como ponerle @pytest.mark.ALL a CADA test.
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.ALL)


# ============================================================================
# FIXTURES SLOAN DR12
# ============================================================================


@functools.lru_cache
def _dr12_load():

    # I point to the parts folder
    path = RESOURCES_PATH / "SLOAN_dr12_10"

    # I list all part files in order
    parts_names = sorted(path.glob("part_*.joblib.bz2"))

    # load all files in memory
    parts = map(joblib.load, parts_names)

    # I concatenate all the arrays into one
    arr = np.concatenate(list(parts))

    # I turn it into artropy tables
    table = Table(arr)

    return table


@pytest.fixture
def dr12():
    table = _dr12_load()
    return table.copy()


# =============================================================================
# FIXTURES FULL
# =============================================================================


@pytest.fixture
def full_table():
    table = Table.read(RESOURCES_PATH / "full_SDSS.fits")
    return table


@pytest.fixture
def full_bt(full_table):
    bt = bartolina.ReZSpace(
        full_table["RAJ2000"], full_table["DEJ2000"], full_table["zobs"]
    )
    return bt


@pytest.fixture
def full_dmh():
    halos, galingroups = joblib.load(RESOURCES_PATH / "full_dmh.joblib.pkl")
    return halos, galingroups


# =============================================================================
# FIXTURES SAMPLE
# =============================================================================


@pytest.fixture
def sample_table():
    table = joblib.load(RESOURCES_PATH / "sample_SDSS.joblib.pkl")
    return table


@pytest.fixture
def sample_bt(sample_table):
    bt = bartolina.ReZSpace(
        sample_table["RAJ2000"], sample_table["DEJ2000"], sample_table["zobs"]
    )
    return bt


@pytest.fixture
def sample_dmh():
    halos, galingroups = joblib.load(RESOURCES_PATH / "sample_dmh.joblib.pkl")
    return halos, galingroups
