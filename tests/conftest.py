# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import os
import pathlib

from astropy.table import Table

import bartolina

import joblib

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
# FIXTURES
# ============================================================================

# FULL


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


# sample


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
