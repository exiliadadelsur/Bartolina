# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE


"""Bartolina a real space reconstruction algorithm.

Bartolina recreate a real space and corrects galaxy positions by Kaiser and
Finger of God (FoG) effects. We follow the work carried out in
`Shi et al. 2016, ApJ, 833, 241`_ and `Wang et al. 2012, MNRAS, 420, 1809`_.

.. _Shi et al. 2016, ApJ, 833, 241:
    https://iopscience.iop.org/article/10.3847/1538-4357/833/2/241/pdf
.. _Wang et al. 2012, MNRAS, 420, 1809: https://arxiv.org/pdf/1108.1008.pdf

"""

# =============================================================================
# CONSTANTS AND IMPORTS
# =============================================================================

__version__ = "0.2"

__all__ = ["ReZSpace", "FoF"]

from .bartolina import FoF, ReZSpace
