"""Elodie 3.1"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .config import libsdir
from .grid import Grid
from .stellib import Stellib


class Elodie(Stellib):
    """Elodie 3.1 stellar library derived class"""

    def __init__(self, *args, **kwargs):
        self.source = libsdir + "/stellib_ELODIE_3.1.fits"
        self._load_()
        Stellib.__init__(self, *args, **kwargs)
        self.name = "ELODIE v3.1 (Prugniel et al 2007, astro-ph/703658)"

    def _load_(self):
        with pyfits.open(self.source) as f:
            # load data
            self._getWaveLength_(f)
            self._getTGZ_(f)
            self._getSpectra_(f)
            self._getWaveLength_units()

    def _getWaveLength_units(self):
        self.wavelength_unit = "angstrom"

    def _getWaveLength_(self, f: Sequence[pyfits.TableHDU]):
        self._wavelength = np.asarray(f[0].data[-1])

    def _getTGZ_(self, f: Sequence[pyfits.TableHDU]):
        self.grid = Grid(np.array(f[1].data), dict(f[1].header.items()))
        self.grid.header["NAME"] = "TGZ"

    def _getSpectra_(self, f: Sequence[pyfits.TableHDU]):
        self.spectra = f[0].data[:-1]

    def bbox(self, dlogT: float = 0.05, dlogg: float = 0.25) -> npt.NDArray[np.float64]:
        """Boundary of Elodie library

        Parameters
        ----------
        dlogT: float
            log-temperature tolerance before extrapolation limit

        dlogg: float
            log-g tolerance before extrapolation limit

        Returns
        -------
        bbox: ndarray
            (logT, logg) edges of the bounding polygon
        """
        bbox = [
            (3.301 - dlogT, 5.500 + dlogg),
            (3.301 - dlogT, 3.500 - dlogg),
            (3.544 - dlogT, 3.500 - dlogg),
            (3.544 - dlogT, 1.000),
            (3.477, 0.600 + dlogg),
            (3.447 - dlogT, 0.600 + dlogg),
            (3.398 - dlogT, 0.280 + dlogg),
            (3.398 - dlogT, -1.020 - dlogg),
            (3.398, -1.020 - dlogg),
            (3.447, -1.020 - dlogg),
            (3.505 + dlogT, -0.700 - dlogg),
            (3.544 + dlogT, -0.510 - dlogg),
            (3.574 + dlogT, -0.290 - dlogg),
            (3.602 + dlogT, 0.000 - dlogg),
            (3.778, 0.000 - dlogg),
            (3.778 + dlogT, 0.000),
            (3.875 + dlogT, 0.500),
            (3.929 + dlogT, 1.000),
            (3.954 + dlogT, 1.500),
            (4.021 + dlogT, 2.000 - dlogg),
            (4.146, 2.000 - dlogg),
            (4.146 + dlogT, 2.000),
            (4.279 + dlogT, 2.500),
            (4.415 + dlogT, 3.000),
            (4.491 + dlogT, 3.500),
            (4.544 + dlogT, 4.000),
            (4.602 + dlogT, 4.500),
            (4.699 + dlogT, 5.000 - dlogg),
            (4.699 + dlogT, 5.000 + dlogg),
            (3.525 + dlogT, 5.000 + dlogg),
            (3.525 + dlogT, 5.500 + dlogg),
            (3.301 - dlogT, 5.500 + dlogg),
        ]

        return np.array(bbox)

    def get_interpolation_data(self) -> npt.NDArray[np.float64]:
        """Default interpolation"""
        return np.array([self.logT, self.logg, self.logZ]).T

    @property
    def logg(self) -> npt.NDArray:
        return self.grid["logg"]

    @property
    def logT(self) -> npt.NDArray:
        return self.grid["logT"]

    @property
    def Teff(self) -> npt.NDArray:
        return 10**self.logT

    @property
    def Z(self) -> npt.NDArray:
        return self.grid["Z"]

    @property
    def NHI(self) -> npt.NDArray:
        return self.grid["NHI"]

    @property
    def NHeI(self) -> npt.NDArray:
        return self.grid["NHeI"]

    @property
    def NHeII(self) -> npt.NDArray:
        return self.grid["NHeII"]

    @property
    def logZ(self) -> npt.NDArray:
        return np.log10(self.Z)
