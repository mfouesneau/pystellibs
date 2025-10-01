from typing import Sequence

import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .config import libsdir
from .simpletable import SimpleTable
from .stellib import AtmosphereLib


class Kurucz(AtmosphereLib):
    """
    The stellar atmosphere models by Castelli and Kurucz 2004 or ATLAS9

    * LTE
    * PP
    * line blanketing
    """

    def __init__(self, *args, **kwargs):
        self.source = libsdir + "/kurucz2004.grid.fits"
        self._load_()
        AtmosphereLib.__init__(self, *args, **kwargs)
        self.name = "Kurucz 2004"

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
        self._wavelength = np.array(f[0].data[-1])

    def _getTGZ_(self, f: Sequence[pyfits.TableHDU]):
        self.grid = SimpleTable(f[1].data)
        self.grid.header.update(f[1].header.items())
        self.grid.header["NAME"] = "TGZ"

    def _getSpectra_(self, f: Sequence[pyfits.TableHDU]):
        self.spectra = f[0].data[:-1]

    def bbox(self, dlogT: float = 0.05, dlogg: float = 0.25) -> npt.NDArray[np.float64]:
        """Boundary of Kurucz 2004 library

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
            (3.54406 - dlogT, 5.000 + dlogg),
            (3.55403 - dlogT, 0.000 - dlogg),
            (3.778, 0.000 - dlogg),
            (3.778 + dlogT, 0.000),
            (3.875 + dlogT, 0.500),
            (3.929 + dlogT, 1.000),
            (3.954 + dlogT, 1.500),
            (4.146, 2.000 - dlogg),
            (4.146 + dlogT, 2.000),
            (4.279 + dlogT, 2.500),
            (4.415 + dlogT, 3.000),
            (4.491 + dlogT, 3.500),
            (4.591 + dlogT, 4.000),
            (4.689 + dlogT, 4.500),
            (4.699 + dlogT, 5.000 + dlogg),
            (3.544 - dlogT, 5.000 + dlogg),
        ]

        return np.array(bbox)

    def get_interpolation_data(self) -> npt.NDArray[np.float64]:
        """Default interpolation"""
        return np.array([self.logT, self.logg, self.logZ]).T

    @property
    def logT(self) -> npt.NDArray:
        return self.grid["logT"]

    @property
    def logg(self) -> npt.NDArray:
        return self.grid["logg"]

    @property
    def Teff(self) -> npt.NDArray:
        return self.grid["Teff"]

    @property
    def Z(self) -> npt.NDArray:
        return self.grid["Z"]

    @property
    def logZ(self) -> npt.NDArray:
        return self.grid["logz"]
