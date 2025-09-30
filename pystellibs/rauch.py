"""Rauch White Dwarfs stellar atmospheres"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .config import libsdir
from .simpletable import SimpleTable
from .stellib import Stellib


class Rauch(Stellib):
    """
    Rauch White Dwarfs stellar atmospheres

    References
    ----------

    Rauch, T.; Werner, K.; Bohlin, R.; Kruk, J. W., "The virtual observatory service
    TheoSSA: Establishing a database of synthetic stellar flux standards. I. NLTE
    spectral analysis of the DA-type white dwarf G191-B2B"
    """

    def __init__(self, *args, **kwargs):
        self.name = "Rauch"
        self.source = libsdir + "/stellib_Rauch.grid.fits"
        self._load_()
        Stellib.__init__(self, *args, **kwargs)

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
        """Boundary of Rauch library

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
            (4.700 - dlogT, 8.000 + dlogg),
            (4.700 - dlogT, 5.000 - dlogg),
            (5.000 + dlogT, 5.000 - dlogg),
            (5.280 + dlogT, 6.000 - dlogg),
            (5.280 + dlogT, 8.000 + dlogg),
            (4.700 - dlogT, 8.000 + dlogg),
        ]

        return np.array(bbox)

    @property
    def logT(self) -> npt.NDArray:
        return self.grid["logT"]

    @property
    def logg(self) -> npt.NDArray:
        return self.grid["logg"]

    @property
    def Teff(self) -> npt.NDArray:
        return np.power(10, self.grid["logT"])

    @property
    def Z(self) -> npt.NDArray:
        return self.grid["Z"]

    @property
    def logZ(self) -> npt.NDArray:
        return np.log10(self.Z)

    @property
    def NHI(self) -> npt.NDArray:
        return self.grid["NHI"]

    @property
    def NHeI(self) -> npt.NDArray:
        return self.grid["NHeI"]

    @property
    def NHeII(self) -> npt.NDArray:
        return self.grid["NHeII"]
