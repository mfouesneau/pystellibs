from typing import Sequence

import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .config import libsdir
from .grid import Grid
from .stellib import AtmosphereLib


class Munari(AtmosphereLib):
    """
    ATLAS9 stellar atmospheres providing higher res than Kurucz
    medium resolution (1 Ang/pix) in optical (2500-10500 Ang)

    References
    ----------

    Paper: Munari et al. 2005 A&A 442 1127
    http://adsabs.harvard.edu/abs/2005A%26A...442.1127M

    Files available at: http://archives.pd.astro.it/2500-10500/
    """

    def __init__(self, *args, **kwargs):
        self.source = libsdir + "/atlas9-munari.hires.grid.fits"
        self._load_()
        AtmosphereLib.__init__(self, *args, **kwargs)
        self.name = "Munari"

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
        self.grid = Grid(np.array(f[1].data), dict(f[1].header.items()))
        self.grid.header["NAME"] = "TGZ"

    def _getSpectra_(self, f: Sequence[pyfits.TableHDU]):
        self.spectra = f[0].data[:-1]

    def bbox(self, dlogT: float = 0.05, dlogg: float = 0.25) -> npt.NDArray[np.float64]:
        """Boundary of Munari library

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
            (3.54407 - dlogT, 5.0 + dlogg),
            (3.54407 - dlogT, 0.0 - dlogg),
            (3.77815 + dlogT, 0.0 - dlogg),
            (3.87506 + dlogT, 0.5 - dlogg),
            (3.91645 + dlogT, 1.0 - dlogg),
            (3.95424 + dlogT, 1.5 - dlogg),
            (3.98900 + dlogT, 2.0 - dlogg),
            (3.98900 + dlogT, 5.0 + dlogg),
            (3.54407 - dlogT, 5.0 + dlogg),
        ]

        return np.array(bbox)

    def get_interpolation_data(self) -> npt.NDArray[np.float64]:
        """interpolation needs alpha"""
        return np.array([self.logT, self.logg, self.logZ]).T

    @property
    def logT(self) -> npt.NDArray:
        return self.grid["LOGT"]

    @property
    def logg(self) -> npt.NDArray:
        return self.grid["LOGG"]

    @property
    def Teff(self) -> npt.NDArray:
        return self.grid["TEFF"]

    @property
    def Z(self) -> npt.NDArray:
        return self.grid["Z"]

    @property
    def logZ(self) -> npt.NDArray:
        return self.grid["LOGZ"]
