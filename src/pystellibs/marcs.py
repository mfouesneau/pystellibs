from typing import Sequence

import numpy as np
import numpy.typing as npt
from astropy.io import fits as pyfits

from .config import libsdir
from .stellib import AtmosphereLib
from .grid import Grid


class Marcs(AtmosphereLib):
    """
    MARCS stellar atmosphere models

    Gustafsson et al 2008.

    http://marcs.astro.uu.se/
    """

    def __init__(self, *args, **kwargs):
        self.source = libsdir + "/marcs.grid.fits"
        self._load_()
        AtmosphereLib.__init__(self, *args, **kwargs)
        self.name = "MARCS 2008"

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
            (3.39794 - dlogT, 5.500 + dlogg),
            (3.39794 - dlogT, 3.000 - dlogg),
            (3.47700 - dlogT, 3.000 - dlogg),
            (3.47700 - dlogT, 0.000 - dlogg),
            (3.51853 - dlogT, 0.000 - dlogg),
            (3.51853 - dlogT, -0.500 - dlogg),
            (3.62903 - dlogT, -0.5000 - dlogg),
            (3.62903 - dlogT, 0.0000 - dlogg),
            (3.720, 0.000 - dlogg),
            (3.778 + dlogT, 0.500),
            (3.829 + dlogT, 1.000),
            (3.860 + dlogT, 1.500),
            (3.906, 2.000 - dlogg),
            (3.906 + dlogT, 2.000),
            (3.906 + dlogT, 2.500),
            (3.906 + dlogT, 3.000),
            (3.906 + dlogT, 3.500),
            (3.906 + dlogT, 4.000),
            (3.906 + dlogT, 4.500),
            (3.906 + dlogT, 5.000 + dlogg),
            (3.591 + dlogT, 5.000 + dlogg),
            (3.591 + dlogT, 5.500 + dlogg),
        ]

        return np.array(bbox)

    def get_interpolation_data(self) -> npt.NDArray[np.float64]:
        """interpolation needs alpha"""
        return np.array([self.logT, self.logg, self.logZ, self.alpha]).T

    @property
    def logT(self) -> npt.NDArray:
        return np.log10(self.grid["teff"])

    @property
    def logg(self) -> npt.NDArray:
        return self.grid["logg"]

    @property
    def Teff(self) -> npt.NDArray:
        return self.grid["teff"]

    @property
    def Z(self) -> npt.NDArray:
        return 10**self.logZ

    @property
    def logZ(self) -> npt.NDArray:
        return self.grid["logz"]

    @property
    def alpha(self) -> npt.NDArray:
        return self.grid["alpha"]

    def generate_stellar_spectrum(  # pyright: ignore  /  added alpha keyword
        self, logT, logg, logL, Z, alpha=0.0, raise_extrapolation=True, **kwargs
    ):
        """Generates individual spectrum for the given stars APs and the
        stellar library

        Returns NaN spectra if the boundary conditions are not met (no extrapolation)

        Parameters
        ----------
        logT: float
            temperature

        logg: float
            log-gravity

        logL: float
            log-luminosity

        Z: float
            metallicity

        alpha: float
            alpha element

        raise_extrapolation: bool
            if set throw error on extrapolation

        null: value
            value of the flux when extrapolation and raise_extrapolation is not set

        returns
        -------
        s0: ndarray, shape=(len(stars), len(l0))
            array of spectra, one per input star
            Spectrum in ergs/s/AA or ergs/s/AA/Lsun
        """
        null_value = kwargs.pop("null", np.nan)

        # weights to apply during the interpolation (note that radii must be in cm)
        weights = self.get_weights(logT, logg, logL)
        logZ = np.log10(Z)

        l0 = self.wavelength

        # check boundary conditions, keep the data but do not compute the sed
        # if outside
        if not self.points_inside(np.atleast_2d([logT, logg]))[0]:
            if raise_extrapolation:
                raise RuntimeError("Outside library interpolation range")
            else:
                return l0, np.full(len(self.wavelength), null_value)

        aps = logT, logg, logZ, alpha
        spec = self.interpolator.interp(aps) * weights

        return spec

    def generate_individual_spectra(self, stars, nthreads=0, **kwargs):
        """Generates individual spectra for the given stars and stellar library

        Returns NaN spectra if the boundary conditions are not met (no extrapolation)

        Parameters
        ----------
        stars: Table
            contains at least (logT, logg, logL, Z) of the considered stars

        returns
        -------
        l0: ndarray, ndim=1
            wavelength definition of the spectra
            wavelength in AA

        s0: ndarray, shape=(len(stars), len(l0))
            array of spectra, one per input star
            Spectrum in ergs/s/AA or ergs/s/AA/Lsun
        """
        null_value = kwargs.pop("null", np.nan)
        ndata = len(stars)
        logT, logg, logL, Z = stars["logT"], stars["logg"], stars["logL"], stars["Z"]
        try:
            alpha = stars["alpha"]
        except Exception:
            alpha = np.zeros_like(logT)

        # weights to apply during the interpolation (note that radii must be in cm)
        weights = self.get_weights(logT, logg, logL)

        # check boundary conditions, keep the data but do not compute the sed
        # if outside
        bound = self.points_inside(np.array([logT, logg]).T)
        specs = np.empty((ndata, len(self._wavelength)), dtype=float)
        specs[~bound] = np.full(len(self.wavelength), null_value)

        logZ = np.log10(Z)
        aps = np.array([logT, logg, logZ, alpha]).T
        s = self.interpolator.interp(aps[bound]) * weights[bound, None]
        specs[bound] = s

        l0 = self.wavelength
        specs = specs * self.flux_units

        return l0, specs
