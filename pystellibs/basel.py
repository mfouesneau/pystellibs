""" BaSeL 2.2 library """
import numpy as np
from .simpletable import SimpleTable
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

from .stellib import Stellib
from .config import libsdir


class BaSeL(Stellib):
    """ BaSeL 2.2 library derived class
        This library + Rauch is used in Pegase.2

    The BaSeL stellar spectral energy distribution (SED) libraries are libraries
    of theoretical stellar SEDs recalibrated using empirical photometric data.
    Therefore, we call them semi-empirical libraries.

    The BaSeL 2.2 library was calibrated using photometric data from solar
    metallicity stars.

    References
    ----------
    * Lejeune, Cuisiner, and Buser, 1998 A&AS, 130, 65
    * can be downloaded http://www.astro.unibas.ch/BaSeL_files/BaSeL2_2.tar.gz
    """
    def __init__(self, *args, **kwargs):
        self.name = 'BaSeL 2.2'
        self.source = libsdir + '/stellib_BaSeL_v2.2.grid.fits'
        self._load_()
        Stellib.__init__(self, *args, **kwargs)

    def _load_(self):
        with pyfits.open(self.source) as f:
            # load data
            self._getWaveLength_(f)
            self._getTGZ_(f)
            self._getSpectra_(f)
            self._getWaveLength_units(f)

    def _getWaveLength_(self, f):
        self._wavelength = f[0].data[-1]

    def _getWaveLength_units(self, f):
        self.wavelength_unit = 'angstrom'

    def _getTGZ_(self, f):
        self.grid = SimpleTable(f[1].data)
        self.grid.header.update(f[1].header.items())
        self.grid.header['NAME'] = 'TGZ'

    def bbox(self, dlogT=0.05, dlogg=0.25):
        """ Boundary of Basel 2.2 library

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
        bbox = [(3.301 - dlogT, 5.500 + dlogg),
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
                (3.301 - dlogT, 5.500 + dlogg) ]

        return np.array(bbox)

    def _getSpectra_(self, f):
        self.spectra = f[0].data[:-1]

    @property
    def logg(self):
        return self.grid['logg']

    @property
    def logT(self):
        return self.grid['logT']

    @property
    def Teff(self):
        return self.grid['Teff']

    @property
    def Z(self):
        return self.grid['Z']

    @property
    def logZ(self):
        return self.grid['logZ']

    @property
    def NHI(self):
        return self.grid['NHI']

    @property
    def NHeI(self):
        return self.grid['NHeI']

    @property
    def NHeII(self):
        return self.grid['NHeII']
