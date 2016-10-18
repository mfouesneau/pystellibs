""" Rauch White Dwarfs stellar atmospheres """
import numpy as np
from .simpletable import SimpleTable
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

from .stellib import Stellib
from .config import libsdir


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
        self.name = 'Rauch'
        self.source = libsdir + '/stellib_Rauch.grid.fits'
        self._load_()
        Stellib.__init__(self, *args, **kwargs)

    def _load_(self):
        with pyfits.open(self.source) as f:
            # load data
            self._getWaveLength_(f)
            self._getTGZ_(f)
            self._getSpectra_(f)
            self._getWaveLength_units(f)

    def _getWaveLength_units(self, f):
        self.wavelength_unit = 'angstrom'

    def _getWaveLength_(self, f):
        self._wavelength = f[0].data[-1]

    def _getTGZ_(self, f):
        self.grid = SimpleTable(f[1].data)
        self.grid.header.update(f[1].header.items())
        self.grid.header['NAME'] = 'TGZ'

    def _getSpectra_(self, f):
        self.spectra = f[0].data[:-1]

    def bbox(self, dlogT=0.05, dlogg=0.25):
        """ Boundary of Rauch library

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
        bbox = [(4.700 - dlogT, 8.000 + dlogg),
                (4.700 - dlogT, 5.000 - dlogg),
                (5.000 + dlogT, 5.000 - dlogg),
                (5.280 + dlogT, 6.000 - dlogg),
                (5.280 + dlogT, 8.000 + dlogg),
                (4.700 - dlogT, 8.000 + dlogg) ]

        return np.array(bbox)

    @property
    def logT(self):
        return self.grid['logT']

    @property
    def logg(self):
        return self.grid['logg']

    @property
    def Teff(self):
        return 10 ** self.grid['logT']

    @property
    def Z(self):
        return self.grid['Z']

    @property
    def logZ(self):
        return np.log10(self.Z)

    @property
    def NHI(self):
        return self.grid['NHI']

    @property
    def NHeI(self):
        return self.grid['NHeI']

    @property
    def NHeII(self):
        return self.grid['NHeII']
