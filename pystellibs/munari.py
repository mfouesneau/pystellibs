import numpy as np
from .stellib import AtmosphereLib
from .config import libsdir
from .simpletable import SimpleTable
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits


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
        self.name = 'Munari'
        self.source = libsdir + '/libs/atlas9-munari.hires.grid.fits'
        self._load_()
        AtmosphereLib.__init__(self, *args, **kwargs)

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
        """ Boundary of Munari library

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
        bbox = [(3.54407 - dlogT, 5.0 + dlogg),
                (3.54407 - dlogT, 0.0 - dlogg),
                (3.77815 + dlogT, 0.0 - dlogg),
                (3.87506 + dlogT, 0.5 - dlogg),
                (3.91645 + dlogT, 1.0 - dlogg),
                (3.95424 + dlogT, 1.5 - dlogg),
                (3.98900 + dlogT, 2.0 - dlogg),
                (3.98900 + dlogT, 5.0 + dlogg),
                (3.54407 - dlogT, 5.0 + dlogg)]
            
        return np.array(bbox)

    def get_interpolation_data(self):
        """ interpolation needs alpha """
        return np.array([self.logT, self.logg, self.logZ]).T

    @property
    def logT(self):
        return self.grid['logT']

    @property
    def logg(self):
        return self.grid['logg']

    @property
    def Teff(self):
        return self.grid['Teff']

    @property
    def Z(self):
        return self.grid['Z']

    @property
    def logZ(self):
        return self.grid['logZ']
