import numpy as np
from .stellib import AtmosphereLib
from .config import libsdir
from .simpletable import SimpleTable
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits


class BTSettl(AtmosphereLib):
    """
    BT-Settl Library

    References
    ----------

    Paper: Few refereed publications
      Older Ref = http://adsabs.harvard.edu/abs/2000ApJ...539..366A

    Conference Proceedings:
      http://adsabs.harvard.edu/abs/2016sf2a.conf..223A
      http://adsabs.harvard.edu/abs/2012RSPTA.370.2765A

    Files available at: https://phoenix.ens-lyon.fr/Grids/BT-Settl/

    Current Library: AGSS2009 Abundances (due to grid availability)
    Spectra rebinned to match Kurucz, and custom 2 Ang medium resolution
    """
    def __init__(self, medres=True, *args, **kwargs):
        self.name = 'BTSettl'
        if medres:
            self.source = libsdir + '/bt-settl.medres.grid.fits'
        else:
            self.source = libsdir + '/bt-settl.lowres.grid.fits'
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
        """ Boundary of BT-Settl library

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
        bbox = [(3.41497 - dlogT, 6.0 + dlogg),
                (3.41497 - dlogT, -0.5 - dlogg),
                (3.84510 + dlogT, -0.5 - dlogg),
                (4.07918 + dlogT, 0.0 - dlogg),
                (4.17609 + dlogT, 0.5 - dlogg),
                (4.30103 + dlogT, 1.0 - dlogg),
                (4.39794 + dlogT, 1.5 - dlogg),
                (4.47712 + dlogT, 2.0 - dlogg),
                (4.60206 + dlogT, 2.5 - dlogg),
                (4.60206 + dlogT, 3.0 - dlogg),
                (4.69897 + dlogT, 3.5 - dlogg),
                (4.84510 + dlogT, 4.0 - dlogg),
                (4.84510 + dlogT, 4.5 + dlogg),
                (4.00000 + dlogT, 4.5 + dlogg),
                (4.00000 + dlogT, 5.0 + dlogg),
                (3.69897 + dlogT, 5.0 + dlogg),
                (3.69897 + dlogT, 5.5 + dlogg),
                (3.60206 + dlogT, 5.5 + dlogg),
                (3.60206 + dlogT, 6.0 + dlogg),
                (3.41497 - dlogT, 6.0 + dlogg)]
            
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
