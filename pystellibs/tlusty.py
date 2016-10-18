import numpy as np
from .stellib import AtmosphereLib
from .config import libsdir
from .simpletable import SimpleTable
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits


class Tlusty(AtmosphereLib):
    """
    Tlusty O and B stellar atmospheres

    * NLTE
    * Parallel Planes
    * line blanketing

    References
    ----------
    Hubeny 1988 for initial reference
    Lanz, T., & Hubeny, I. (2003) for more recent (NL TE) developments

    * **OSTAR2002 Grid**: O-type stars, 27500 K <= Teff <= 55000 K
        * Reference: Lanz & Hubeny (2003)

    * **BSTAR2006 Grid**: Early B-type stars, 15000 K <= Teff <= 30000 K
            * Reference: Lanz & Hubeny (2007)

    files are available at: http://nova.astro.umd.edu/Tlusty2002/database/

    O and B stars rebinned to nearly 20,000 frequency points (for CLOUDY usage)
    http://nova.astro.umd.edu/Tlusty2002/database/obstar_merged_3d.ascii.gz
    """
    def __init__(self, *args, **kwargs):
        self.name = 'Tlusty'
        self.source = libsdir + '/tlusty.lowres.grid.fits'
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
        """ Boundary of Tlusty library

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
        bbox = [(4.176 - dlogT, 4.749 + dlogg),
                (4.176 - dlogT, 1.750 - dlogg),
                (4.176 + dlogT, 1.750 - dlogg),
                (4.255 + dlogT, 2.000 - dlogg),
                (4.447 + dlogT, 2.750 - dlogg),
                (4.478 + dlogT, 3.000 - dlogg),
                (4.544 + dlogT, 3.250 - dlogg),
                (4.740 + dlogT, 4.000 - dlogg),
                (4.740 + dlogT, 4.749 + dlogg),
                (4.176 - dlogT, 4.749 + dlogg) ]

        return np.array(bbox)

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
        return np.log10(self.Z)
