"""
Stellar library class

Intent to implement a generic module to manage stellar library from various
sources.

The interpolation is implemented from the pegase.2 fortran converted algorithm.
(this may not be super pythonic though)

.. note::

    a cython version is available for speed up and should be used transparently when available
    (run make once)
"""
import numpy as np

from .ezunits import unit, hasUnit
from .helpers import nbytes, isNestedInstance
from .future import Path
from .interpolator import NDLinearInterpolator
from .ezmap import map as ezmap
from .ezmap import Partial


lsun = 3.839e+26   # in W (Watts)
sig_stefan = 5.67037321 * 1e-8  # W * m**-2 * K**-4
rsun = 6.955e8  # in meters


def _drop_units(q):
    """ Drop the unit definition silently """
    try:
        return q.magnitude
    except:
        return q


class Stellib(object):
    """ Basic stellar library class

    Attributes
    ----------
    interpolator: interpolator.BaseInterpolator
        interpolator to use, default LeujeuneInterpolator
    """
    def __init__(self, *args, **kwargs):
        """ Contructor """
        self.interpolator = kwargs.pop('interpolator', None)
        if self.interpolator is None:
            self.interpolator = NDLinearInterpolator(self)
        if not hasattr(self, 'wavelength_unit'):
            self.wavelength_unit = None

    def get_interpolation_data(self):
        """ Default interpolation """
        return np.array([self.logT, self.logg, self.logZ]).T

    @property
    def wavelength(self):
        l = np.copy(self._wavelength)
        if self.wavelength_unit is not None:
            return l * unit[self.wavelength_unit]
        else:
            return l

    @property
    def flux_units(self):
        if self.wavelength_unit is not None:
            return unit['erg/s/' + self.wavelength_unit]
        else:
            return 1.

    def _load_(self):
        """ Load the library """
        raise NotImplementedError

    @property
    def nbytes(self):
        """ return the number of bytes of the object """
        return nbytes(self)

    def plot_boundary(self, ax=None, dlogT=0., dlogg=0., **kwargs):
        """
        Parameters
        ----------

        dlogT: float
            margin in logT (see get_boundaries)

        dlogg: float
            margin in logg (see get_boundaries)

        .. see also::
            :func:`matplotlib.plot`
                For additional kwargs
        """
        import matplotlib.patches as patches
        from pylab import gca
        if ax is None:
            ax = gca()
        p = self.get_boundaries(dlogT=dlogT, dlogg=dlogg)
        ax.add_patch(patches.PathPatch(p, **kwargs))
        return p

    def __add__(self, other):
        if not isNestedInstance(other, Stellib):
            raise ValueError('expecting a Stellib object, got {0}'.format(type(other)))

        return CompositeStellib([self, other])

    def __repr__(self):
        return "{0:s}, ({1:s})\n{2:s}".format(self.name, nbytes(self, pprint=True),
                                              object.__repr__(self))

    def get_weights(self, logT, logg, logL, weights=None):
        """ Returns the proper weights for the interpolation

        in spectra libraries the default is to have Lbol=1 normalization

        Parameters
        ----------
        logT: float or ndarray
            log-temperatures
        logg: float or ndarray
            log-gravity
        logL: float or ndarray
            bolometric luminosity
        """
        # weights to apply during the interpolation (note that radii must be in cm)
        # Stellar library models are given in cm^-2  ( 4 pi R)
        # Compute radii of each point using log(T) and log(L)
        Lsun = unit['lsun'].to("ergs/s").magnitude
        L = 10 ** logL * Lsun
        if weights is not None:
            weights *= L
        else:
            weights = L
        return weights

    def generate_stellar_spectrum(self, logT, logg, logL, Z,
                                  raise_extrapolation=True, **kwargs):
        """ Generates individual spectrum for the given stars APs and the
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
        null_value = kwargs.pop('null', np.nan)

        # weights to apply during the interpolation (note that radii must be in cm)
        weights = self.get_weights(logT, logg, logL)
        logZ = np.log10(Z)

        l0 = self.wavelength

        # check boundary conditions, keep the data but do not compute the sed
        # if outside
        if not self.points_inside(np.atleast_2d([logT, logg]))[0]:
            if raise_extrapolation:
                raise RuntimeError('Outside library interpolation range')
            else:
                return l0, np.full(len(self.wavelength), null_value)

        aps = logT, logg, logZ
        spec = self.interpolator.interp(aps) * weights

        return spec

    def generate_individual_spectra(self, stars, nthreads=0, **kwargs):
        """ Generates individual spectra for the given stars and stellar library

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
        null_value = kwargs.pop('null', np.nan)
        ndata = len(stars)
        logT, logg, logL, Z = stars['logT'], stars['logg'], stars['logL'], stars['Z']

        # weights to apply during the interpolation (note that radii must be in cm)
        weights = self.get_weights(logT, logg, logL)

        # check boundary conditions, keep the data but do not compute the sed
        # if outside
        bound = self.points_inside(np.array([logT, logg]).T)
        specs = np.empty((ndata, len(self._wavelength)), dtype=float)
        specs[~bound] = np.full(len(self.wavelength), null_value)

        logZ = np.log10(Z)
        aps = np.array([logT, logg, logZ]).T
        s = self.interpolator.interp(aps) * weights[:, None]
        specs[bound] = s[bound]

        l0 = self.wavelength
        specs = specs * self.flux_units

        return l0, specs

    def points_inside(self, xypoints, dlogT=0.1, dlogg=0.3):
        """
        Returns if a point is inside the polygon defined by the boundary of the library

        Parameters
        ----------
        xypoints: sequence
            a sequence of N logg, logT pairs.

        dlogT: float
            margin in logT

        dlogg: float
            margin in logg

        returns
        -------
        r: ndarray(dtype=bool)
            a boolean ndarray, True for points inside the polygon.
            A point on the boundary may be treated as inside or outside.
        """
        p = self.get_boundaries(dlogT=dlogT, dlogg=dlogg)
        return p.contains_points(xypoints)

    def get_radius(self, logl, logt):
        """ Returns the radius of a star given its luminosity and temperature

        Assuming a black body, it comes:

        .. math::

                R ^ 2 = L / ( 4 \pi \sigma T ^ 4 ),

        with:

            * L, luminosity in W,
            * pi, 3.141592...
            * sig, Stefan constant in  W * m**-2 * K**-4
            * T, temperature in K

        Parameters
        ----------
        logl: ndarray[float, ndim=1]
            log luminosities from the isochrones, in Lsun

        logt: ndarray[float, ndim=1]
            log temperatures from the isochrones, in K

        returns
        -------
        radii: ndarray[float, ndim=1]
            array of radii in m (SI units)
        """
        return np.sqrt( (10 ** logl) * lsun / (4.0 * np.pi * sig_stefan * ((10 ** logt) ** 4)) )

    def get_boundaries(self, dlogT=0.1, dlogg=0.3, **kwargs):
        """ Returns the closed boundary polygon around the stellar library with
        given margins

        Parameters
        ----------
        s: Stellib
            Stellar library object

        dlogT: float
            margin in logT

        dlogg: float
            margin in logg

        returns
        -------
        b: ndarray[float, ndim=2]
            closed boundary edge points: [logT, logg]

        .. note::

            as computing the boundary could take time, it is saved in the object
            and only recomputed when parameters are updated
        """
        # if bbox is defined then assumes it is more precise and use it instead.
        if dlogT is None:
            dlogT = 0.1
        if dlogg is None:
            dlogg = 0.3
        if hasattr(self, 'bbox'):
            return Path(self.bbox(dlogT, dlogg))

        if getattr(self, '_bound', None) is not None:
            # check if recomputing is needed
            if ((self._bound[1] - dlogT) < 1e-3) and (abs(self._bound[2] - dlogg) < 1e-3):
                return self._bound[0]

        leftb   = [(np.max(self.logT[self.logg == k]) + dlogT, k ) for k in np.unique(self.logg)]
        leftb  += [(leftb[-1][1], leftb[-1][0] + dlogg)]
        leftb   = [(leftb[0][1], leftb[0][0] - dlogg)] + leftb

        rightb  = [(np.min(self.logT[self.logg == k]) - dlogT, k) for k in np.unique(self.logg)[::-1]]
        rightb += [(rightb[-1][1], rightb[-1][0] - dlogg)]
        rightb  = [(rightb[0][1], rightb[0][0] + dlogg)] + rightb

        b = leftb + rightb
        b += [b[0]]

        self._bound = (Path(np.array(b)), dlogT, dlogg)
        return self._bound[0]


class AtmosphereLib(Stellib):
    """
    Almost identical to a spectral library. The difference lies into the units
    of the input libraries.
    """
    def get_weights(self, logT, logg, logL, weights=None):
        """ Returns the proper weights for the interpolation """
        # weights to apply during the interpolation (note that radii must be in cm)
        # Stellar library models are given in cm^-2  ( 4 pi R)
        # Compute radii of each point using log(T) and log(L)
        radii = self.get_radius(logL, logT)
        if weights is not None:
            weights *= 4. * np.pi * (radii * 1e2) ** 2
        else:
            weights = 4. * np.pi * (radii * 1e2) ** 2
        return weights


class CompositeStellib(Stellib):
    """ Generates an object from the union of multiple individual libraries """
    def __init__(self, osllist, *args, **kwargs):
        self._olist = osllist
        self._dlogT = 0.1
        self._dlogg = 1.0

    @property
    def name(self):
        return ' + '.join([sl.name for sl in self._olist])

    def set_default_extrapolation_bounds(self, dlogT=None, dlogg=None):
        if dlogT is not None:
            self._dlogT = dlogT
        if dlogg is not None:
            self._dlogg = dlogg

    def __add__(self, other):
        """ Adding a library after """
        if not isNestedInstance(other, Stellib):
            raise ValueError('expecting a Stellib object, got {0}'.format(type(other)))

        lst = [k for k in self._olist] + [other]
        return CompositeStellib(lst)

    def __radd__(self, other):
        """ Adding a library before """
        if not isNestedInstance(other, Stellib):
            raise ValueError('expecting a Stellib object, got {0}'.format(type(other)))

        lst = [other] + [k for k in self._olist]
        return CompositeStellib(lst)

    @property
    def wavelength(self):
        """ return a common wavelength sampling to all libraries. This can be
        used to reinterpolate any spectrum onto a common definition """
        # check units
        has_units = [hasUnit(osl) for osl in self._olist]
        test = sum(has_units)
        if (test == 0):
            return np.unique(np.asarray([ osl.wavelength for osl in self._olist ]))

        # which library sets the units
        common_unit = self._olist[0].wavelength_unit
        libset_unit = self._olist[0].name
        # if some libraries do not have units... Should not happen often!
        if (test < len(self._olist)):
            for k, osl in enumerate(self._olist):
                common_unit = osl.wavelength_unit
                libset_unit = osl.name
                if common_unit is not None:
                    break
            print("Warning: Some libraries do not have units. Assuming consistency with {0:s}".format(libset_unit))

        wave = []
        for osl in self._olist:
            wave.append(osl.wavelength.to(common_unit).magnitude)
        return np.unique(np.array(wave)) * unit[common_unit]

    @property
    def source(self):
        return ' + '.join([k.name for k in self._olist])

    @property
    def logT(self):
        return np.hstack([osl.logT for osl in self._olist])

    @property
    def logg(self):
        return np.hstack([osl.logg for osl in self._olist])

    @property
    def Teff(self):
        return np.hstack([osl.Teff for osl in self._olist])

    @property
    def Z(self):
        return np.hstack([osl.Z for osl in self._olist])

    @property
    def logZ(self):
        return np.hstack([osl.logZ for osl in self._olist])

    def which_osl(self, xypoints, **kwargs):
        """
        Returns the library indice that contains each point in xypoints

        The decision is made from a two step search:

            * first, each point is checked against the strict boundary of each
              library (i.e., dlogT = 0, dlogg = 0).
            * second, if points are not found in strict mode, the boundary is
              relaxed and a new search is made.

        Each point is associated to the first library matching the above conditions.

        Parameters
        ----------
        xypoints: sequence
            a sequence of N logg, logT pairs.

        dlogT: float
            margin in logT

        dlogg: float
            margin in logg

        returns
        -------
        res: ndarray(dtype=int)
            a ndarray, 0 meaning no library covers the point, and 1, ... n, for the n-th library
        """
        dlogT = kwargs.pop('dlogT', self._dlogT)
        dlogg = kwargs.pop('dlogg', self._dlogg)

        xy = np.atleast_2d(xypoints)

        # check that all points are in the full boundary area
        # MF: testing why points_inside does not agree on all computers...
        # as we do not keep individual results, no need to store then all
        # first, collapse directly

        # res_temp = np.zeros((len(xy),len(self._olist)))
        # for ek,ok in enumerate(self._olist):
        #    res_temp[:, ek] = ok.points_inside(xy, dlogT=dlogT, dlogg=dlogg).astype(int)
        res_temp = np.zeros(len(xy), dtype=int)
        for ek, ok in enumerate(self._olist):
            res_temp += ok.points_inside(xy, dlogT=dlogT, dlogg=dlogg).astype(int)

        ind = res_temp > 0
        res = np.zeros(len(xy), dtype=int)
        res[ind] = 1
        res = res - 1

        # res = self.points_inside(xy, dlogT=dlogT, dlogg=dlogg).astype(int) - 1
        # if res == -1: invalid point, res == 0: proceed

        if max(res) < 0:
            # DEBUG: should generate an exeception in further functions
            # TODO: get rid and replace
            return res
            # return res

        # Strict mode
        # ===========
        # Not extrapolation allowed >> dlogT = 0, dlogg = 0
        # 0 is used to flag points without a matching library yet
        # libraries are then indexed from 1 to n
        # -1 means point outside the compound library
        for ek, ok in enumerate(self._olist):
            if 0 in res:
                ind = np.atleast_1d(np.squeeze(np.where(res == 0)))
                r = ok.points_inside(xy[ind], dlogT=0., dlogg=0.)
                res[ind[r]] = ek + 1

        # Relaxed mode
        # ============
        # In this case we accept some flexibility in the boundary limits,
        # which allows limited extrapolation ranges.
        # this only affects points not already matched
        if 0 in res:
            for ek, ok in enumerate(self._olist):
                if 0 in res:
                    ind = np.atleast_1d(np.squeeze(np.where(res == 0)))
                    r = ok.points_inside(xy[ind], dlogT=dlogT, dlogg=dlogg)
                    res[ind[r]] = ek + 1
        return res

    def __repr__(self):
        return "CompositeStellib, {0}\n{1}".format(object.__repr__(self), '\n'.join([k.name for k in self._olist]))

    def get_boundaries(self, **kwargs):
        """ Returns the closed boundary polygon around the stellar library with
        given margins

        Parameters
        ----------
        s: Stellib
            Stellar library object

        dlogT: float
            margin in logT

        dlogg: float
            margin in logg

        returns
        -------
        b: ndarray[float, ndim=2]
            (closed) boundary points: [logg, Teff] (or [Teff, logg] is swap is True)

        .. note::

            as computing the boundary could take time, it is saved in the object
            and only recomputed when parameters are updated
        """
        dlogT = kwargs.pop('dlogT', self._dlogT)
        dlogg = kwargs.pop('dlogg', self._dlogg)

        if getattr(self, '_bound', None) is not None:
            if ((self._bound[1] - dlogT) < 1e-3) and (abs(self._bound[2] - dlogg) < 1e-3):
                return self._bound[0]

        b = [osl.get_boundaries(dlogT=dlogT, dlogg=dlogg, **kwargs) for osl in self._olist]
        self._bound = (Path.make_compound_path(*b), dlogT, dlogg)
        return self._bound[0]

    def generate_stellar_spectrum(self, logT, logg, logL, Z,
                                  raise_extrapolation=True, **kwargs):
        """ Generates individual spectrum for the given stars APs and the
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
        try:
            bounds = kwargs.pop('bounds', None)
            dlogT = bounds.get('dlogT', self._dlogT)
            dlogg = bounds.get('dlogg', self._dlogg)
        except:
            dlogT = None
            dlogg = None

        osl_index = self.which_osl(np.atleast_2d([logT, logg]), dlogT=dlogT, dlogg=dlogg)[0]
        osl = self._olist[osl_index - 1]
        specs = osl.generate_stellar_spectrum(logT, logg, logL, Z,
                                              raise_extrapolation,
                                              **kwargs)
        specs = self.reinterpolate_spectra(osl.wavelength, specs, left=0., right=0.)
        return specs

    def reinterpolate_spectra(self, l0, specs, **kwargs):
        """ One-dimensional linear interpolation onto the common wavelength.

        Returns the one-dimensional interpolated spectrum

        Parameters
        ----------
        l0 : 1-D sequence of floats (with units or not)
            wavelength of the spectrum to interpolate

        specs : 1-D sequence of floats
            spectrum to reinterpolate

        left : float, optional
            Value to return for `x < xp[0]`, default is `fp[0]`.

        right : float, optional
            Value to return for `x > xp[-1]`, default is `fp[-1]`.

        period : None or float, optional
            A period for the x-coordinates. This parameter allows the proper
            interpolation of angular x-coordinates. Parameters `left` and `right`
            are ignored if `period` is specified.

        Returns
        -------
        spec : ndarray
            The interpolated values
        """
        # TODO: proper reinterpolation that conserves energy... but makes a new
        # resolution
        wave = self.wavelength
        try:
            wave = wave.to(l0.unit)
        except:
            wave = _drop_units(wave)
        f = np.interp(_drop_units(wave), _drop_units(l0), _drop_units(specs), **kwargs)
        return f

    def generate_individual_spectra(self, stars, nthreads=0, **kwargs):
        """ Generates individual spectra for the given stars and stellar library

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
        null_value = kwargs.pop('null', np.nan)
        try:
            bounds = kwargs.pop('bounds', None)
            dlogT = bounds.get('dlogT', self._dlogT)
            dlogg = bounds.get('dlogg', self._dlogg)
        except:
            dlogT = None
            dlogg = None

        ndata = len(stars)
        logT, logg, logL, Z = stars['logT'], stars['logg'], stars['logL'], stars['Z']
        osl_index = self.which_osl(list(zip(logT, logg)), dlogT=dlogT, dlogg=dlogg)

        # Do the actual interpolation, avoiding exptrapolations
        specs = np.empty( (ndata, len(self.wavelength)), dtype=float )
        func = Partial(self.generate_stellar_spectrum,
                       raise_extrapolation=False, null=null_value)
        specs = ezmap.map(func, zip(logT, logg, logL, Z), ncpu=nthreads, **kwargs)

        l0 = self.wavelength
        specs = specs * self.flux_units

        return l0, specs, osl_index
