""" Lejeune interpolator is basically a linear interpolator for a Lejeune grid
based spectral library.

This is the simplest interpolator but most commonly used.

It takes care of boundary conditions by imposing limits to extrapolation on a
given grid.
"""

import numpy as np
from ..ezmap import map
from .interpolator import BaseInterpolator


def __det3x3__(a):
    """ compute the 3x3 determinant of an array

    Hard coded equations are 8 times faster than np.linalg.det for a matrix 3x3

    Parameters
    ----------
    a: ndarray, shape=(3,3), dtype=float
        array matrix

    Returns
    -------
    val: float
        determinant of a
    """
    val  = +a[0] * (a[4] * a[8] - a[7] * a[5])
    val += -a[1] * (a[3] * a[8] - a[6] * a[5])
    val += +a[2] * (a[3] * a[7] - a[6] * a[4])
    return val


def __interp__(T0, g0, T,g, dT_max=0.1, eps=1e-6):
    """
    Interpolation of the (T,g) grid at fixed Z

    Translated from Pegase.2 fortran version
    (this may not be pythonic though)

    Note: preference is always given to the temperature over
        the gravity when needed.

    Parameters
    ----------
    T0: float
        log(Teff) to obtain

    g0: float
        log(g) to obtain

    T: float
        log(Teff) of the grid

    g: float
        log(g) of the grid

    dT_max: float, optional
        If, T2 (resp. T1) is too far from T compared to T1 (resp. T2), i2
        (resp. i1) is not used.  (see below for namings)

    eps: float
        temperature sensitivity under which points are considered to have
        the same temperature

    Returns
    -------
    idx: ndarray, dtype=int, size=4
        4 star indexes

    w: ndarray, dtype=float, size=4
        4 associated weights

    ..note::
        if index is -1, this means the point is rejected and the associated
        weight is 0.


    Naming conventions
    ------------------

    i1 = index of the star with temperature > T and gravity > g.
    Among all such stars, one chooses the one minimizing
    |Delta T|+kappa*|Delta g|.
    If no star with temperature > T and gravity > g exists, i1 = -1

    i2 = index of the star with temperature > T and gravity < g.

    i3 = index of the star with temperature < T and gravity > g.

    i4 = index of the star with temperature < T and gravity < g.

    g
    /|\
    | i3  |
    |     |  i1
    | ----x------
    |     |    i2
    |  i4 |
    |__________\ T
                /
    """
    kappa  = 0.1

    idx    = np.arange(len(g))
    deltag = g - g0
    deltaT = T - T0
    dist   = kappa * abs(deltag) + abs(deltaT)

    if dist.min() == 0:
        return (dist.argmin(),-1,-1,-1), (1.,0.,0.,0.)

    # Looking for i_{1..4}
    ind_dT = deltaT >= 0
    ind_dg = deltag >= 0

    # i1
    ind = (ind_dT & ind_dg)
    if True in ind:
        i1  = idx[ind][dist[ind].argmin()]
    else:
        i1 = -1

    # i2
    ind = (ind_dT & ~ind_dg)
    if True in ind:
        i2  = idx[ind][dist[ind].argmin()]
    else:
        i2 = -1

    # i3
    ind = (~ind_dT & ind_dg)
    if True in ind:
        i3  = idx[ind][dist[ind].argmin()]
    else:
        i3 = -1

    # i4
    ind = (~ind_dT & ~ind_dg)
    if True in ind:
        i4  = idx[ind][dist[ind].argmin()]
    else:
        i4 = -1

    # checking integrity
    if ( (i1 < 0) & (i2 < 0) & (i3 < 0) & (i4 < 0) ):
        raise ValueError("Interp. Error, could not find appropriate knots")

    T1 = T[i1]
    T2 = T[i2]
    T3 = T[i3]
    T4 = T[i4]
    g1 = g[i1]
    g2 = g[i2]
    g3 = g[i3]
    g4 = g[i4]

    # If, T2 (resp. T1) is too far from T compared to T1
    # (resp. T2), i2 (resp. i1) is not used.
    # The same for i3 and i4.
    if ( (i1 > 0) & (i2 > 0) ):
        if (T1 < T2 - dT_max):
            i2 = -1
        elif (T2 < T1 - dT_max):
            i1 = -1

    if ( (i3 > 0) & (i4 > 0) ):
        if (T3 > T4 + dT_max):
            i4 = -1
        elif (T4 > T3 + dT_max):
            i3 = -1

    if ( (i1 < 0) & (i2 < 0) & (i3 < 0) & (i4 < 0) ):
        raise ValueError("Interp. Error, could not find appropriate knots")

    # Interpolation in the (T, g) plane between the used points
    # (at least 1, at most 4).
    # Code "0110" means that i1 = i4 = 0, i2 /=0 and i3 /= 0.
    #
    # Note: preference is always given to the temperature over
    #   the gravity when needed.
    if (i1 < 0):
        if (i2 < 0):
            if (i3 < 0):
                if (i4 < 0):
                    #                   # 0000
                    raise ValueError("Error")  # should not be possible
                else:                   # 0001
                    alpha1 = 0.
                    alpha2 = 0.
                    alpha3 = 0.
                    alpha4 = 1.
                # endif
            elif (i4 < 0):              # 0010
                alpha1 = 0.
                alpha2 = 0.
                alpha3 = 1.
                alpha4 = 0.
            else:                       # 0011
                alpha1 = 0.
                alpha2 = 0.
                if ( abs(T3 - T4) < eps ):
                    if (g3 == g4):
                        alpha3 = 0.5
                    else:
                        alpha3 = (g0 - g4) / (g3 - g4)
                    # endif
                    alpha4 = 1. - alpha3
                else:
                    if (T3 > T4):
                        alpha3 = 1.
                        alpha4 = 0.
                        i4 = -1
                    else:
                        alpha3 = 0.
                        i3 = -1
                        alpha4 = 1.
                    # endif
                # endif
            # endif
        elif (i3 < 0):
            if (i4 < 0):
                #                        # 0100
                alpha1 = 0.
                alpha2 = 1.
                alpha3 = 0.
                alpha4 = 0.
            else:                        # 0101
                alpha1 = 0.
                if (T2 == T4):
                    alpha2 = 0.5
                else:
                    alpha2 = (T0 - T4) / (T2 - T4)
                # endif
                alpha3 = 0.
                alpha4 = 1. - alpha2
            # endif
        elif (i4 < 0):                   # 0110
            alpha1 = 0.
            if (T2 == T3):
                alpha2 = 0.5
            else:
                alpha2 = (T0 - T3) / (T2 - T3)
            # endif
            alpha3 = 1. - alpha2
            alpha4 = 0.
        else:                            # 0111
            # Assume that (T, g) is within the triangle i
            # formed by the three points.

            mat0 = np.asarray([
                [ T2, T3, T4 ],
                [ g2, g3, g4 ],
                [ 1., 1., 1. ]  ])
            mat2 = np.asarray([
                [ T0, T3, T4 ],
                [ g0, g3, g4 ],
                [ 1., 1.,  1.]  ])
            mat3 = np.asarray([
                [ T2, T0, T4 ],
                [ g2, g0, g4 ],
                [ 1., 1.,  1.]  ])
            mat4 = np.asarray([
                [ T2, T3, T0 ],
                [ g2, g3, g0 ],
                [ 1., 1., 1. ]  ])
            det0 = __det3x3__(mat0.ravel())
            det2 = __det3x3__(mat2.ravel())
            det3 = __det3x3__(mat3.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = 0.
            alpha2 = det2 / det0
            alpha3 = det3 / det0
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed
            # by the three used points use only two points.
            if ((alpha2 < 0.) | (alpha2 > 1. ) | (alpha3 < 0.) | (alpha3 > 1.) | (alpha4 < 0.) | (alpha4 > 1. ) ):
                alpha1 = 0.
                if (T2 == T3):
                    alpha2 = 0.5
                else:
                    alpha2 = (T0 - T3) / (T2 - T3)
                # endif
                alpha3 = 1. - alpha2
                alpha4 = 0.
                i4 = -1
            # endif
        # endif
    elif (i2 < 0):
        if (i3 < 0):
            if (i4 < 0):
                #                      # 1000
                alpha1 = 1.
                alpha2 = 0.
                alpha3 = 0.
                alpha4 = 0.
            else:                      # 1001
                if (T1 == T4):
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.
                alpha3 = 0.
                alpha4 = 1. - alpha1
            # endif
        elif (i4 < 0):                 # 1010
            if (T1 == T3):
                alpha1 = 0.5
            else:
                alpha1 = (T0 - T3) / (T1 - T3)
            # endif
            alpha2 = 0.
            alpha3 = 1. - alpha1
            alpha4 = 0.
        else:                          # 1011
            # Assume that (T, g) is within the triangle formed by the three points.
            mat0 = np.asarray([
                [ T1, T3, T4 ],
                [ g1, g3, g4 ],
                [ 1., 1.,  1.]  ])
            mat1 = np.asarray([
                [ T0, T3, T4 ],
                [ g0, g3, g4 ],
                [ 1., 1.,  1.]  ])
            mat3 = np.asarray([
                [ T1, T0, T4 ],
                [ g1, g0, g4 ],
                [ 1., 1.,  1.]  ])
            mat4 = np.asarray([
                [ T1, T3, T0 ],
                [ g1, g3, g0 ],
                [ 1., 1.,  1.]  ])
            det0 = __det3x3__(mat0.ravel())
            det1 = __det3x3__(mat1.ravel())
            det3 = __det3x3__(mat3.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = det1 / det0
            alpha2 = 0.
            alpha3 = det3 / det0
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed by the three used points,
            # use only two points.

            if ((alpha1 < 0.) | (alpha1 > 1.) | (alpha3 < 0.) | (alpha3 > 1.) | (alpha4 < 0.) | (alpha4 > 1.) ):
                if (T1 == T4):
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.
                alpha3 = 0.
                i3 = -1
                alpha4 = 1. - alpha1
            # endif
        # endif
    elif (i3 < 0):
        if (i4 < 0):
            #                       # 1100
            if (abs(T1 - T2) < eps):
                if (g1 == g2):
                    alpha1 = 0.5
                else:
                    alpha1 = (g0 - g2) / (g1 - g2)
                # endif
                alpha2 = 1. - alpha1
            else:
                if (T1 < T2):
                    alpha1 = 1.
                    alpha2 = 0.
                    i2 = -1
                else:
                    alpha1 = 0.
                    i1 = -1
                    alpha2 = 1.
                # endif
            # endif
            alpha3 = 0.
            alpha4 = 0.
        else:                       # 1101
            # Assume that (T, g) is within the triangle formed by the three points.
            mat0 = np.asarray([
                [ T1, T2, T4 ],
                [ g1, g2, g4 ],
                [ 1., 1.,  1.]  ])
            mat1 = np.asarray([
                [ T0, T2, T4 ],
                [ g0, g2, g4 ],
                [ 1., 1.,  1.]  ])
            mat2 = np.asarray([
                [ T1, T0, T4 ],
                [ g1, g0, g4 ],
                [ 1., 1.,  1.]  ])
            mat4 = np.asarray([
                [ T1, T2, T0 ],
                [ g1, g2, g0 ],
                [ 1., 1.,  1. ]  ])
            det0 = __det3x3__(mat0.ravel())
            det1 = __det3x3__(mat1.ravel())
            det2 = __det3x3__(mat2.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = det1 / det0
            alpha2 = det2 / det0
            alpha3 = 0.
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed by the three used points,
            # use only two points.
            if ((alpha1 < 0.) | (alpha1 > 1.) | (alpha2 < 0.) | (alpha2 > 1.) | (alpha4 < 0.) | (alpha4 > 1.) ):
                if (T1 == T4):
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.
                i2 = -1
                alpha3 = 0.
                alpha4 = 1. - alpha1
            # endif
        # endif
    elif (i4 < 0):
        #                           # 1110
        # Assume that (T, g) is within the triangle formed by the three points.
        mat0 = np.asarray([
            [ T1, T2, T3 ],
            [ g1, g2, g3 ],
            [ 1., 1.,  1.]  ])
        mat1 = np.asarray([
            [ T0, T2, T3 ],
            [ g0, g2, g3 ],
            [ 1., 1.,  1.]  ])
        mat2 = np.asarray([
            [ T1, T0, T3 ],
            [ g1, g0, g3 ],
            [ 1., 1.,  1.]  ])
        mat3 = np.asarray([
            [ T1, T2, T0 ],
            [ g1, g2, g0 ],
            [ 1., 1.,  1.]  ])
        det0 = __det3x3__(mat0.ravel())
        det1 = __det3x3__(mat1.ravel())
        det2 = __det3x3__(mat2.ravel())
        det3 = __det3x3__(mat3.ravel())
        alpha1 = det1 / det0
        alpha2 = det2 / det0
        alpha3 = det3 / det0
        alpha4 = 0.

        # If (T, g) is outside the triangle formed by the three used points,
        # use only two points.
        if ((alpha1 < 0.) | (alpha1 > 1.) | (alpha2 < 0.) | (alpha2 > 1.) | (alpha3 < 0.) | (alpha3 > 1.) ):
            alpha1 = 0.
            i1 = -1
            if (T2 == T3):
                alpha2 = 0.5
            else:
                alpha2 = (T0 - T3) / (T2 - T3)
            # endif
            alpha3 = 1. - alpha2
            alpha4 = 0.
        # endif
    # endif

    # All four points used.

    if ( (i3 >= 0) & (i4 >= 0) & (i1 >= 0) & (i2 >= 0) ):
        if (T1 != T3):
            alpha = (T0 - T3) / (T1 - T3)
        else:
            alpha = 0.5
        # endif
        if (T2 != T4):
            beta = (T0 - T4) / (T2 - T4)
        else:
            beta = 0.5
        # endif
        gprim = alpha * g1 + (1 - alpha) * g3
        gsec  = beta * g2  + (1 - beta ) * g4
        if (gprim != gsec):
            gamma = ( g0 - gsec ) / ( gprim - gsec )
        else:
            gamma = 0.5
        # endif
        alpha1 = alpha * gamma
        alpha2 = beta * ( 1 - gamma )
        alpha3 = ( 1 - alpha ) * gamma
        alpha4 = (  1 - beta ) * ( 1 - gamma )
    # endif
    return np.asarray((i1, i2, i3, i4)), np.asarray((alpha1, alpha2, alpha3, alpha4))


class LejeuneInterpolator(BaseInterpolator):
    """ Interpolation for grid based on the Lejeune library definition

    The interpolation is N-D linear in log-temperature, log-gravity, and linear
    in metallicity Z. Preference is always given to the temperature over the
    gravity when needed.

    This version is translated from Pegase

    Attributes
    ----------

    dT_max: float, optional
        If, T2 (resp. T1) is too far from T compared to T1 (resp. T2), i2
        (resp. i1) is not used.  (see below for namings)

    eps: float
        temperature sensitivity under which points are considered to have
        the same temperature
    """
    def __init__(self, dT_max=0.1, eps=1e-6, *args, **kwargs):
        BaseInterpolator.__init__(self, *args, **kwargs)
        self.dT_max = 0.1
        self.eps = eps

    def interp(self, T0, g0, Z0, L0, T, g, Z, weight=1.):
        """ Interpolation of the T,g grid

        Interpolate on the grid and returns star indices and associated weights,
        and Z.

        3 to 12 stars are returned.
        It calls _interp_, but reduce the output to the relevant stars.

        Parameters
        ----------
        T0: float
            log(Teff) to obtain

        g0: float
            log(g) to obtain

        Z0: float
            metallicity to obtain

        L0: float
            bolometric luminosity to get

        T: float
            log(Teff) of the grid

        g: float
            log(g) of the grid

        weight: float, optional
            weight to apply to the selected star

        Returns
        -------
        idx: ndarray, dtype=int, size=4
            4 star indexes

        w: ndarray, dtype=float, size=4
            4 associated weights

        Returns 3 to 12 star indexes and associated weights

        see _interp_

        TODO: compute new weights accounting for Z
        """
        dT_max = self.dT_max
        eps = self.eps
        _Z    = Z
        _Zv   = np.unique(_Z)
        _T    = np.asarray(T)
        _g    = np.asarray(g)

        bZ_m  = True in (_Zv == Z0)  # Z_match bool
        r     = np.where((_Zv < Z0))[0]
        Z_inf = _Zv[r.max()] if len(r) > 0 else -1.
        r     = np.where((_Zv > Z0))[0]
        Z_sup = _Zv[r.min()] if len(r) > 0 else -1.

        index   = np.zeros(4 * 3) - 1
        weights = np.zeros(4 * 3)
        Z       = np.zeros(4 * 3)

        if weight is None:
            weight = 1.

        if (bZ_m):
            ind         = np.where(_Z == Z0)
            i, w        = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
            index[8:]   = ind[0][i]
            weights[8:] = w
            Z[8:]       = [Z0] * 4
        else:
            if (Z_inf > 0.):
                ind         = np.where(_Z == Z_inf)
                i, w        = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
                index[:4]   = ind[0][i]
                weights[:4] = w
                Z[:4]       = [Z_inf] * 4

            if (Z_sup > 0.):
                ind          = np.where(_Z == Z_sup)
                i, w         = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
                index[4:8]   = ind[0][i]
                weights[4:8] = w
                Z[4:8]       = [Z_sup] * 4

            if ((Z_inf > 0.) & (Z_sup > 0.)):
                if ( Z_sup - Z_inf ) > 0.:
                    fz = (Z0 - Z_inf) / ( Z_sup - Z_inf )
                    weights[:4]  *= fz
                    weights[4:8] *= ( 1. - fz )
                else:
                    weights[:8]  *= 0.5

        ind = np.where(weights > 0)
        return index[ind].astype(int), 10 ** L0 * weight * weights[ind]  # / (weights[ind].sum()) #, Z[ind]

    def interp_single(self, args, **kwargs):
        """ proxy to interp with a list of arguements """
        return np.asarray(self.interp(args[0], args[1], args[2], args[3],
                                      args[4], args[5], args[6], args[7],
                                      args[8], args[9])).T

    def interp_many(self, oSL, logT, logg, Z, logL, weights=None, pool=None, nthreads=0):
        """ Interpolate multiple stars at the same time (optionally in parallel)

        Parameters
        ----------
        oSL: Stellib instance
            spectral library to interpolate

        logT: ndarray, dtype=float
            log-effective temperature to get

        logg: ndarray, dtype=float
            log-surface gravity to get

        Z: float
            metallicity target

        logL: ndarray
            log-luminosity to get

        weights: ndarray, dtype=float, optional
            optional weighting scheme for each star

        pool: Pool instance, optional
            pool that can be used to do the computation (expecting a map attribute)

        nthreads: int, optiona
            if set without providing a pool arguement, configure a new Pool object
            to do the computations

        Returns
        -------
        (idx, w): tuples
            idx: ndarray, dtype=int, size=4
                4 star indexes

            w: ndarray, dtype=float, size=4
                4 associated weights
        """
        if weights is None:
            seq = [(logT[k], logg[k], Z, logL[k], oSL.logT, oSL.logg, oSL.Z, 1.)
                   for k in range(len(logT)) ]
        else:
            seq = [ (logT[k], logg[k], Z, logL[k], oSL.logT, oSL.logg, oSL.Z,
                     weights[k]) for k in range(len(logT)) ]

        if (pool is not None):
            r = pool.map( self.interp_single, seq )
        else:
            # using ezmap map function this takes care of closing the pool
            r = map(self.interp_single, seq, ncpu=nthreads)
        return np.vstack(r)
