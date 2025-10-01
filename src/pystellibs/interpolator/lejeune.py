"""Lejeune interpolator is basically a linear interpolator for a Lejeune grid
based spectral library.

This is the simplest interpolator but most commonly used.

It takes care of boundary conditions by imposing limits to extrapolation on a
given grid.
"""

import numpy as np
from .interpolator import BaseInterpolator


def __det3x3__(a):
    """compute the 3x3 determinant of an array

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
    val = +a[0] * (a[4] * a[8] - a[7] * a[5])
    val += -a[1] * (a[3] * a[8] - a[6] * a[5])
    val += +a[2] * (a[3] * a[7] - a[6] * a[4])
    return val


def __interp__(T0, g0, T, g, dT_max=0.1, eps=1e-6):
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
    kappa = 0.1

    idx = np.arange(len(g))
    deltag = g - g0
    deltaT = T - T0
    dist = kappa * abs(deltag) + abs(deltaT)

    if dist.min() == 0:
        return np.array((dist.argmin(), -1, -1, -1)), np.array((1.0, 0.0, 0.0, 0.0))

    # Looking for i_{1..4}
    ind_dT = deltaT >= 0
    ind_dg = deltag >= 0

    # i1
    ind = ind_dT & ind_dg
    if True in ind:
        i1 = idx[ind][dist[ind].argmin()]
    else:
        i1 = -1

    # i2
    ind = ind_dT & ~ind_dg
    if True in ind:
        i2 = idx[ind][dist[ind].argmin()]
    else:
        i2 = -1

    # i3
    ind = ~ind_dT & ind_dg
    if True in ind:
        i3 = idx[ind][dist[ind].argmin()]
    else:
        i3 = -1

    # i4
    ind = ~ind_dT & ~ind_dg
    if True in ind:
        i4 = idx[ind][dist[ind].argmin()]
    else:
        i4 = -1

    # checking integrity
    if (i1 < 0) & (i2 < 0) & (i3 < 0) & (i4 < 0):
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
    if (i1 > 0) & (i2 > 0):
        if T1 < T2 - dT_max:
            i2 = -1
        elif T2 < T1 - dT_max:
            i1 = -1

    if (i3 > 0) & (i4 > 0):
        if T3 > T4 + dT_max:
            i4 = -1
        elif T4 > T3 + dT_max:
            i3 = -1

    if (i1 < 0) & (i2 < 0) & (i3 < 0) & (i4 < 0):
        raise ValueError("Interp. Error, could not find appropriate knots")

    # Interpolation in the (T, g) plane between the used points
    # (at least 1, at most 4).
    # Code "0110" means that i1 = i4 = 0, i2 /=0 and i3 /= 0.
    #
    # Note: preference is always given to the temperature over
    #   the gravity when needed.
    if i1 < 0:
        if i2 < 0:
            if i3 < 0:
                if i4 < 0:
                    #                   # 0000
                    raise ValueError("Error")  # should not be possible
                else:  # 0001
                    alpha1 = 0.0
                    alpha2 = 0.0
                    alpha3 = 0.0
                    alpha4 = 1.0
                # endif
            elif i4 < 0:  # 0010
                alpha1 = 0.0
                alpha2 = 0.0
                alpha3 = 1.0
                alpha4 = 0.0
            else:  # 0011
                alpha1 = 0.0
                alpha2 = 0.0
                if abs(T3 - T4) < eps:
                    if g3 == g4:
                        alpha3 = 0.5
                    else:
                        alpha3 = (g0 - g4) / (g3 - g4)
                    # endif
                    alpha4 = 1.0 - alpha3
                else:
                    if T3 > T4:
                        alpha3 = 1.0
                        alpha4 = 0.0
                        i4 = -1
                    else:
                        alpha3 = 0.0
                        i3 = -1
                        alpha4 = 1.0
                    # endif
                # endif
            # endif
        elif i3 < 0:
            if i4 < 0:
                #                        # 0100
                alpha1 = 0.0
                alpha2 = 1.0
                alpha3 = 0.0
                alpha4 = 0.0
            else:  # 0101
                alpha1 = 0.0
                if T2 == T4:
                    alpha2 = 0.5
                else:
                    alpha2 = (T0 - T4) / (T2 - T4)
                # endif
                alpha3 = 0.0
                alpha4 = 1.0 - alpha2
            # endif
        elif i4 < 0:  # 0110
            alpha1 = 0.0
            if T2 == T3:
                alpha2 = 0.5
            else:
                alpha2 = (T0 - T3) / (T2 - T3)
            # endif
            alpha3 = 1.0 - alpha2
            alpha4 = 0.0
        else:  # 0111
            # Assume that (T, g) is within the triangle i
            # formed by the three points.

            mat0 = np.asarray([[T2, T3, T4], [g2, g3, g4], [1.0, 1.0, 1.0]])
            mat2 = np.asarray([[T0, T3, T4], [g0, g3, g4], [1.0, 1.0, 1.0]])
            mat3 = np.asarray([[T2, T0, T4], [g2, g0, g4], [1.0, 1.0, 1.0]])
            mat4 = np.asarray([[T2, T3, T0], [g2, g3, g0], [1.0, 1.0, 1.0]])
            det0 = __det3x3__(mat0.ravel())
            det2 = __det3x3__(mat2.ravel())
            det3 = __det3x3__(mat3.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = 0.0
            alpha2 = det2 / det0
            alpha3 = det3 / det0
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed
            # by the three used points use only two points.
            if (
                (alpha2 < 0.0)
                | (alpha2 > 1.0)
                | (alpha3 < 0.0)
                | (alpha3 > 1.0)
                | (alpha4 < 0.0)
                | (alpha4 > 1.0)
            ):
                alpha1 = 0.0
                if T2 == T3:
                    alpha2 = 0.5
                else:
                    alpha2 = (T0 - T3) / (T2 - T3)
                # endif
                alpha3 = 1.0 - alpha2
                alpha4 = 0.0
                i4 = -1
            # endif
        # endif
    elif i2 < 0:
        if i3 < 0:
            if i4 < 0:
                #                      # 1000
                alpha1 = 1.0
                alpha2 = 0.0
                alpha3 = 0.0
                alpha4 = 0.0
            else:  # 1001
                if T1 == T4:
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.0
                alpha3 = 0.0
                alpha4 = 1.0 - alpha1
            # endif
        elif i4 < 0:  # 1010
            if T1 == T3:
                alpha1 = 0.5
            else:
                alpha1 = (T0 - T3) / (T1 - T3)
            # endif
            alpha2 = 0.0
            alpha3 = 1.0 - alpha1
            alpha4 = 0.0
        else:  # 1011
            # Assume that (T, g) is within the triangle formed by the three points.
            mat0 = np.asarray([[T1, T3, T4], [g1, g3, g4], [1.0, 1.0, 1.0]])
            mat1 = np.asarray([[T0, T3, T4], [g0, g3, g4], [1.0, 1.0, 1.0]])
            mat3 = np.asarray([[T1, T0, T4], [g1, g0, g4], [1.0, 1.0, 1.0]])
            mat4 = np.asarray([[T1, T3, T0], [g1, g3, g0], [1.0, 1.0, 1.0]])
            det0 = __det3x3__(mat0.ravel())
            det1 = __det3x3__(mat1.ravel())
            det3 = __det3x3__(mat3.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = det1 / det0
            alpha2 = 0.0
            alpha3 = det3 / det0
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed by the three used points,
            # use only two points.

            if (
                (alpha1 < 0.0)
                | (alpha1 > 1.0)
                | (alpha3 < 0.0)
                | (alpha3 > 1.0)
                | (alpha4 < 0.0)
                | (alpha4 > 1.0)
            ):
                if T1 == T4:
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.0
                alpha3 = 0.0
                i3 = -1
                alpha4 = 1.0 - alpha1
            # endif
        # endif
    elif i3 < 0:
        if i4 < 0:
            #                       # 1100
            if abs(T1 - T2) < eps:
                if g1 == g2:
                    alpha1 = 0.5
                else:
                    alpha1 = (g0 - g2) / (g1 - g2)
                # endif
                alpha2 = 1.0 - alpha1
            else:
                if T1 < T2:
                    alpha1 = 1.0
                    alpha2 = 0.0
                    i2 = -1
                else:
                    alpha1 = 0.0
                    i1 = -1
                    alpha2 = 1.0
                # endif
            # endif
            alpha3 = 0.0
            alpha4 = 0.0
        else:  # 1101
            # Assume that (T, g) is within the triangle formed by the three points.
            mat0 = np.asarray([[T1, T2, T4], [g1, g2, g4], [1.0, 1.0, 1.0]])
            mat1 = np.asarray([[T0, T2, T4], [g0, g2, g4], [1.0, 1.0, 1.0]])
            mat2 = np.asarray([[T1, T0, T4], [g1, g0, g4], [1.0, 1.0, 1.0]])
            mat4 = np.asarray([[T1, T2, T0], [g1, g2, g0], [1.0, 1.0, 1.0]])
            det0 = __det3x3__(mat0.ravel())
            det1 = __det3x3__(mat1.ravel())
            det2 = __det3x3__(mat2.ravel())
            det4 = __det3x3__(mat4.ravel())
            alpha1 = det1 / det0
            alpha2 = det2 / det0
            alpha3 = 0.0
            alpha4 = det4 / det0

            # If (T, g) is outside the triangle formed by the three used points,
            # use only two points.
            if (
                (alpha1 < 0.0)
                | (alpha1 > 1.0)
                | (alpha2 < 0.0)
                | (alpha2 > 1.0)
                | (alpha4 < 0.0)
                | (alpha4 > 1.0)
            ):
                if T1 == T4:
                    alpha1 = 0.5
                else:
                    alpha1 = (T0 - T4) / (T1 - T4)
                # endif
                alpha2 = 0.0
                i2 = -1
                alpha3 = 0.0
                alpha4 = 1.0 - alpha1
            # endif
        # endif
    elif i4 < 0:
        #                           # 1110
        # Assume that (T, g) is within the triangle formed by the three points.
        mat0 = np.asarray([[T1, T2, T3], [g1, g2, g3], [1.0, 1.0, 1.0]])
        mat1 = np.asarray([[T0, T2, T3], [g0, g2, g3], [1.0, 1.0, 1.0]])
        mat2 = np.asarray([[T1, T0, T3], [g1, g0, g3], [1.0, 1.0, 1.0]])
        mat3 = np.asarray([[T1, T2, T0], [g1, g2, g0], [1.0, 1.0, 1.0]])
        det0 = __det3x3__(mat0.ravel())
        det1 = __det3x3__(mat1.ravel())
        det2 = __det3x3__(mat2.ravel())
        det3 = __det3x3__(mat3.ravel())
        alpha1 = det1 / det0
        alpha2 = det2 / det0
        alpha3 = det3 / det0
        alpha4 = 0.0

        # If (T, g) is outside the triangle formed by the three used points,
        # use only two points.
        if (
            (alpha1 < 0.0)
            | (alpha1 > 1.0)
            | (alpha2 < 0.0)
            | (alpha2 > 1.0)
            | (alpha3 < 0.0)
            | (alpha3 > 1.0)
        ):
            alpha1 = 0.0
            i1 = -1
            if T2 == T3:
                alpha2 = 0.5
            else:
                alpha2 = (T0 - T3) / (T2 - T3)
            # endif
            alpha3 = 1.0 - alpha2
            alpha4 = 0.0
        # endif
    # endif

    # All four points used.

    if (i3 >= 0) & (i4 >= 0) & (i1 >= 0) & (i2 >= 0):
        if T1 != T3:
            alpha = (T0 - T3) / (T1 - T3)
        else:
            alpha = 0.5
        # endif
        if T2 != T4:
            beta = (T0 - T4) / (T2 - T4)
        else:
            beta = 0.5
        # endif
        gprim = alpha * g1 + (1 - alpha) * g3
        gsec = beta * g2 + (1 - beta) * g4
        if gprim != gsec:
            gamma = (g0 - gsec) / (gprim - gsec)
        else:
            gamma = 0.5
        # endif
        alpha1 = alpha * gamma
        alpha2 = beta * (1 - gamma)
        alpha3 = (1 - alpha) * gamma
        alpha4 = (1 - beta) * (1 - gamma)
    # endif
    return np.asarray((i1, i2, i3, i4)), np.asarray((alpha1, alpha2, alpha3, alpha4))


'''
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
    def __init__(self, osl, dT_max=0.1, eps=1e-6, *args, **kwargs):
        BaseInterpolator.__init__(self, osl, *args, **kwargs)
        self.dlogT_max = dT_max
        self.eps = eps
        self.osl = osl

    def interp(self, aps, weights=None, **kwargs):
        return self.interp_other(aps, self.osl.spectra, weights=weights, **kwargs)

    def interp_other(self, aps, other, weights=None, **kwargs):
        # get osl data
        osl_logT, osl_logg, osl_logZ = self.osl.get_interpolation_data().T[:3]
        grid_logZ   = np.unique(osl_logZ)
        if np.ndim(other) < 2:
            values = np.atleast_2d([other]).T
        else:
            values = np.atleast_2d(other)

        # params
        library_index = np.arange(len(osl_logT), dtype=int)
        _aps = np.atleast_2d(aps)
        if weights is None:
            _weights = np.ones(len(_aps), dtype=float)
        elif np.ndim(weights) == 0:
            _weights = np.ones(len(_aps), dtype=float) * weights
        else:
            _weights = weights

        final_values = []
        for current_aps, current_weight in zip(np.atleast_2d(aps), _weights):
            logT, logg, logZ = current_aps
            # logZ = np.log10(Z)

            # find Zsup and Zinf
            where = np.searchsorted(grid_logZ, logZ)
            if where >=0:
                logZinf = grid_logZ[where]
            else:
                raise ValueError("Metallicity extrapolation")
            if abs(logZinf - logZ) < 1e-4:
                # exact match no need to interpolate twice.
                select = (abs(logZinf - osl_logZ) < 1e-4)
                # call Pegase interpolation scheme
                #   Interpolation of the (logT, logg) grid at fixed Z from pegase.2
                #   it returns the knots'indices from the input data and their weights, resp.
                #   the final result is then the weighted sum.
                indices, alphas = __interp__(logT, logg,
                                             osl_logT[select], osl_logg[select],
                                             dT_max=self.dlogT_max, eps=self.eps)
                # indices are for the selection
                # if indices[k] = -1, then one corner is rejected
                data_indices = library_index[select][indices[indices >= 0]]
                data_weights = alphas[indices >= 0]
                spectrum = np.sum(values[data_indices] * data_weights[:, None], axis=0)
                # store the weighted sum * the input requested weight
                final_values.append(spectrum * current_weight)
            else:
                logZsup = grid_logZ[where + 1]
                # interpolate within each (therefore calling interp with Zinf, Zsup, resp.)
                # then linearly interpolate between logZ values.
                inf_spectrum = self.interp_other((logT, logg, logZinf), values, weights=current_weight, **kwargs)
                sup_spectrum = self.interp_other((logT, logg, logZsup), values, weights=current_weight, **kwargs)
                spectrum = inf_spectrum * (logZ - logZinf) / (logZsup - logZinf) + sup_spectrum * (logZsup - logZ) / (logZsup - logZinf)
                final_values.append(spectrum)
        return np.squeeze(final_values)
'''


class LejeuneInterpolator(BaseInterpolator):
    """Interpolation for grid based on the Lejeune library definition

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

    def __init__(self, osl, dT_max=0.1, eps=1e-6, *args, **kwargs):
        BaseInterpolator.__init__(self, osl, *args, **kwargs)
        self.dlogT_max = dT_max
        self.eps = eps
        self.osl = osl

    def _osl_interp_weights(self, osl, T0, g0, Z0, dT_max=0.1, eps=1e-6):
        """Interpolation of the T,g grid

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

        Returns 3 to 12 star indexes and associated weights
        """
        # interpolation must be by construction from logT, logg, Z
        # logZ could be an alternative.
        osl_logT, osl_logg, osl_logZ = self.osl.get_interpolation_data().T[:3]
        _Z = 10**osl_logZ
        _Zv = np.unique(_Z)
        _T = np.asarray(osl_logT, dtype=np.double)
        _g = np.asarray(osl_logg, dtype=np.double)

        bZ_m = True in (abs(_Zv - Z0) < 1e-28)  # Z_match bool
        r = np.where((_Zv < Z0))[0]
        Z_inf = _Zv[r.max()] if len(r) > 0 else -1.0
        r = np.where((_Zv > Z0))[0]
        Z_sup = _Zv[r.min()] if len(r) > 0 else -1.0

        index = np.zeros(4 * 3) - 1
        weights = np.zeros(4 * 3)
        Z = np.zeros(4 * 3)

        if bZ_m:
            ind = np.where((abs(_Z - Z0) < 1e-28))
            i, w = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
            index[8:] = ind[0][i]
            weights[8:] = np.squeeze(w)
            Z[8:] = [Z0] * 4
        else:
            if Z_inf > 0.0:
                ind = np.where(_Z == Z_inf)
                i, w = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
                index[:4] = ind[0][i]
                weights[:4] = np.squeeze(w)
                Z[:4] = [Z_inf] * 4

            if Z_sup > 0.0:
                ind = np.where(_Z == Z_sup)
                i, w = __interp__(T0, g0, _T[ind], _g[ind], dT_max, eps)
                index[4:8] = ind[0][i]
                weights[4:8] = np.squeeze(w)
                Z[4:8] = [Z_sup] * 4

            if (Z_inf > 0.0) & (Z_sup > 0.0):
                if (Z_sup - Z_inf) > 0.0:
                    fz = (Z0 - Z_inf) / (Z_sup - Z_inf)
                    weights[:4] *= fz
                    weights[4:8] *= 1.0 - fz
                else:
                    weights[:8] *= 0.5

        ind = np.where(weights > 0)
        return index[ind].astype(int), weights[ind]  # / (weights[ind].sum()) #, Z[ind]

    def _interp_weights(self, aps, weights=None, **kwargs):
        """returns interpolation nodes and weights

        Parameters
        ----------
        aps: ndarray
            (logT, logg, logZ) sequence.
            Or appropriately defined similarly to self.osl.get_interpolation_data
        weights: ndarray
            optional weights of each ap vector to apply during the interpolation

        Returns
        -------
        node_weights: array
            osl grid node indices and interpolation weights
        """
        _aps = np.atleast_2d(aps)

        if weights is None:
            _weights = np.ones(len(_aps), dtype=float)
        elif np.ndim(weights) == 0:
            _weights = np.ones(len(_aps), dtype=float) * weights
        else:
            _weights = weights

        node_weights = []
        for s, current_weight in zip(_aps, _weights):
            logT, logg, logZ = s[:3]
            Z = 10**logZ
            current_nodes = np.array(
                self._osl_interp_weights(self.osl, logT, logg, Z, **kwargs)
            ).T
            current_nodes[:, 1] *= current_weight
            node_weights.append(current_nodes)

        return node_weights

    def _evaluate_from_weights(self, r, other):
        """Evaluate the interpolation from interpolation nodes and weights

        Basically do a weighted sum on the grid using the interpolation weights

        Parameters
        ----------
        node_weights: array
            osl grid node indices and interpolation weights
            result of interp_weights

        other: array
            values to interpolate

        Returns
        -------
        interpolated: ndarray (size(node_weights), )
            interpolated values
        """
        if np.ndim(other) < 2:
            values = np.atleast_2d([other]).T
        else:
            values = np.atleast_2d(other)
        interpolated = [
            ((values[rk[:, 0].astype(int)].T) * rk[:, 1]).sum(1) for rk in r
        ]
        return np.squeeze(interpolated)

    def interp(self, aps, weights=None, **kwargs):
        """
        Interpolate spectra

        Parameters
        ----------
        aps: ndarray
            (logT, logg, logZ) sequence.
            Or appropriately defined similarly to self.osl.get_interpolation_data
        weights: ndarray
            optional weights of each ap vector to apply during the interpolation

        Returns
        -------
        s0: ndarray (len(aps), len(l0))
            interpolated spectra
        """
        s0 = self.interp_other(aps, self.osl.spectra, weights=weights, **kwargs)
        return s0

    def interp_other(self, aps, other, weights=None, **kwargs):
        """Interpolate other grid values

        Basically do a weighted sum on the grid using the interpolation weights

        Parameters
        ----------
        aps: ndarray
            (logT, logg, logZ) sequence.
            Or appropriately defined similarly to self.osl.get_interpolation_data
        weights: ndarray
            optional weights of each ap vector to apply during the interpolation

        Returns
        -------
        interpolated: ndarray (size(node_weights), )
            interpolated values
        """
        r = self._interp_weights(aps, weights, **kwargs)
        interpolated = self._evaluate_from_weights(r, other)
        return interpolated
