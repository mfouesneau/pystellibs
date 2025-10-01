"""
N-D linear interpolation
"""

from .interpolator import BaseInterpolator
from scipy.interpolate import LinearNDInterpolator


class NDLinearInterpolator(BaseInterpolator):
    def __init__(self, osl, *args, **kwargs):
        BaseInterpolator.__init__(self, osl, *args, **kwargs)
        data = osl.get_interpolation_data()
        values = osl.spectra
        self.func = LinearNDInterpolator(data, values, **kwargs)

    def interp(self, aps, weights=1.0, **kwargs):
        """Interpolate spectra"""
        return self.func(aps) * weights

    def interp_other(self, aps, values, **kwargs):
        """Interpolate on other values"""
        f = LinearNDInterpolator(self.func.tri, values)
        return f(aps)
