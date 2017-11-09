""" Base interpolator: a dummy class is derived by the different interpolator schemes """


class BaseInterpolator(object):
    """ Base class for interpolation

    It sets what can be expected as methods during the interpolation calls
    """
    def __init__(self, osl, *args, **kwargs):
        pass

    def interp(self, aps, *args, **kwargs):
        """ Interpolation over spectra """
        raise NotImplementedError()

    def interp_other(self, aps, values, *args, **kwargs):
        """ Interpolation over provided values """
        raise NotImplementedError()
