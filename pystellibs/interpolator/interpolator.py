""" Base interpolator: a dummy class is derived by the different interpolator schemes """


class BaseInterpolator(object):
    def __init__(self, osl, *args, **kwargs):
        pass

    def interp(self, *args, **kwargs):
        raise NotImplementedError()

    def interp_many(self, *args, **kwargs):
        raise NotImplementedError()

    def interp_single(self, *args, **kwargs):
        raise NotImplementedError()
