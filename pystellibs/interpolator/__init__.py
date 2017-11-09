from .interpolator import BaseInterpolator
from .lejeune import LejeuneInterpolator
from .ndlinear import NDLinearInterpolator


def find_interpolator(name, osl=None, **kwargs):
    """ Find an interpolator from its name and 
        instanciate it if an osl was provided

    Parameters
    ----------
    name: str
        name of the interpolation
    osl: Stellib instance, optional
        library to work with
    """
    mapping = {"lejeune": LejeuneInterpolator,
               "ndlinear": NDLinearInterpolator,
               "lejeuneinterpolator": LejeuneInterpolator,
               "ndlinearinterpolator": NDLinearInterpolator}

    try:
        cls = mapping.get(name.lower(), None)
        if cls is not None:
            if osl is not None:
                return cls(osl, **kwargs)
            else:
                return cls
    except AttributeError:
        pass

    return None
