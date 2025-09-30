from typing import Type
from .interpolator import BaseInterpolator
from .lejeune import LejeuneInterpolator
from .ndlinear import NDLinearInterpolator


def find_interpolator(name: str) -> Type[BaseInterpolator]:
    """Find an interpolator from its name and instanciate it if an osl was provided

    Parameters
    ----------
    name: str
        name of the interpolation

    Returns
    -------
    interpolator instance or class if no osl was provided, None if not found
    """
    mapping = {
        "lejeune": LejeuneInterpolator,
        "ndlinear": NDLinearInterpolator,
        "lejeuneinterpolator": LejeuneInterpolator,
        "ndlinearinterpolator": NDLinearInterpolator,
    }
    if isinstance(name, BaseInterpolator):
        return name

    cls = mapping.get(name.lower(), None)
    if cls is not None:
        return cls
    else:
        raise ValueError(f"Interpolator '{name}' not found")