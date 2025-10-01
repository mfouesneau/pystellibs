"""
Declare missing photometric and spectral units for use with astropy.
"""
from typing import Any

from astropy.units import Unit, def_unit, add_enabled_units
from astropy.units import Quantity

__all__ = ["Unit", "Quantity", "has_unit"]

new_units = dict(
    flam="erg * s ** (-1) * AA ** (-1) * cm **(-2)",
    fnu="erg * s ** (-1) * Hz ** (-1) * cm **(-2)",
    photflam="photon * s ** (-1) * AA ** (-1) * cm **(-2)",
    photfnu="photon * s ** (-1) * Hz ** (-1) * cm **(-2)",
    angstroms="angstrom",
    lsun="Lsun",
    ergs="erg"
)

add_enabled_units([def_unit([k], Unit(v)) for k, v in new_units.items()]).__enter__()

def has_unit(val: Any) -> bool:
    """Check if a unit is defined in astropy."""
    return hasattr(val, 'units')
