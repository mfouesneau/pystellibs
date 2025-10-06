from . import interpolator
from .astropy_units import Unit
from .basel import BaSeL
from .btsettl import BTSettl
from .elodie import Elodie
from .kurucz import Kurucz
from .marcs import Marcs
from .munari import Munari
from .rauch import Rauch
from .stellib import AtmosphereLib, CompositeStellib, Stellib
from .tlusty import Tlusty

__all__ = [
    "AtmosphereLib",
    "BaSeL",
    "BTSettl",
    "CompositeStellib",
    "Elodie",
    "Kurucz",
    "Marcs",
    "Munari",
    "Rauch",
    "Stellib",
    "Tlusty",
    "Unit",
    "interpolator",
]

libraries = {
    "basel": BaSeL,
    "btsettl": BTSettl,
    "elodie": Elodie,
    "kurucz": Kurucz,
    "marcs": Marcs,
    "munari": Munari,
    "rauch": Rauch,
    "tlusty": Tlusty,
}
