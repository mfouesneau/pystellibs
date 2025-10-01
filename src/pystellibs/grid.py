from dataclasses import dataclass
from typing import Dict

import numpy.typing as npt


@dataclass
class Grid:
    """A simple table class to hold data and header information."""

    data: npt.NDArray
    header: Dict

    def __getitem__(self, key):
        return self.data[key]
