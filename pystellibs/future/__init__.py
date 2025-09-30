""" This module provides a compatibility layer for the `Path` class from `matplotlib.path`."""
import matplotlib.path
if hasattr(matplotlib.path.Path, 'contains_points'):
    from matplotlib.path import Path
else:
    from .path import Path
