from __future__ import print_function
import os
import inspect

from importlib import resources

#directories
__ROOT__ = '/'.join(os.path.abspath(inspect.getfile(inspect.currentframe())).split('/')[:-1])
# default library directory
# libsdir = os.path.abspath(os.path.join(__ROOT__, '../libs/'))
libsdir = os.path.join(resources.files('pystellibs'), 'libs')
