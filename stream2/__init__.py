"""STREAM2."""

from ._settings import settings
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from .readwrite import *

__version__ = "0.1a"

import sys
sys.modules.update(
    {f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl']})