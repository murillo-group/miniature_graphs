"""
Miniaturize
===========

Python package to miniaturize large networks using the Metropolis-Hastings Algorithm
"""

# Import Metropolis class created by David
from miniaturize.Metropolis import *

# Import utilities
from miniaturize.utils import * 

from importlib.util import find_spec

MINIATURIZE_DIR=find_spec('miniaturize').submodule_search_locations[0]
