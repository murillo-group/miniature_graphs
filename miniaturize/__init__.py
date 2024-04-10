"""
Miniaturize
===========

Python package to miniaturize large networks using the Metropolis-Hastings Algorithm
"""
import os

# Import Metropolis class created by David
from miniaturize.Metropolis import *

# Import utilities
from miniaturize.utils import * 

from importlib.util import find_spec

# Set environment variables
MINIATURIZE_DIR=find_spec('miniaturize').submodule_search_locations[0]
DATA_DIR=os.path.join(os.path.split(MINIATURIZE_DIR)[0],"data")
