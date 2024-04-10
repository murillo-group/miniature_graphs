"""
Visualization module for miniaturize library
"""
import os

from miniaturize import MINIATURIZE_DIR
from miniaturize.viz.viz import * 

# Environment variables
STYLE_SHEET_DIR=os.path.join(MINIATURIZE_DIR,"viz","sheets")
PAPER_STYLE_SHEET="viz.mplstyle"
PAPER_STYLE_SHEET_PATH=os.path.join(STYLE_SHEET_DIR,PAPER_STYLE_SHEET)
