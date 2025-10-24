"""
SCNN Lane Detection Module
"""

from .scnn_model import SCNN
from .postprocess import run_scnn_on_frame

__all__ = ['SCNN', 'run_scnn_on_frame']
