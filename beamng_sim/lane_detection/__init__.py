"""
Lane Detection Module

This module provides a complete lane detection pipeline with the following components:

- thresholding.py: Color and gradient thresholding for lane line detection
- perspective.py: Perspective transformation for bird's eye view
- lane_finder.py: Sliding window search to find lane lines
- metrics.py: Calculate curvature and deviation metrics
- visualization.py: Draw lane overlay and text information
- main.py: Main processing function that ties everything together

Usage:
    from beamng_sim.lane_detection import process_frame
    
    result_image, metrics = process_frame(input_frame)
"""

from .main import process_frame

__all__ = ['process_frame']
