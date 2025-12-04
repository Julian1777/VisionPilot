"""
Shared Features:
    1. Number of detected lane lines
    2. Length/Continuity of each detected lane line
    3. Proportion of ROI covered by detected lanes
    4. Lane Geometry, are the two lanes parallel, are they posittioned with a reasonable distance
    5. Constancy over time, how stable is the detection over multiple frames
"""

import numpy as np

weights = {
    'num_lines': 0.2,
    'length': 0.2,
    'geometry': 0.4,
    'temporal': 0.2  # Add temporal consistency weight
}


def num_lane_lines(left_fitx, right_fitx):
    """
    Checks if the detection has both a left and right lane
    Args:
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        int: number of detected lane lines (0, 1, or 2)
    """
    left_hand_detected = left_fitx is not None and len(left_fitx) > 0
    right_hand_detected = right_fitx is not None and len(right_fitx) > 0

    num_lanes = (1 if left_hand_detected else 0) + (1 if right_hand_detected else 0)

    if num_lanes == 2:
        num_lanes_score = 1.0
    elif num_lanes == 1:
        num_lanes_score = 0.5
    else:
        num_lanes_score = 0.0

    return num_lanes_score

def lane_length_continuity(ploty, left_fitx, right_fitx):
    """
    Computes the length and continuity of each detected lane line
    Args:
        ploty (np.array): y coordinates of the lane lines
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        float: confidence score based on lane length and continuity
    """
    if ploty is None or len(ploty) == 0:
        return 0.0
    
    expected_length = len(ploty)

    left_valid = 0 if left_fitx is None else np.count_nonzero(~np.isnan(left_fitx))
    right_valid = 0 if right_fitx is None else np.count_nonzero(~np.isnan(right_fitx))

    if left_valid > 0 and right_valid > 0:
        avg_continuity = (left_valid + right_valid) / (2 * expected_length)
    elif left_valid > 0:
        avg_continuity = left_valid / expected_length
    elif right_valid > 0:
        avg_continuity = right_valid / expected_length
    else:
        avg_continuity = 0.0
    
    return min(1.0, avg_continuity)
    


#def lane_coverage():
    # Placeholder for lane coverage function

def lane_geometry(left_fitx, right_fitx, ploty):
    """
    Evaluates the geometry of the detected lanes
    Args:
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        float: confidence score based on lane geometry
    """

    if left_fitx is None or right_fitx is None or len(left_fitx) == 0 or len(right_fitx) == 0:
        return 0.0
    
    try:
        indices = [int(len(ploty)*0.9), int(len(ploty)*0.5), int(len(ploty)*0.1)]
        lane_widths = []
        
        for idx in indices:
            if idx < len(left_fitx) and idx < len(right_fitx):
                if not np.isnan(left_fitx[idx]) and not np.isnan(right_fitx[idx]):
                    width = right_fitx[idx] - left_fitx[idx]
                    lane_widths.append(width)
        
        if len(lane_widths) == 0:
            return 0.0
        
        reasonable_width_min = 100  # Minimum reasonable lane width in pixels
        reasonable_width_max = 700  # Maximum reasonable lane width in pixels
        
        width_scores = []
        for width in lane_widths:
            if width < 0:
                width_scores.append(0.0)
            elif reasonable_width_min <= width <= reasonable_width_max:
                width_scores.append(1.0)
            else:
                deviation = min(abs(width - reasonable_width_min), abs(width - reasonable_width_max))
                width_scores.append(max(0.0, 1.0 - deviation / reasonable_width_max))
        
        if len(width_scores) >= 2:
            width_variance = np.std(lane_widths) / np.mean(lane_widths)
            parallelism_score = max(0.0, 1.0 - width_variance)
        else:
            parallelism_score = 0.5
        
        return 0.7 * np.mean(width_scores) + 0.3 * parallelism_score
        
    except Exception as e:
        print(f"Error in lane_geometry: {e}")
        return 0.0

def temporal_consistency(current_fit, previous_fit):
    """
    Assesses the stability of lane detection over multiple frames
    Args:
        current_fit (tuple): current frame's lane line polynomial coefficients
        previous_fit (tuple): previous frame's lane line polynomial coefficients
    Returns:
        float: confidence score based on temporal consistency
    """

    if current_fit is None or previous_fit is None:
        return 0.5
    
    try:
        left_current, right_current = current_fit
        left_previous, right_previous = previous_fit
        
        if left_current is None or right_current is None or left_previous is None or right_previous is None:
            return 0.3
        
        left_diff = np.sum(np.abs(np.array(left_current) - np.array(left_previous)))
        right_diff = np.sum(np.abs(np.array(right_current) - np.array(right_previous)))
        
        max_expected_diff = 1.0
        left_score = max(0.0, 1.0 - left_diff / max_expected_diff)
        right_score = max(0.0, 1.0 - right_diff / max_expected_diff)
        
        return (left_score + right_score) / 2.0
        
    except Exception as e:
        print(f"Error in temporal_consistency: {e}")
        return 0.5

def compute_shared_confidence(left_fitx, right_fitx, ploty, current_fit=None, previous_fit=None):

    num_lines_score = num_lane_lines(left_fitx, right_fitx)
    length_continuity_score = lane_length_continuity(ploty, left_fitx, right_fitx)
    geometry_score = lane_geometry(left_fitx, right_fitx, ploty)
    temporal_score = temporal_consistency(current_fit, previous_fit)


    return {
        'num_lines_score': num_lines_score,
        'length_continuity_score': length_continuity_score,
        'geometry_score': geometry_score,
        'temporal_score': temporal_score,
    }


def compute_confidence_cv(left_fitx, right_fitx, ploty, current_fit=None, previous_fit=None):
    shared_score = compute_shared_confidence(left_fitx, right_fitx, ploty, current_fit=current_fit, previous_fit=previous_fit)

    # Add CV-specific metrics
    
    # Combine scores
    confidence = (
        weights['num_lines'] * shared_score['num_lines_score'] +
        weights['length'] * shared_score['length_continuity_score'] +
        weights['geometry'] * shared_score['geometry_score'] +
        weights['temporal'] * shared_score['temporal_score']
        # weights['cv_specific'] * cv_specific_score
    ) / sum(weights.values())
    
    return min(max(confidence, 0.0), 1.0)


def compute_confidence_scnn(left_fitx, right_fitx, ploty, current_fit=None, previous_fit=None, 
                            exist_pred=None, segmentation_quality=None):
    """
    Compute confidence for SCNN lane detection.
    
    SCNN-specific advantages:
    - Existence predictions: Direct probability for each of 6 lanes
    - Segmentation quality: Average confidence across detected lane pixels
    
    Args:
        left_fitx, right_fitx, ploty: Standard lane line parameters
        current_fit, previous_fit: For temporal consistency
        exist_pred (np.array): SCNN existence predictions (6 values, 0-1)
        segmentation_quality (float): Average segmentation confidence
        
    Returns:
        float: Confidence score (0-1)
    """
    # Get shared confidence scores (geometry, temporal, etc.)
    shared_score = compute_shared_confidence(left_fitx, right_fitx, ploty, 
                                            current_fit=current_fit, 
                                            previous_fit=previous_fit)
    
    # SCNN-specific: Existence prediction confidence
    existence_score = 0.5  # Default if not provided
    if exist_pred is not None:
        # The middle lanes (indices 2, 3) are typically the ego lane
        # We care most about detecting at least the lanes around us
        try:
            # Average of all lane existence predictions, weighted toward center lanes
            lane_weights = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])  # Center lanes weighted higher
            existence_score = np.sum(exist_pred * lane_weights)
            
            # Bonus if we have at least 2 lanes with high confidence
            high_conf_lanes = np.sum(exist_pred > 0.5)
            if high_conf_lanes >= 2:
                existence_score = min(1.0, existence_score * 1.1)
                
        except Exception as e:
            print(f"Error computing SCNN existence score: {e}")
            existence_score = 0.5
    
    # SCNN-specific: Segmentation quality (average confidence of predicted lane pixels)
    seg_quality_score = 0.5  # Default if not provided
    if segmentation_quality is not None:
        seg_quality_score = min(1.0, max(0.0, segmentation_quality))
    
    # Adjusted weights for SCNN (existence and segmentation quality are very valuable)
    scnn_weights = {
        'num_lines': 0.15,
        'length': 0.15,
        'geometry': 0.25,
        'temporal': 0.15,
        'existence': 0.20,      # SCNN-specific: direct lane existence probability
        'seg_quality': 0.10     # SCNN-specific: segmentation confidence
    }
    
    # Combine scores
    confidence = (
        scnn_weights['num_lines'] * shared_score['num_lines_score'] +
        scnn_weights['length'] * shared_score['length_continuity_score'] +
        scnn_weights['geometry'] * shared_score['geometry_score'] +
        scnn_weights['temporal'] * shared_score['temporal_score'] +
        scnn_weights['existence'] * existence_score +
        scnn_weights['seg_quality'] * seg_quality_score
    ) / sum(scnn_weights.values())
    
    return min(max(confidence, 0.0), 1.0)