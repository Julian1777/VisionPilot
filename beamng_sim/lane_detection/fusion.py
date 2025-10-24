def fuse_lane_metrics(cv_metrics, cv_conf, dl_metrics, dl_conf, method_name="DL"):
    """
    Fuse lane detection metrics from CV and Deep Learning methods based on their confidence scores.
    CV is weighted more heavily (65%) as it is generally more trustworthy, while SCNN/DL (35%) is still used.
    
    Args:
        cv_metrics (dict): Metrics from the CV-based lane detection.
        cv_conf (float): Confidence score for the CV-based detection.
        dl_metrics (dict): Metrics from the DL-based lane detection (UNet or SCNN).
        dl_conf (float): Confidence score for the DL-based detection.
        method_name (str): Name of DL method for logging ("UNet" or "SCNN")
    Returns:
        dict: Fused lane detection metrics.
    """
    cv_weight = 0.65
    dl_weight = 0.35
    
    if cv_conf == 0.0 and dl_conf == 0.0:
        return{
            'left_curverad': 0.0,
            'right_curverad': 0.0,
            'deviation': 0.0,
            'smoothed_deviation': 0.0,
            'effective_deviation': 0.0,
            'lane_center': 0.0,
            'vehicle_center': 0.0,
            'confidence': 0.0,
        }
    
    def weighted(key):
        cv_val = cv_metrics.get(key, 0.0)
        dl_val = dl_metrics.get(key, 0.0)
        
        cv_val = 0.0 if cv_val is None else float(cv_val)
        dl_val = 0.0 if dl_val is None else float(dl_val)
        
        cv_conf_norm = cv_conf if cv_conf > 0 else 0.5
        dl_conf_norm = dl_conf if dl_conf > 0 else 0.5
        
        fused_val = (cv_val * cv_weight) + (dl_val * dl_weight)
        print(f"Fusing '{key}': CV={cv_val:.2f}, {method_name}={dl_val:.2f}, CV_conf={cv_conf:.2f}, {method_name}_conf={dl_conf:.2f} => Fused={fused_val:.2f}")
        return fused_val

    fused = {
        'left_curverad': weighted('left_curverad'),
        'right_curverad': weighted('right_curverad'),
        'deviation': weighted('deviation'),
        'smoothed_deviation': weighted('smoothed_deviation'),
        'effective_deviation': weighted('effective_deviation'),
        'lane_center': weighted('lane_center'),
        'vehicle_center': weighted('vehicle_center'),
        'confidence': (cv_conf * cv_weight + dl_conf * dl_weight)
    }
    print(f"Fused metrics (CV 65%, {method_name} 35%): {fused}")
    return fused