def fuse_lane_metrics(cv_metrics, cv_conf, unet_metrics, unet_conf):
    """
    Fuse lane detection metrics from CV and UNet methods based on their confidence scores.
    
    Args:
        cv_metrics (dict): Metrics from the CV-based lane detection.
        cv_conf (float): Confidence score for the CV-based detection.
        unet_metrics (dict): Metrics from the UNet-based lane detection.
        unet_conf (float): Confidence score for the UNet-based detection.
    Returns:
        dict: Fused lane detection metrics.
    """
    total_conf = cv_conf + unet_conf
    if total_conf == 0:
        return{
            'left_curverad': 0.0,
            'right_curverad': 0.0,
            'deviation': 0.0,
            'smoothed_deviation': 0.0,
            'effective_deviation': 0.0,
            'lane_center': 0.0,
            'vehicle_center': 0.0,
            'confidence': 0.0
        }
    def weighted(key):
        return (cv_metrics.get(key, 0.0) * cv_conf + unet_metrics.get(key, 0.0) * unet_conf) / total_conf
    
    fused = {
        'left_curverad': weighted('left_curverad'),
        'right_curverad': weighted('right_curverad'),
        'deviation': weighted('deviation'),
        'smoothed_deviation': weighted('smoothed_deviation'),
        'effective_deviation': weighted('effective_deviation'),
        'lane_center': weighted('lane_center'),
        'vehicle_center': weighted('vehicle_center'),
        'confidence': max(cv_conf, unet_conf)
    }
    return fused