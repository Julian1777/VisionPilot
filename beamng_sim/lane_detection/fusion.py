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
        cv_val = cv_metrics.get(key, 0.0)
        unet_val = unet_metrics.get(key, 0.0)
        fused_val = (cv_val * cv_conf + unet_val * unet_conf) / total_conf
        print(f"Fusing '{key}': CV={cv_val:.2f}, UNet={unet_val:.2f}, CV_conf={cv_conf:.2f}, UNet_conf={unet_conf:.2f} => Fused={fused_val:.2f}")
        return fused_val

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
    print(f"Fused metrics: {fused}")
    return fused