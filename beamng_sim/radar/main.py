
def process_frame(radar_sensor, camera_detections, speed):
    try:
        radar_data = radar_sensor.poll()
        
        if radar_data is None or len(radar_data) == 0:
            return []

        filtered_points = []

        for point in radar_data:
            range_val, doppler_v, azimuth, elevation, rcs, snr = point
            doppler_speed = abs(doppler_v)

            if 2 < doppler_speed < 50:
                if -30 < azimuth < 30:
                    if range_val < 60:
                        filtered_points.append(point)

        return filtered_points
        
    except Exception as e:
        print(f"Radar processing error: {e}")
        return []
    