
def process_frame(radar_sensor, camera_detections, speed, debug_window=None):
    radar_data = radar_sensor.poll()

    for point in radar_data.values():
        range, doppler_v, azimuth, elevation, rcs, snr = point # rcs = radar cross section snr = signal to noise ratio
        speed = abs(doppler_v)

        filtered_points_speed = []

        if 2 < speed < 50: # vehicle
            filtered_points_speed.append(point)
        else:
            continue

        filtered_points_azimuth = []
        
        for point in filtered_points_speed:
            if -30 < azimuth < 30:
                filtered_points_azimuth.append(point)
            else:
                continue

        filtered_points = []

        for point in filtered_points_azimuth:
            if range < 60:
                filtered_points.append(point)
            else:
                continue
        

    return filtered_points
    