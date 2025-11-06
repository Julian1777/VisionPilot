import foxglove
from foxglove import Channel
import numpy as np


class FoxgloveBridge:
    def __init__(self):
        self.sign_channel = Channel("/detections/sign", message_encoding="json")
        self.traffic_light_channel = Channel("/detections/traffic_light", message_encoding="json")
        self.lane_channel = Channel("/detections/lane", message_encoding="json")
        self.vehicle_channel = Channel("/detections/vehicle", message_encoding="json")
        self.vehicle_state_channel = Channel("/vehicle/state", message_encoding="json")
        self.lidar_channel = Channel("/lidar/points", message_encoding="json")
    
    def send_sign_detection(self, sign_type, x, y, confidence):
        message = {
            "type": sign_type,
            "x": x,
            "y": y,
            "confidence": confidence
        }
        try:
            print(f"[FoxgloveBridge] Sending sign detection: {message}")
            self.sign_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending sign detection: {e}")
    
    # def send_traffic_light_detection(self, state, x, y, confidence):
    #     self.traffic_light_channel.log({
    #         "state": state,
    #         "x": x,
    #         "y": y,
    #         "confidence": confidence
    #     })
    
    def send_lane_detection(self, lane_center, vehicle_center, deviation, confidence, left_lane_points=None, right_lane_points=None):
        message = {
            "lane_center": lane_center,
            "vehicle_center": vehicle_center,
            "deviation": deviation,
            "confidence": confidence
        }
        if left_lane_points is not None:
            message["left_lane_points"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                for p in left_lane_points
            ]
        if right_lane_points is not None:
            message["right_lane_points"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                for p in right_lane_points
            ]
        try:
            print(f"[FoxgloveBridge] Sending lane detection: {message}")
            self.lane_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending lane detection: {e}")

    def send_vehicle_detection(self, detection_type, x, y, width, height, confidence):
        message = {
            "type": detection_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "confidence": confidence
        }
        try:
            print(f"[FoxgloveBridge] Sending vehicle detection: {message}")
            self.vehicle_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle detection: {e}")
    
    def send_vehicle_state(self, speed_kph, steering, throttle, x, y, z):
        # Ensure all values are standard Python floats for JSON compatibility
        message = {
            "speed_kph": float(speed_kph),
            "steering": float(steering),
            "throttle": float(throttle),
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }
        try:
            print(f"[FoxgloveBridge] Sending vehicle state: {message}")
            self.vehicle_state_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle state: {e}")
    
    def send_lidar(self, points):
        if points is None or len(points) == 0:
            print("[FoxgloveBridge] No LiDAR points to send.")
            return
        try:
            points_array = np.asarray(points, dtype=np.float32)
            # Limit to first 1000 points for testing
            if len(points_array) > 1000:
                points_array = points_array[:1000]
            points_data = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in points_array
            ]
            message = {
                "points": points_data,
                "count": len(points_data)
            }
            print(f"[FoxgloveBridge] Sending LiDAR: {len(points_data)} points")
            self.lidar_channel.log(message)
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending LiDAR: {e}")
    