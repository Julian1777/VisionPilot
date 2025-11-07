import foxglove
from foxglove import Channel
from foxglove.channels import PointCloudChannel
from foxglove.schemas import Timestamp, PointCloud, PackedElementField, PackedElementFieldNumericType
import numpy as np
import time

f32 = PackedElementFieldNumericType.Float32


class FoxgloveBridge:
    def __init__(self):
        self._initialized = False
        self._server_started = False
        # JSON channels
        self.sign_channel = None
        self.traffic_light_channel = None
        self.lane_channel = None
        self.vehicle_channel = None
        self.vehicle_state_channel = None
        # PointCloud channel
        self.lidar_channel = None
    
    def start_server(self):
        """Start the Foxglove WebSocket server"""
        if self._server_started:
            return
        
        print("[FoxgloveBridge] Starting Foxglove WebSocket server...")
        foxglove.set_log_level("INFO")
        foxglove.start_server()
        self._server_started = True
        print("[FoxgloveBridge] Foxglove server started on ws://localhost:8765")
    
    def initialize_channels(self):
        """Initialize channels after server has started"""
        if self._initialized:
            return
        
        if not self._server_started:
            self.start_server()
        
        print("[FoxgloveBridge] Initializing channels...")
        
        # JSON channels
        self.sign_channel = Channel("/detections/sign", message_encoding="json")
        self.traffic_light_channel = Channel("/detections/traffic_light", message_encoding="json")
        self.lane_channel = Channel("/detections/lane", message_encoding="json")
        self.vehicle_channel = Channel("/detections/vehicle", message_encoding="json")
        self.vehicle_state_channel = Channel("/vehicle/state", message_encoding="json")
        
        # PointCloud channel for LiDAR visualization
        self.lidar_channel = PointCloudChannel(topic="/lidar/points")
        
        self._initialized = True
        print("[FoxgloveBridge] All channels initialized successfully")
        print("  - /detections/sign")
        print("  - /detections/traffic_light")
        print("  - /detections/lane")
        print("  - /detections/vehicle")
        print("  - /vehicle/state")
        print("  - /lidar/points (PointCloud)")
    
    def send_sign_detection(self, sign_type, x, y, confidence):
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping sign detection")
            return
        
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
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping lane detection")
            return
        
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
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping vehicle detection")
            return
        
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
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping vehicle state")
            return
        
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
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping LiDAR")
            return
        
        if points is None or len(points) == 0:
            print("[FoxgloveBridge] No LiDAR points to send.")
            return
        try:
            points_array = np.asarray(points, dtype=np.float32)
            # Limit to avoid overwhelming the connection
            if len(points_array) > 10000:
                points_array = points_array[:10000]
            
            # Create PointCloud message
            # 3 floats per point (x, y, z)
            data_bytes = points_array.tobytes()
            
            fields = [
                PackedElementField(name="x", offset=0, type=f32),
                PackedElementField(name="y", offset=4, type=f32),
                PackedElementField(name="z", offset=8, type=f32),
            ]
            
            stamp = int(time.time() * 1e9)
            timestamp = Timestamp(sec=int(stamp // 1e9), nsec=int(stamp % 1e9))
            
            pc = PointCloud(
                timestamp=timestamp,
                frame_id="base_link",
                point_stride=12,  # 3 * 4 bytes
                fields=fields,
                data=data_bytes,
            )
            
            self.lidar_channel.log(pc, log_time=stamp)
            print(f"[FoxgloveBridge] LiDAR sent: {len(points_array)} points")
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending LiDAR: {e}")
    