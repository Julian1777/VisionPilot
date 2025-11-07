import foxglove
from foxglove import Channel
from foxglove.channels import SceneUpdateChannel, PointCloudChannel
from foxglove.schemas import (
    Color,
    CubePrimitive,
    SceneEntity,
    SceneUpdate,
    Vector3,
    ModelPrimitive,
    Pose,
    Timestamp,
    PointCloud,
    PackedElementField,
    PackedElementFieldNumericType,
    Quaternion,
    FrameTransform,
)

import numpy as np
import os
import time



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
        # PointCloud channel (native Foxglove schema)
        self.lidar_channel = None
        self.scene_channel = None
        # Transform channel for base_link → world mapping
        self.transforms_channel = None
    
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
        
        print("[FoxgloveBridge] Initializing channels...")
        
        # JSON channels
        self.sign_channel = Channel("/detections/sign", message_encoding="json")
        self.traffic_light_channel = Channel("/detections/traffic_light", message_encoding="json")
        self.lane_channel = Channel("/detections/lane", message_encoding="json")
        self.vehicle_channel = Channel("/detections/vehicle", message_encoding="json")
        self.vehicle_state_channel = Channel("/vehicle/state", message_encoding="json")
        
        # PointCloud channel (native Foxglove schema)
        self.lidar_channel = PointCloudChannel(topic="/lidar/pointcloud")
        
        # Scene channel for 3D models
        self.scene_channel = SceneUpdateChannel("/scene")
        
        # Transforms channel for frame mapping
        self.transforms_channel = Channel("/tf", message_encoding="json")
        
        self._initialized = True
        print("[FoxgloveBridge] Channels initialized successfully")
    
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
    
    def publish_vehicle_transform(self, x, y, z, yaw=0.0):
        """
        Publish transform from world → base_link to establish frame hierarchy in Foxglove.
        This allows base_link-relative data (like LiDAR) to follow the vehicle.
        
        Args:
            x, y, z: Vehicle position in world coordinates
            yaw: Vehicle yaw angle in radians
        """
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping transform")
            return
        
        try:
            # Convert yaw to quaternion
            half_yaw = yaw / 2.0
            qx = 0.0
            qy = 0.0
            qz = np.sin(half_yaw)
            qw = np.cos(half_yaw)
            
            # Create FrameTransform message
            tf_msg = {
                "timestamp": int(time.time() * 1e9),
                "parent_frame_id": "world",
                "child_frame_id": "base_link",
                "translation": {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                },
                "rotation": {
                    "x": float(qx),
                    "y": float(qy),
                    "z": float(qz),
                    "w": float(qw)
                }
            }
            
            self.transforms_channel.log(tf_msg)
            print(f"[FoxgloveBridge] Published transform: world → base_link at ({x:.1f}, {y:.1f}, {z:.1f})")
            
        except Exception as e:
            print(f"[FoxgloveBridge] Error publishing transform: {e}")
    
    def send_lidar(self, points):
        """
        Send LiDAR point cloud data using native Foxglove PointCloud schema.
        Uses world frame to match vehicle position.
        Args:
            points: numpy array or list of points with shape (N, 3) containing x, y, z coordinates
        """
        if not self._initialized:
            print("[FoxgloveBridge] Warning: Bridge not initialized, skipping LiDAR")
            return
        
        if points is None or len(points) == 0:
            print("[FoxgloveBridge] No LiDAR points to send.")
            return
        
        try:
            # Convert to float32 numpy array
            points_array = np.asarray(points, dtype=np.float32)
            
            # Ensure shape is (N, 3)
            if len(points_array.shape) == 1:
                points_array = points_array.reshape(-1, 3)
            
            # Define PointCloud fields (x, y, z)
            f32 = PackedElementFieldNumericType.Float32
            fields = [
                PackedElementField(name="x", offset=0, type=f32),
                PackedElementField(name="y", offset=4, type=f32),
                PackedElementField(name="z", offset=8, type=f32),
            ]
            
            # Flatten to bytes
            data_bytes = points_array.tobytes()
            
            # Create timestamp
            stamp = int(time.time() * 1e9)
            timestamp = Timestamp(sec=int(stamp // 1e9), nsec=int(stamp % 1e9))
            
            # Create PointCloud message using world frame instead of base_link
            pc = PointCloud(
                timestamp=timestamp,
                frame_id="world",  # Changed from base_link to world
                point_stride=12,  # 3 fields * 4 bytes each
                fields=fields,
                data=data_bytes,
            )
            
            # Send to Foxglove
            self.lidar_channel.log(pc, log_time=stamp)
            print(f"[FoxgloveBridge] Sent {len(points_array)} LiDAR points to /lidar/pointcloud (world frame)")
            
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending LiDAR: {e}")
    
    def send_vehicle_3d(self, x, y, z, yaw=0.0):
        """
        Send vehicle position and orientation with BMW X5 GLB models.
        Uses Quaternion helper for orientation.
        """
        try:
            # Convert yaw to quaternion (rotation around Z axis)
            half_yaw = yaw / 2.0
            qx = 0.0
            qy = 0.0
            qz = np.sin(half_yaw)
            qw = np.cos(half_yaw)

            entities = []

            # Base directory for GLB files
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "model", "bmw_x5", "meshes"
            )

            # Car body
            body_glb_path = os.path.join(model_dir, "car_body.glb")
            if os.path.exists(body_glb_path):
                body_entity = SceneEntity(
                    id="bmw_body",
                    frame_id="world",
                    model=ModelPrimitive(
                        url=f"file://{body_glb_path}",
                        pose=Pose(
                            position=Vector3(x=float(x), y=float(y), z=float(z)),
                            orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
                        ),
                        scale=Vector3(x=1.0, y=1.0, z=1.0),
                    ),
                )
                entities.append(body_entity)

            # Wheel positions from URDF (relative to body)
            wheels = [
                ("front_left_wheel", 1.3, 0.8, -0.3, "wheel_front_left.glb"),
                ("front_right_wheel", 1.3, -0.8, -0.3, "wheel_front_right.glb"),
                ("rear_left_wheel", -1.3, 0.8, -0.3, "wheel_rear_left.glb"),
                ("rear_right_wheel", -1.3, -0.8, -0.3, "wheel_rear_right.glb"),
            ]

            for wheel_id, rel_x, rel_y, rel_z, mesh_file in wheels:
                glb_path = os.path.join(model_dir, mesh_file)
                if os.path.exists(glb_path):
                    wheel_entity = SceneEntity(
                        id=wheel_id,
                        frame_id="world",
                        model=ModelPrimitive(
                            url=f"file://{glb_path}",
                            pose=Pose(
                                position=Vector3(x=float(x + rel_x), y=float(y + rel_y), z=float(z + rel_z)),
                                orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
                            ),
                            scale=Vector3(x=1.0, y=1.0, z=1.0),
                        ),
                    )
                    entities.append(wheel_entity)

            if entities:
                scene_update = SceneUpdate(entities=entities)
                print(f"[FoxgloveBridge] Sending BMW X5 3D model at ({x:.1f}, {y:.1f}, {z:.1f}), yaw={np.degrees(yaw):.1f}°")
                self.scene_channel.publish(scene_update)
            else:
                print(f"[FoxgloveBridge] Warning: No GLB files found in {model_dir}")
        except Exception as e:
            print(f"[FoxgloveBridge] Error sending vehicle 3D: {e}")
    