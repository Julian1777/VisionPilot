"""
Foxglove Bridge for BeamNG Simulation
Handles all communication with Foxglove Studio via WebSocket
"""

import json
import time
import struct
import numpy as np
import cv2
from pathlib import Path
import foxglove
from foxglove import start_server, Channel
from foxglove.channels import (
    PosesInFrameChannel,
    SceneUpdateChannel,
    PointCloudChannel,
    FrameTransformsChannel,
    CompressedImageChannel,
    LinePrimitiveChannel,
)
from foxglove.schemas import (
    Timestamp,
    PointCloud,
    PackedElementField,
    PackedElementFieldNumericType,
    PosesInFrame,
    Pose,
    Quaternion,
    Vector3,
    SceneUpdate,
    SceneEntity,
    ModelPrimitive,
    CubePrimitive,
    CompressedImage,
    Color,
    FrameTransform,
    FrameTransforms,
    LinePrimitive,
    LinePrimitiveLineType,
)

class FoxgloveBridge:
    """Bridge class for sending data to Foxglove Studio"""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.channels = {}
        self._vehicle_3d_sent = False
        self._urdf_path = self._get_urdf_path()
        
    def start_server(self):
        """Start the Foxglove WebSocket server in a background thread"""
        try:
            self.server = start_server(
                name="BeamNG Simulation",
                host=self.host,
                port=self.port
            )
            print(f"Foxglove server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Error starting Foxglove server: {e}")
            raise
    
    def initialize_channels(self):
        """Initialize all channels for different data types"""
        # Lane detection channel (JSON)
        self.channels['lane'] = Channel(
            topic="/lane_detection",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "lane_center": {"type": "number"},
                    "vehicle_center": {"type": "number"},
                    "deviation": {"type": "number"},
                    "confidence": {"type": "number"},
                    "left_lane_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"}
                            }
                        }
                    },
                    "right_lane_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"}
                            }
                        }
                    }
                }
            }
        )
        self.lane_channel = self.channels['lane']
        
        # Vehicle control channel (JSON)
        self.channels['vehicle_control'] = Channel(
            topic="/vehicle_control",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "speed_kph": {"type": "number"},
                    "steering": {"type": "number"},
                    "throttle": {"type": "number"},
                    "brake": {"type": "number"}
                }
            }
        )
        
        # Vehicle pose channel (PosesInFrame)
        self.channels['vehicle_pose'] = PosesInFrameChannel(topic="/vehicle_pose")
        
        # TF tree channel (FrameTransforms)
        self.channels['tf'] = FrameTransformsChannel(topic="/tf")
        
        # Scene update channel (for 3D model)
        self.channels['scene'] = SceneUpdateChannel(topic="/scene")
        
        # LiDAR point cloud channel (PointCloud)
        self.channels['lidar'] = PointCloudChannel(topic="/lidar")
        
        # Camera image channel (CompressedImage)
        self.channels['camera'] = CompressedImageChannel(topic="/camera/image/compressed")
        
        # Lane path channel (LinePrimitive)
        self.channels['lane_path'] = LinePrimitiveChannel(topic="/lane_path")
        
        # Vehicle detection channel (JSON)
        self.channels['vehicle_detection'] = Channel(
            topic="/vehicle_detections",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "type": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "confidence": {"type": "number"}
                }
            }
        )
        self.vehicle_channel = self.channels['vehicle_detection']
        
        # Sign detection channel (JSON)
        self.channels['sign_detection'] = Channel(
            topic="/sign_detections",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "type": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "confidence": {"type": "number"}
                }
            }
        )
        self.sign_channel = self.channels['sign_detection']

        # Traffic Light detection channel (JSON)
        self.channels['traffic_light_detection'] = Channel(
            topic="/traffic_light_detections",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "type": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "confidence": {"type": "number"}
                }
            }
        )
        self.traffic_light_channel = self.channels['traffic_light_detection']
        
        print("All Foxglove channels initialized")
    
    def _timestamp_to_time(self, timestamp_ns):
        """Convert nanoseconds timestamp to Timestamp message"""
        sec = timestamp_ns // 1_000_000_000
        nsec = timestamp_ns % 1_000_000_000
        return Timestamp(sec=sec, nsec=nsec)
    
    def send_vehicle_control(self, timestamp_ns, speed_kph, steering, throttle, brake):
        """Send vehicle control state"""
        message = {
            "timestamp": timestamp_ns,
            "speed_kph": float(speed_kph),
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake)
        }
        self.channels['vehicle_control'].log(message)
    
    def send_vehicle_pose(self, timestamp_ns, x, y, z, quat_x, quat_y, quat_z, quat_w, frame_id="map"):
        """Send vehicle pose as PosesInFrame in the map frame"""
        timestamp = self._timestamp_to_time(timestamp_ns)
        
        # Send the pose of the vehicle (base_link) in the map frame
        # This visualizes where base_link is positioned in the world
        pose = PosesInFrame(
            timestamp=timestamp,
            frame_id=frame_id,
            poses=[
                Pose(
                    position=Vector3(x=float(x), y=float(y), z=float(z)),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                )
            ]
        )
        
        self.channels['vehicle_pose'].log(pose)
    
    def send_tf_tree(self, timestamp_ns, x, y, z, quat_x, quat_y, quat_z, quat_w):
        """Send TF tree with map -> base_link -> lidar_top and map -> base_link -> camera_front transforms"""
        timestamp = self._timestamp_to_time(timestamp_ns)
        
        # Build transforms for complete hierarchy:
        # map (root/world origin) - base_link (vehicle body at position)
        # lidar_top (LiDAR sensor mount)
        # camera_front (front camera mount)
        transforms = [
            # Vehicle position in world (map - base_link establishes where vehicle is)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="map",
                child_frame_id="base_link",
                translation=Vector3(x=float(x), y=float(y), z=float(z)),
                rotation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
            ),
            # LiDAR sensor mount (base_link - lidar_top is fixed offset)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="base_link",
                child_frame_id="lidar_top",
                translation=Vector3(x=0.0, y=-0.35, z=1.425),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            # Front camera mount (base_link - camera_front is fixed offset)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="base_link",
                child_frame_id="camera_front",
                translation=Vector3(x=0.0, y=-1.3, z=1.4),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        ]
        
        # Send as FrameTransforms message
        tf_message = FrameTransforms(transforms=transforms)
        self.channels['tf'].log(tf_message)
    
    def _get_urdf_path(self):
        """Get URDF file path as file:// URL"""
        bridge_dir = Path(__file__).parent
        urdf_path = bridge_dir / "model" / "bmw_x5" / "bmw_x5.urdf"
        if urdf_path.exists():
            file_url = urdf_path.as_uri()
            print(f"URDF file available at: {file_url}")
            return file_url
        else:
            print(f"Warning: URDF file not found at {urdf_path}")
            return None
    
    def send_vehicle_3d(self, timestamp_ns, x, y, z, quat_x, quat_y, quat_z, quat_w, frame_id="map"):
        """Send 3D vehicle model using SceneUpdate
        
        Tries multiple approaches:
        1. URDF file via file:// URL (can be added to Foxglove 3D panel as custom layer)
        2. GLB meshes embedded in ModelPrimitives
        3. Fallback to simple cube
        """
        if self._vehicle_3d_sent:
            # Only send once
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        entities = []
        
        if self._urdf_path:
            print(f"URDF model available at: {self._urdf_path}")
            print("To view in Foxglove: Open the 3D panel → Add layer → select URDF layer → paste URL above")
        
        # Approach 2: Send as GLB meshes (embedded in scene)
        try:
            with open(r"c:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\foxglove_integration\model\bmw_x5\meshes\car_body.glb", "rb") as f:
                body_data = f.read()
            
            body_model = ModelPrimitive(
                pose=Pose(
                    position=Vector3(x=float(x), y=float(y), z=float(z)),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                data=body_data,
                media_type="model/gltf-binary"
            )
            
            body_entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="vehicle_body",
                models=[body_model]
            )
            entities.append(body_entity)
            print("Vehicle body mesh loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load car body mesh: {e}")
        
        # Front left wheel
        try:
            with open(r"c:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\foxglove_integration\model\bmw_x5\meshes\wheel_front_left.glb", "rb") as f:
                wheel_fl_data = f.read()
            
            wheel_fl_model = ModelPrimitive(
                pose=Pose(
                    position=Vector3(x=float(x) + 1.3, y=float(y) + 0.8, z=float(z) - 0.3),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                data=wheel_fl_data,
                media_type="model/gltf-binary"
            )
            
            wheel_fl_entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="wheel_front_left",
                models=[wheel_fl_model]
            )
            entities.append(wheel_fl_entity)
        except Exception as e:
            print(f"Warning: Could not load front left wheel mesh: {e}")
        
        # Front right wheel
        try:
            with open(r"c:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\foxglove_integration\model\bmw_x5\meshes\wheel_front_right.glb", "rb") as f:
                wheel_fr_data = f.read()
            
            wheel_fr_model = ModelPrimitive(
                pose=Pose(
                    position=Vector3(x=float(x) + 1.3, y=float(y) - 0.8, z=float(z) - 0.3),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                data=wheel_fr_data,
                media_type="model/gltf-binary"
            )
            
            wheel_fr_entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="wheel_front_right",
                models=[wheel_fr_model]
            )
            entities.append(wheel_fr_entity)
        except Exception as e:
            print(f"Warning: Could not load front right wheel mesh: {e}")
        
        # Rear left wheel
        try:
            with open(r"c:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\foxglove_integration\model\bmw_x5\meshes\wheel_rear_left.glb", "rb") as f:
                wheel_rl_data = f.read()
            
            wheel_rl_model = ModelPrimitive(
                pose=Pose(
                    position=Vector3(x=float(x) - 1.3, y=float(y) + 0.8, z=float(z) - 0.3),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                data=wheel_rl_data,
                media_type="model/gltf-binary"
            )
            
            wheel_rl_entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="wheel_rear_left",
                models=[wheel_rl_model]
            )
            entities.append(wheel_rl_entity)
        except Exception as e:
            print(f"Warning: Could not load rear left wheel mesh: {e}")
        
        # Rear right wheel
        try:
            with open(r"c:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\foxglove_integration\model\bmw_x5\meshes\wheel_rear_right.glb", "rb") as f:
                wheel_rr_data = f.read()
            
            wheel_rr_model = ModelPrimitive(
                pose=Pose(
                    position=Vector3(x=float(x) - 1.3, y=float(y) - 0.8, z=float(z) - 0.3),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                data=wheel_rr_data,
                media_type="model/gltf-binary"
            )
            
            wheel_rr_entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="wheel_rear_right",
                models=[wheel_rr_model]
            )
            entities.append(wheel_rr_entity)
        except Exception as e:
            print(f"Warning: Could not load rear right wheel mesh: {e}")
        
        if not entities:
            print("No meshes loaded, using cube fallback")
            cube = CubePrimitive(
                pose=Pose(
                    position=Vector3(x=float(x), y=float(y), z=float(z)),
                    orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                ),
                size=Vector3(x=2.0, y=1.0, z=1.0),
                color=Color(r=0.1, g=0.1, b=0.8, a=1.0)
            )
            entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id="vehicle_model",
                cubes=[cube]
            )
            entities.append(entity)
        
        if entities:
            scene_update = SceneUpdate(entities=entities)
            self.channels['scene'].log(scene_update)
            self._vehicle_3d_sent = True
    
    def send_lidar(self, points, timestamp_ns, frame_id="lidar_top"):
        """
        Send LiDAR point cloud
        Args:
            points: numpy array of shape (N, 3) or (N, 4) with x, y, z, [intensity]
            timestamp_ns: timestamp in nanoseconds
            frame_id: frame ID for the point cloud (should be "lidar_top" for sensor-relative, or "map" for world-relative)
        """
        if points is None or len(points) == 0:
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if points.shape[1] == 3:
            # x, y, z only
            num_points = points.shape[0]
            point_stride = 12
            fields = [
                PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
                PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
                PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32)
            ]
            data = points.astype(np.float32).tobytes()
        elif points.shape[1] == 4:
            num_points = points.shape[0]
            point_stride = 16
            fields = [
                PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
                PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
                PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
                PackedElementField(name="intensity", offset=12, type=PackedElementFieldNumericType.Float32)
            ]
            data = points.astype(np.float32).tobytes()
        else:
            raise ValueError(f"Points must have shape (N, 3) or (N, 4), got {points.shape}")
        
        point_cloud = PointCloud(
            timestamp=timestamp,
            frame_id=frame_id,
            pose=Pose(
                position=Vector3(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            point_stride=point_stride,
            fields=fields,
            data=data
        )
        
        self.channels['lidar'].log(point_cloud)
    
    def send_camera_image(self, image, timestamp_ns, frame_id="camera"):
        """
        Send camera image as CompressedImage
        Args:
            image: numpy array (BGR format from OpenCV)
            timestamp_ns: timestamp in nanoseconds
            frame_id: frame ID for the image
        """
        if image is None:
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        result, encoded_image = cv2.imencode('.jpg', image_rgb, encode_param)
        
        if not result:
            print("Error encoding image")
            return
        
        compressed_image = CompressedImage(
            timestamp=timestamp,
            frame_id=frame_id,
            data=encoded_image.tobytes(),
            format="jpeg"
        )
        
        self.channels['camera'].log(compressed_image)

    def send_lane_path(self, lane_points, timestamp_ns, lane_id="lane_path", color=None, thickness=0.1, frame_id="map"):
        """
        Send lane path as LinePrimitive on dedicated /lane_path channel
        Args:
            lane_points: numpy array of shape (N, 3) with x, y, z coordinates
            timestamp_ns: timestamp in nanoseconds
            lane_id: unique identifier for this lane
            color: Color object (default: yellow)
            thickness: line thickness in meters
            frame_id: frame ID (default: "map")
        """
        if lane_points is None or len(lane_points) < 2:
            return
        
        if color is None:
            color = Color(r=1.0, g=1.0, b=0.0, a=1.0)
        
        points = [
            Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            for p in lane_points
        ]
        
        line_primitive = LinePrimitive(
            type=LinePrimitiveLineType.LINE_STRIP,
            pose=Pose(
                position=Vector3(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            thickness=thickness,
            scale_invariant=False,
            points=points,
            color=color
        )
        
        self.channels['lane_path'].log(line_primitive)

