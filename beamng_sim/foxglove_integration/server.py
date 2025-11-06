import foxglove
import time
from .bridge_instance import bridge

foxglove.set_log_level("INFO")
foxglove.start_server()

print("Foxglove WebSocket server running on ws://localhost:8765")
print("Channels registered:")
print("  - /detections/sign")
print("  - /detections/traffic_light")
print("  - /detections/lane")
print("  - /detections/vehicle")
print("  - /vehicle/state")
print("  - /lidar/points")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped.")