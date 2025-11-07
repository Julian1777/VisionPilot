import foxglove
import time

foxglove.set_log_level("INFO")
foxglove.start_server()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped by user.")