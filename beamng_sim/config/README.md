# BeamNG Simulation Configuration

This folder contains YAML configuration files that control all aspects of the BeamNG simulation, including simulation settings, scenarios, vehicles, sensors, and control parameters.

## Configuration Files

### `beamng_sim.yaml`
**Main simulation and vehicle configuration**

#### `simulation` section:
- `host`: BeamNG server IP address (default: `localhost`)
- `port`: BeamNG server port (default: `64256`)
- `home`: Path to BeamNG.tech installation directory
- `deterministic_fps`: Frame rate for deterministic simulation (default: `60`)

#### `vehicles` section:
Defines modular vehicle configurations. Each vehicle entry includes:
- `name`: Internal name for the vehicle in simulation
- `model`: BeamNG vehicle model (e.g., `etk800`, `rsq8_600_tfsi`)
- `license`: Vehicle license plate string
- `spawn_pos`: Initial [x, y, z] coordinates for vehicle spawn
- `spawn_yaw`: Initial heading angle in degrees (yaw rotation)

**Example vehicles:**
- `etk800_highway`: ETK800 vehicle configured for highway scenario
- `etk800_city`: ETK800 vehicle configured for city scenario
- `rsq8_default_highway`: Audi Q8 for highway testing

---

### `scenarios.yaml`
**Scenario definitions that link maps, scenes, and vehicles**

Each scenario entry includes:
- `map`: BeamNG map name (e.g., `west_coast_usa`)
- `scene`: Specific scene within the map (e.g., `SLAM_highway`, `SLAM_city`)
- `vehicle`: Reference to a vehicle config from `beamng_sim.yaml`

**Example scenarios:**
- `highway`: Highway testing scenario with ETK800
- `city`: City testing scenario with ETK800

---

### `sensors.yaml`
**Sensor configuration for camera, LiDAR, and radar**

#### `camera` section:
- `name`: Sensor identifier in BeamNG
- `enabled`: Whether to initialize the camera (true/false)
- `requested_update_time`: Update interval in seconds
- `is_using_shared_memory`: Use shared memory for faster data transfer (true/false)
- `pos`: Camera position relative to vehicle [x, y, z]
- `dir`: Camera direction vector [x, y, z]
- `field_of_view_y`: Vertical field of view in degrees
- `near_far_planes`: [near_clip, far_clip] for rendering depth
- `resolution`: [width, height] in pixels
- `is_streaming`: Enable continuous streaming (true/false)
- `is_render_colours`: Render RGB color image (true/false)

#### `lidar` section:
- `name`: LiDAR sensor identifier
- `enabled`: Whether to initialize LiDAR
- `requested_update_time`: Update interval in seconds
- `is_using_shared_memory`: Shared memory optimization
- `is_rotate_mode`: Rotating scan mode (true/false)
- `horizontal_angle`: Horizontal field of view in degrees
- `vertical_angle`: Vertical field of view in degrees
- `vertical_resolution`: Number of vertical scan lines/channels
- `density`: Point density (higher = more points)
- `frequency`: Update frequency in Hz
- `max_distance`: Maximum detection range in meters
- `pos`: LiDAR position relative to vehicle [x, y, z]
- `is_visualised`: Show point cloud visualization (true/false)

#### `radar` section:
- `name`: Radar sensor identifier
- `enabled`: Whether to initialize radar
- `requested_update_time`: Update interval
- `pos`: Radar position [x, y, z]
- `dir`: Radar direction [x, y, z]
- `up`: Radar up vector [x, y, z]
- `size`: Radar aperture size [width, height]
- `near_far_planes`: [min_range, max_range] in meters
- `field_of_view_y`: Vertical field of view in degrees

---

### `control.yaml`
**Vehicle control and PID tuning parameters**

#### `ai` section:
- `enabled`: Enable AI driving assistance
- `drive_in_lane`: Constrain vehicle to lane (true/false)
- `mode`: AI driving mode (e.g., `span`)

#### `control` section:

**`steering_pid`** - PID controller for steering angle:
- `Kp`: Proportional gain (0.025)
- `Ki`: Integral gain (0.0)
- `Kd`: Derivative gain (0.02)
- `derivative_filter_alpha`: Low-pass filter for derivative (0.2)

**Speed and throttle control:**
- `max_steering_change`: Maximum steering change per frame (0.22)
- `base_throttle`: Base throttle value (0.12)
- `target_speed_kph`: Target speed in km/h (50)

**`speed_pid`** - PID controller for cruise control:
- `Kp`: Proportional gain (0.1)
- `Ki`: Integral gain (0.01)
- `Kd`: Derivative gain (0.05)

---

## Usage Example

```python
from beamng_sim.beamng import load_config, sim_setup

# Load all configurations
beamng_config, scenarios_config, sensors_config = load_config()

# Setup simulation with highway scenario
beamng, scenario, vehicle, camera, lidar, radar = sim_setup(scenario_name='highway')

# Or switch to city scenario with different vehicle
beamng, scenario, vehicle, camera, lidar, radar = sim_setup(scenario_name='city')
```

## Quick Tips

- **Modify vehicle positions**: Edit `spawn_pos` and `spawn_yaw` in `beamng_sim.yaml`
- **Change scenarios**: Add new entries to `scenarios.yaml` and `beamng_sim.yaml`
- **Adjust sensor quality**: Modify resolution, density, and frequency in `sensors.yaml`
- **Tune vehicle behavior**: Adjust PID gains in `control.yaml` for better steering/speed control
- **Disable sensors**: Set `enabled: false` for any sensor to skip initialization
