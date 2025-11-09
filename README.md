# Self-Driving Car Simulation: Computer Vision, Deep Learning & Real-Time Perception (BeamNG.tech)

<p align="center">
  <a href="https://star-history.com/#Julian1777/self-driving-project&Date">
    <img src="https://api.star-history.com/svg?repos=Julian1777/self-driving-project&type=Date" alt="Star History Chart" />
  </a>
</p>

A modular Python project for autonomous driving research and prototyping, fully integrated with the BeamNG.tech simulator and Foxglove visualization. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) with real-time sensor fusion and autonomous vehicle control to tackle:

- Lane detection (Traditional CV, SCNN, capable of city & highway scenarios)
- Traffic sign classification & detection (CNN, YOLOv8)
- Traffic light detection & classification (YOLOv8, CV, CNN)
- Vehicle & pedestrian detection and recognition (YOLOv8)
- Multi-sensor fusion (Camera, LiDAR, Radar)
- Multi-model inference, real-time simulation, autonomous driving with PID control (BeamNG.tech)
- Real-time visualization and monitoring (Foxglove WebSocket)

Features robust training pipelines, modular sensor integration, multi-model inference, and a flexible folder structure for easy experimentation and extension. The project is designed for research and prototyping in realistic driving environments using BeamNG.tech with professional-grade visualization through Foxglove.



## Demos

Below are sample demos of the system's capabilities. More demos (including new models and tasks) will be added as development progresses.

| Lane Detection (CV) | Lane Detection (Neural Net) |
|---------------------|----------------------------|
| ![lane-cv](assets/lane_cv.gif) <br> *(coming soon)* | ![lane-nn](assets/lane_nn.gif) <br> *(coming soon)* |

| Sign Detection/Classification | Traffic Light Detection/Classification |
|------------------------------|---------------------------------------|
| ![sign](assets/sign.gif) <br> *(detection & classification)* | ![light](assets/light.gif) <br> *(detection & classification)* |

| Vehicle/Object/Pedestrian Detection | |
|-------------------------------------|--|
| ![vehicle](assets/vehicle.gif) <br> *(coming soon)* | |

> More demo videos and visualizations will be added as features are completed and models are improved.



## Features

- Lane detection with SCNN and traditional OpenCV
- Traffic Sign Classification + Detection
- Traffic Light Classification + Detection
- Vehicle & Pedestrian Detection
- Multi-sensor fusion (Camera, LiDAR, Radar)
- Real-time autonomous driving with PID control
- Cruise control
- Real-time visualization via Foxglove WebSocket
- Modular configuration system (YAML-based)
- Drive logging and telemetry
- Support for multiple scenarios (highway, city)


## Built With

- **Simulation:** BeamNG.tech (https://www.beamng.tech/)
- **Visualization:** Foxglove Studio (WebSocket real-time visualization)
- **Deep Learning:** TensorFlow / Keras, PyTorch
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Language:** Python 3.8+
- **Control Systems:** PID controllers, sensor fusion


## Datasets Used

- **CU Lane Dataset** for lane segmentation
- **DLDT / LISA** for traffic light classification & detection
- **Mapillary** for sign detection
- **BDD** for vehicle and pedestrian detection

## Quickstart & Usage

1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

2. **Configure simulation (Optional):**
  Edit configuration files in `beamng_sim/config/`:
  - `beamng_sim.yaml` - BeamNG host, port, and vehicle settings
  - `scenarios.yaml` - Available scenarios
  - `sensors.yaml` - Sensor parameters (camera, LiDAR, radar)
  - `control.yaml` - PID tuning and vehicle control parameters
  
  See `beamng_sim/config/README.md` for detailed parameter descriptions.

3. **Run the simulation:**
  ```bash
  python -m beamng_sim.beamng
  ```
  - Make sure BeamNG.tech is installed, running, and properly licensed. See [BeamNG.tech documentation](https://www.beamng.tech/) for setup.
  - Foxglove visualization will be available at `ws://localhost:8765`

4. **View real-time visualization:**
  - Open [Foxglove Studio](https://app.foxglove.dev/)
  - Connect to WebSocket server: `ws://localhost:8765`
  - Load the provided Foxglove layout or create your own

  > **Important:** You must ensure that all required models (e.g., trained weights, .h5/.pt files) and configuration files are placed in the correct directories as expected by the code. The folder structure shown below must be followed, and missing files or incorrect paths will cause errors. See each module's README or script comments for details on required files and their locations.

## Setup & Installation
- Python 3.8+
- See `requirements.txt` for all dependencies
- Required: BeamNG.tech simulator for real-time testing ([Download & License](https://www.beamng.tech/))


## Model Details
All models are located in the models folder
- **Lane Detection:** SCNN
- **Traffic Sign Detect/Class:** CNN classifier, YOLOv8 detector
- **Traffic Light Detect/Class:** YOLOv8 detector, CNN classifier
- **Vehicle/Pedestrian:** YOLOv8

## Configuration Files
Configuration files are located in the `beamng_sim/config/` directory:
> Descriptions of the configuration files can be found in the `config/README.md` file.

## Roadmap

### Perception
- [x] Sign classification & Detection (CNN / YOLOv8)
- [x] Traffic light classification & Detection (CNN / YOLOv8)
- [x] Lane detection Fusion (SCNN / CV)
- [x] ⭐ Advanced lane detection using OpenCV (robust city/highway, lighting, outlier handling)
- [x] Integrate Majority Voting system for CV
- [x] Camera Calibration
- [ ] Stop Sign Yield Sign Detection and Response
- [ ] 💤 Lane Change Logic
- [ ] 💤 Overtaking, Merging

### Sensor Fusion & Calibration
- [x] ⭐ Integrate Radar
- [x] Integrate Lidar
- [ ] Sensor Calibration Routines
- [ ] Lidar Object Detection
- [ ] Lidar lane boundary detection
- [ ] 💤 SLAM (simultaneous localization and mapping)
- [ ] 💤 GPS/IMU sensor
- [ ] 💤 Map Matching algorithm

### Control & Planning
- [x] ⭐ Integrate vehicle control (Throttle, Steering, Braking Implemented) (PID needs further tuning)
- [x] ⭐ Adaptive Cruise Control (Currently only basic Cruise Control implemented)
- [ ] Emergency Braking / Collision Avoidance
- [ ] Blindspot Monitoring
- [ ] 💤 Global and Local path planning
- [ ] 💤 Behaviour planning and anticipation
- [ ] 💤 End-to-end driving policy learning (RL, imitation learning)
- [ ] 💤 Advanced traffic participant prediction (trajectory, intent)

### Simulation & Scenarios
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA)
- [x] Modularize and clean up BeamNG.tech pipeline
- [x] Tweak lane detection parameters and thresholds
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Test different weather and lighting conditions
- [ ] 💤 Test using actual RC car
- [ ] 💤 Multi Camera
- [ ] 💤 Docker containerization

### Visualization & Logging
- [x] ⭐ Full Foxglove visualization integration
- [x] Modular YAML configuration system
- [x] Real-time drive logging and telemetry
- [ ] Real time Annotations Overlay in Foxglove

### README To-Dos
- [ ] Add demo images and videos to README
- [ ] Add performance benchmarks section
- [ ] Add Table of Contents for easier navigation
  
> ⭐ = Complete but still being improved/tuned/changed (not final version)

> 💤 = Minimal Priority, can be addressed later

## Credits
- Datasets: CU Lane, LISA, GTRSB, Mapillary, BDD100K
- Models: Ultralytics YOLOv8, custom CNNs
- Simulation: BeamNG.tech ([BeamNG GmbH](https://www.beamng.tech/))
- Special thanks to [Kaggle](https://www.kaggle.com/) for providing free GPU resources for model training without them it would've been imposible to train such good models.

### BeamNG.tech Citation

> **Title:** BeamNG.tech  
> **Author:** BeamNG GmbH  
> **Address:** Bremen, Germany  
> **Year:** 2025  
> **Version:** 0.35.0.0  
> **URL:** https://www.beamng.tech/
