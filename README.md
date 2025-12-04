<p align="center">
  <img src="images/bannernobg.png" alt="VisionPilot Banner" height="200" />
</p>

# VisionPilot: Autonomous Driving Simulation, Computer Vision & Real-Time Perception (BeamNG.tech)

<p align="center" style="margin-bottom:0;">
  <a href="https://star-history.com/#visionpilot-project/VisionPilot&Date">
    <img src="https://api.star-history.com/svg?repos=visionpilot-project/VisionPilot&type=Date" alt="Star History Chart" height="300" />
  </a>
</p>

## Table of Contents
- [VisionPilot: Autonomous Driving Simulation, Computer Vision \& Real-Time Perception (BeamNG.tech)](#visionpilot-autonomous-driving-simulation-computer-vision--real-time-perception-beamngtech)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Demos](#demos)
    - [Pure Sign Detection \& Detection and classification of vehicles and pedestrians Demo](#pure-sign-detection--detection-and-classification-of-vehicles-and-pedestrians-demo)
  - [Traffic Light Detection \& Classification Demo](#traffic-light-detection--classification-demo)
    - [Latest Lane Detection Demo (v2)](#latest-lane-detection-demo-v2)
      - [Previous Lane Detection Demo (v1)](#previous-lane-detection-demo-v1)
  - [Foxglove Visualization Demo](#foxglove-visualization-demo)
  - [Features](#features)
  - [Built With](#built-with)
  - [Datasets Used](#datasets-used)
  - [Quickstart \& Usage](#quickstart--usage)
  - [Setup \& Installation](#setup--installation)
  - [Model Details](#model-details)
  - [Configuration Files](#configuration-files)
  - [Roadmap](#roadmap)
    - [Perception](#perception)
    - [Sensor Fusion \& Calibration](#sensor-fusion--calibration)
    - [Control \& Planning](#control--planning)
    - [Simulation \& Scenarios](#simulation--scenarios)
    - [Visualization \& Logging](#visualization--logging)
    - [README To-Dos](#readme-to-dos)
  - [Legend](#legend)
  - [Engineering Journal](#engineering-journal)
  - [Credits](#credits)
    - [BeamNG.tech Citation](#beamngtech-citation)

## Overview

A modular Python project for autonomous driving research and prototyping, fully integrated with the BeamNG.tech simulator and Foxglove visualization. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) with real-time sensor fusion and autonomous vehicle control to tackle:

- Lane detection (Traditional CV, SCNN, capable of highway scenarios)
- Traffic sign classification & detection (CNN, YOLOv8)
- Traffic light detection & classification (YOLOv8, CV, CNN)
- Vehicle & pedestrian detection and recognition (YOLOv8)
- Multi-sensor fusion (Camera, LiDAR, Radar)
- Multi-model inference, real-time simulation, autonomous driving with PID control (BeamNG.tech)
- Real-time visualization and monitoring (Foxglove WebSocket)

## Demos

### Pure Sign Detection & Detection and classification of vehicles and pedestrians Demo

This demo shows real-time traffic sign detection (no classification) as well as vehicle and pedestrian on a video:

![Sign Detection Demo & Vehicle Pedestrian](images/traffic-sign-classification/sign_detection_demo.gif)

> Only detection is shown; classification is not included in this demo.

## Traffic Light Detection & Classification Demo

This demo shows real-time traffic light detection and classification on a video:

![Traffic Light Detection & Classification Demo](images/traffic-lights/traffic_light_demo.gif)

> Both detection and classification are shown in this demo.

### Latest Lane Detection Demo (v2)

Watch the improved autonomous lane keeping demo (v2) in BeamNG.tech, featuring smoother fused CV+SCNN lane detection, stable PID steering, and robust adaptive cruise control:

[![Autonomous Lane Keeping v2 Demo](https://img.youtube.com/vi/7eA_XfIkLWQ/0.jpg)](https://www.youtube.com/watch?v=7eA_XfIkLWQ)

Watch on YouTube: [https://www.youtube.com/watch?v=7eA_XfIkLWQ](https://www.youtube.com/watch?v=7eA_XfIkLWQ)

This v2 demo shows:
- Autonomous lane keeping with tuned PID steering (stable, minimal oscillation)
- Fused CV & SCNN lane detection (smoother, more robust)
- Adaptive cruise control (ACC) holding ~50 km/h
- Handles shadows, bridges, and recovers from distractions
- Real-time visualization and multi-model inference

> Note: Very low-light (tunnel) scenarios are not yet supported.

---

#### Previous Lane Detection Demo (v1)

The original demo is still available for reference:

[Lane Keeping & Multi-Model Detection Demo (v1)](https://youtu.be/f9mHigMKME8)


## Foxglove Visualization Demo

See real-time LiDAR point cloud streaming and autonomous vehicle telemetry in Foxglove Studio:

[![Foxglove LiDAR Visualization Demo](https://img.youtube.com/vi/4HJDvL2Q6AY/0.jpg)](https://www.youtube.com/watch?v=4HJDvL2Q6AY)

Watch on YouTube: [https://www.youtube.com/watch?v=4HJDvL2Q6AY](https://www.youtube.com/watch?v=4HJDvL2Q6AY)

This demo shows:
- Real-time LiDAR point cloud visualization
- Foxglove Studio WebSocket integration
- Autonomous vehicle simulation with BeamNG.tech
- Modular Python pipeline for sensor fusion and control

> More demo videos and visualizations will be added as features are completed.

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
- **TUSimple** for SCNN lane detection

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
- [x] Sign classification & Detection (CNN / YOLOv11)
- [x] Traffic light classification & Detection (CNN / YOLOv11m)
- [x] Lane detection Fusion (SCNN / CV)
- [x] Advanced lane detection using OpenCV (robust highway, lighting, outlier handling)
- [x] Integrate Majority Voting system for CV
- [ ] Blind Spot Monitoring
- [ ] Real-Time Object Tracking
- [ ] Handle dashed lines better in lane detection
- [ ] Stop Sign Yield Sign Detection and Response (Will be implemented after improving sign classification accuracy; currently only warning is possible)
- [ ] Detect multiple lanes
- [ ] Lane Change Logic
- [ ] 💤 Multi Camera Setup (Will implement after all other camera-based features are finished)
- [ ] 💤 Overtaking, Merging (Will be part of Path Planning)

### Sensor Fusion & Calibration
- [x] ⭐ Integrate Radar
- [ ] Multi radar integration (1 front, 2 rear on each side)
- [x] Integrate Lidar
- [ ] Ultrasonic Sensor Integration
- [ ] 🔥 Lidar Object Detection
- [ ] Map Matching algorithm
- [ ] 💤 💤 SLAM (simultaneous localization and mapping)
- [ ] 🔥 GPS/IMU sensor

### Control & Planning
- [x] Integrate vehicle control (Throttle, Steering, Braking Implemented) (PID needs further tuning)
- [x] Integrate PIDF controller
- [x] ⭐ Adaptive Cruise Control (Currently only basic Cruise Control implemented)
- [ ] 🔥 Emergency Braking / Collision Avoidance
- [ ] Blindspot Monitoring (Can easily be implemented with prebuilt Beamng ADAS module)
- [ ] Path planning
- [ ] 💤 Behaviour planning and anticipation
- [ ] 💤 End-to-end driving policy learning (RL, imitation learning)
- [ ] 💤💤 Advanced traffic participant prediction (trajectory, intent)

### Simulation & Scenarios
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA)
- [x] Modularize and clean up BeamNG.tech pipeline
- [x] Tweak lane detection parameters and thresholds
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Test Lighting conditions
- [ ] 💤 Test using actual RC car
- [ ] 💤💤 Docker containerization

### Visualization & Logging
- [x] ⭐ Full Foxglove visualization integration
- [x] Modular YAML configuration system
- [x] Real-time drive logging and telemetry
- [ ] 🔥 Real time Annotations Overlay in Foxglove

### README To-Dos
- [x] 🔥 Add demo images and videos to README
- [ ] Add performance benchmarks section
- [x] Add Table of Contents for easier navigation

## Legend
> 🔥 = High Priority

> ⭐ = Complete but still being improved/tuned/changed (not final version)

> 💤 = Minimal Priority, can be addressed later

> 💤💤 = Very Low Priority, may not be implement

## Engineering Journal

You can access the periodically updated engineering journal here:

[Engineering Journal (Google Drive)](https://drive.google.com/file/d/15garXT9LaYUK_GlcT68EEbLpoLRc_XoL/view?usp=share_link)

## Credits
- Datasets: CU Lane, LISA, GTRSB, Mapillary, BDD100K
- Models: Ultralytics YOLOv8, custom CNNs
- Simulation: BeamNG.tech ([BeamNG GmbH](https://www.beamng.tech/))
- Special thanks to [Kaggle](https://www.kaggle.com/) for providing free GPU resources for model training without them it would've been imposible to train such good models.
- I would also like to thank my teacher and supervisor Mr. Pratt for their guidance and support throughout this project.

### BeamNG.tech Citation

> **Title:** BeamNG.tech  
> **Author:** BeamNG GmbH  
> **Address:** Bremen, Germany  
> **Year:** 2025  
> **Version:** 0.35.0.0  
> **URL:** https://www.beamng.tech/
