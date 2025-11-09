
# 🚗 Self-Driving Car Simulation: Computer Vision, Deep Learning & Real-Time Perception (BeamNG.tech)

<p align="center">
  <a href="https://star-history.com/#Julian1777/self-driving-project&Date">
    <img src="https://api.star-history.com/svg?repos=Julian1777/self-driving-project&type=Date" alt="Star History Chart" />
  </a>
</p>

A modular Python project for autonomous driving research and prototyping, fully integrated with the BeamNG.tech simulator and Foxglove visualization. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) with real-time sensor fusion and autonomous vehicle control to tackle:

- 🛣️ Lane detection (Traditional CV, SCNN, capable of city & highway scenarios)
- 🛑 Traffic sign classification & detection (CNN, YOLOv8)
- 🚦 Traffic light detection & classification (YOLOv8, CV, CNN)
- 🚗 Vehicle & pedestrian detection and recognition (YOLOv8)
- 📡 Multi-sensor fusion (Camera, LiDAR, Radar)
- 🧠 Multi-model inference, real-time simulation, autonomous driving with PID control (BeamNG.tech)
- 📊 Real-time visualization and monitoring (Foxglove WebSocket)

Features robust training pipelines, modular sensor integration, multi-model inference, and a flexible folder structure for easy experimentation and extension. The project is designed for research and prototyping in realistic driving environments using BeamNG.tech with professional-grade visualization through Foxglove.



## 🎥 Demos

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



## 🔧 Features

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


## 🛠️ Built With

- **Simulation:** BeamNG.tech (https://www.beamng.tech/)
- **Visualization:** Foxglove Studio (WebSocket real-time visualization)
- **Deep Learning:** TensorFlow / Keras, PyTorch
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Language:** Python 3.8+
- **Control Systems:** PID controllers, sensor fusion


## 📚 Datasets Used

- **CU Lane Dataset** for lane segmentation
- **DLDT / LISA** for traffic light classification & detection
- **Mapillary** for sign detection
- **BDD** for vehicle and pedestrian detection

## 📊 Results

For qualitative and quantitative results, see the demo section above and the `results/` folder for visualizations, metrics, and sample outputs. Example outputs include:

  - `results/traffic-sign-classification/metrics/` (JSON, curves)
  - `results/traffic-sign-detection/weights/` (YOLO checkpoints)
  - `results/vehicle-pedestrian/visualizations/` (confusion matrices, sample batches)


## ⚡ Quickstart & Usage

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

5. **Train a model:**
  See notebooks or scripts in each module folder.

  > **Note:** You must download and prepare the required datasets yourself (e.g., sorting, cropping, formatting, or converting to the expected structure) as described in each module's documentation or script. The code will not work without properly prepared data.


## 📝 Setup & Installation
- Python 3.8+
- See `requirements.txt` for all dependencies
- Required: BeamNG.tech simulator for real-time testing ([Download & License](https://www.beamng.tech/))


## 🧠 Model Details
All models are located in the models folder
- **Lane Detection:** SCNN
- **Traffic Sign Detect/Class:** CNN classifier, YOLOv8 detector
- **Traffic Light Detect/Class:** YOLOv8 detector, CNN classifier
- **Vehicle/Pedestrian:** YOLOv8

## 📂 Folder Structure

> **Currently Outdated**
<details>
  <summary>Click to expand folder structure</summary>


```
self-driving-project/
├── beamng_sim/                          # BeamNG.tech simulation & real-time perception
│   ├── __init__.py
│   ├── beamng.py                        # Main BeamNG.tech interface/entry point
│   ├── drive_log/                       # Simulation drive logs (CSV)
│   ├── debug_output/
│   │   └── alotofnoise/                 # Debug images for lane detection, perspective
│   │
│   ├── lane_detection/                  # Lane detection algorithms
│   │   ├── __init__.py
│   │   ├── main.py                      # Process frames (CV, UNet, SCNN)
│   │   ├── fusion.py                    # Multi-model fusion logic
│   │   ├── perspective.py               # Bird's eye view transformation
│   │   ├── metrics.py                   # Lane metrics calculation
│   │   ├── color_threshold_debug.py
│   │   ├── lane_finder.py
│   │   ├── thresholding.py
│   │   ├── visualization.py
│   │   ├── old_lane_detection.py
│   │   └── scnn/                        # SCNN model files
│   │       └── scnn_model.py
│   │
│   ├── lidar/                           # LiDAR sensor processing
│   │   ├── __init__.py
│   │   ├── main.py                      # LiDAR frame processing
│   │   ├── lidar.py
│   │   ├── lidar_testing.py
│   │   └── visualization_tool.py
│   │
│   ├── radar/                           # Radar sensor processing
│   │   ├── __init__.py
│   │   ├── main.py                      # Radar frame processing
│   │   └── radar.py
│   │
│   ├── sign/                            # Traffic sign detection & classification
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── detect_classify.py
│   │   └── augmentation.py              # Data augmentation (random_brightness, etc)
│   │
│   ├── vehicle_obstacle/                # Vehicle & pedestrian detection
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── vehicle_obstacle_detection.py
│   │
│   ├── utils/                           # Utility modules
│   │   ├── __init__.py
│   │   └── pid_controller.py            # PID control for steering/speed
│   │
│   └── traffic_lights/                  # Traffic light detection & classification
│       ├── __init__.py
│       ├── main.py
│       └── detection.py
│
├── config/                              # Configuration files
│   ├── __init__.py
│   ├── config.py                        # Global config (paths, models, calibration)
│   ├── README.md                        # 📖 Configuration guide - descriptions of each YAML file
│   ├── beamng_sim.yaml                  # Simulation, vehicles, and scenarios config
│   ├── scenarios.yaml                   # Scenario definitions (highway, city, etc)
│   ├── sensors.yaml                     # Sensor configs (camera, lidar, radar)
│   └── control.yaml                     # Vehicle control & PID tuning parameters
│
├── lane-detection/                      # Traditional CV lane detection (standalone)
│   ├── city/                            # City-specific algorithms
│   └── highway/                         # Highway-specific algorithms
│
├── lane-detection-cnn/                  # CNN/SCNN lane detection training
│   ├── lane_detection.py                # Model training & evaluation
│   ├── dataset/                         # CULane dataset
│   └── results/                         # Training outputs
│
├── traffic_sign/                        # Traffic sign detection & classification
│   ├── detection_kaggle.py              # YOLO training script
│   ├── realtime.py                      # Real-time inference
│   ├── dataset/                         # Mapillary dataset
│   └── results/
│
├── traffic-lights/                      # Traffic light detection & classification
│   ├── detection.py                     # YOLO training script
│   ├── classification.py                # CNN classification training
│   ├── dataset_verification.py          # Dataset validation
│   ├── yolo_test.py                     # Testing & evaluation
│   ├── dtld_dataset/                    # DTLD dataset (German Traffic Lights)
│   ├── lisa_dataset/                    # LISA dataset (US Traffic Lights)
│   ├── yolo_dataset/                    # Prepared YOLO format dataset
│   └── results/                         # Training outputs
│
├── vehicle-pedestrian-detection/        # Vehicle & pedestrian detection
│   ├── training.py
│   ├── dataset/                         # BDD100K dataset
│   └── results/
│
├── models/                              # Pretrained models
│   ├── lane_detection_unet.h5           # U-Net lane detection
│   ├── scnn.pth                         # SCNN lane detection
│   ├── sign_detection.pt                # YOLOv8 sign detection
│   ├── sign_classification.h5           # CNN sign classifier
│   ├── vehicle_pedestrian_detection.pt  # YOLOv8 vehicle/pedestrian
│   ├── traffic_light_detect_class.pt    # YOLOv8 traffic light
│   └── camera_calibration.pkl           # Camera calibration data
│
├── datasets/                            # All datasets (organized)
│   ├── lane-detection/
│   │   ├── culane/                      # CULane dataset
│   │   ├── cityscapes/
│   │   └── processed/
│   ├── traffic-light/
│   │   ├── dtld/
│   │   ├── lisa/
│   │   └── merged/
│   ├── traffic-sign/
│   │   ├── mapillary/
│   │   └── gtsrb/
│   └── vehicle-pedestrian/
│       ├── bdd100k/
│       └── processed/
│
├── results/                             # Training & experiment results
│   ├── lane-detection/
│   │   ├── metrics/
│   │   └── visualizations/
│   ├── traffic-sign/
│   │   ├── metrics/
│   │   └── visualizations/
│   ├── traffic-light/
│   │   ├── metrics/
│   │   └── visualizations/
│   └── vehicle-pedestrian/
│       ├── metrics/
│       └── visualizations/
│
├── images/                              # Sample images & predictions
│   ├── lane-detection/
│   ├── traffic-signs/
│   ├── traffic-lights/
│   └── vehicle-pedestrian/
│
├── videos/                              # Video clips for testing/demo
│   ├── lane-detection/
│   ├── traffic-lights/
│   └── simulation/
│
├── notebooks/                           # Jupyter notebooks (experiments, analysis)
│   ├── collab/
│   │   └── traffic_sign_detection.ipynb
│   ├── kaggle/
│   │   └── traffic-sign-detection.ipynb
│   ├── analysis/
│   ├── training/
│   └── visualization/
│
├── assets/                              # Project assets (GIFs, diagrams, etc)
│   ├── lane_cv.gif
│   ├── lane_nn.gif
│   ├── sign.gif
│   ├── light.gif
│   ├── vehicle.gif
│   └── architecture_diagram.png
│
├── .gitignore
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation (this file)
├── LICENSE
└── setup.py                             # Package setup (optional)
```

</details>

> Descriptions of the configuration files can be found in the `config/README.md` file.

## 🚀 Roadmap

- [x] Sign classification & Detection (CNN)
- [x] Traffic light classification & Detection
- [x] Lane detection (SCNN, CV)
- [x] ⭐ Advanced lane detection using OpenCV (robust city/highway, lighting, outlier handling)
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA)
- [x] Tweak lane detection parameters and thresholds
- [x] ⭐ Integrate Radar
- [x] Integrate Lidar
- [ ] Lidar Object Detection
- [ ] Lidar lane boundry detection
- [x] Modularize and clean up BeamNG.tech pipeline
- [x] ⭐ Integrate vehicle control (autonomous driving logic)
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Test different weather and lighting conditions
- [x] ⭐ Begin integration of other models (sign, light, pedestrian, etc.)
- [x] ⭐ Adaptive Cruise Control
- [ ] Emergency Breaking / Collision Avoidance
- [ ] Weather condition detection
- [x] ⭐ Full Foxglove visualization integration
- [x] ⭐ Modular YAML configuration system
- [x] ⭐ Real-time drive logging and telemetry
- [ ] Blindspot Monitoring

**Future / Stretch Goals**
- [ ] Docker containarization
- [ ] SLAM (simultaneous localization and mapping)
- [ ] GPS/IMU sensor
- [ ] Map Matching algorithm
- [ ] 💤 Global and Local path planning
- [ ] 💤 Behaviour planning and anticipation
- [ ] Test using actual RC car
- [ ] 💤 End-to-end driving policy learning (RL, imitation learning)
- [ ] Multi Camera
- [ ] 💤 Advanced traffic participant prediction (trajectory, intent)

> ⭐ = Complete but still being improved/tuned/changed (not final version)

> 💤 = Minimal Priority, can be addressed later

## 🙏 Credits
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
