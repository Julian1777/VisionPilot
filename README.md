
# 🚗 Self-Driving Car Simulation: Computer Vision, Deep Learning & Real-Time Perception (BeamNG.tech)

<p align="center">
  <a href="https://star-history.com/#Julian1777/self-driving-project&Date">
    <img src="https://api.star-history.com/svg?repos=Julian1777/self-driving-project&type=Date" alt="Star History Chart" />
  </a>
</p>

A modular Python project for autonomous driving research and prototyping, now fully integrated with the BeamNG.tech simulator. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) to tackle:

- 🛣️ Lane detection (Traditional CV, SCNN, capable of city & highway scenarios)
- 🛑 Traffic sign classification & detection (CNN, YOLOv8)
- 🚦 Traffic light detection & classification (YOLOv8, CV, CNN)
- 🚗 Vehicle & pedestrian detection and recognition (YOLOv8)
- 🧠 Multi-model inference, real-time simulation, autonomous driving, and visualization (BeamNG.tech)

Features robust training pipelines, multi-model inference, and a flexible folder structure for easy experimentation and extension. The project is designed for research and prototyping in realistic driving environments using BeamNG.tech.



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

- Lane detection with SCNN
- Lane Detection using Traditional CV
- Traffic Sign Classification + Detection
- Traffic Light Classification + Detection
- Video-based inference pipeline
- Sensor Fusion using Lidar + Radar
- Multi-window simulation using BeamNG.tech
- Real-time perception and control in BeamNG.tech


## 🛠️ Built With

- TensorFlow / Keras
- OpenCV
- YOLOv8 (Ultralytics)
- Python
- BeamNG.tech (https://www.beamng.tech/)


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
2. **Run a demo (BeamNG.tech):**
  ```bash
  python -m beamng_sim.beamng
  ```
  - Make sure you have BeamNG.tech installed and properly licensed. See [BeamNG.tech documentation](https://www.beamng.tech/) for setup instructions.

> **Important:** You must ensure that all required models (e.g., trained weights, .h5/.pt files) and configuration files are placed in the correct directories as expected by the code. The folder structure shown below must be followed, and missing files or incorrect paths will cause errors. See each module's README or script comments for details on required files and their locations.

3. **Train a model:**
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
self-driving-car-simulation/
├── beamng_sim/                   # BeamNG.tech simulation modules and utilities
│   ├── __init__.py
│   ├── beamng.py                 # Main BeamNG.tech interface/entry point
│   ├── debug_output/
│   │   └── alotofnoise/          # Debug images for lane detection, perspective, etc.
│   ├── lane_detection/           # Lane detection algorithms and utilities
│   │   ├── __init__.py
│   │   ├── color_threshold_debug.py
│   │   ├── lane_finder.py
│   │   ├── main.py
│   │   ├── metrics.py
│   │   ├── old_lane_detection.py
│   │   ├── perspective.py
│   │   ├── thresholding.py
│   │   └── visualization.py
│   ├── lidar/                    # LiDAR sensor simulation and visualization
│   │   ├── lidar_testing.py
│   │   ├── lidar.py
│   │   ├── main.py
│   │   ├── Screenshot 2025-09-03 183757.png
│   │   └── visualization_tool.py
│   ├── sign/                     # Traffic sign detection/classification
│   │   ├── detect_classify.py
│   │   └── main.py
│   ├── utils/                    # Utility modules (e.g., PID controller)
│   │   └── pid_controller.py
│   └── vehicle_obstacle/         # Vehicle and obstacle detection
│       ├── main.py
│       └── vehicle_obstacle_detection.py
├── lane-detection/               # Lane detection (CV, city/highway)
│   ├── city/
│   └── highway/
├── lane-detection-cnn/           # CNN/SCNN lane detection, model tests
├── traffic_sign/                 # Traffic sign detection/classification
├── traffic-lights/               # Traffic light detection/classification
├── vehicle-pedestrian-detection/ # Vehicle & pedestrian detection
├── models/                       # Pretrained models (YOLO, SCNN, CNN, etc)

├── datasets/                     # All datasets (see above)
│   ├── lane-detection/
│   ├── traffic-light/
│   ├── traffic-sign/
│   └── vehicle-pedestrian/

> **Note:** Due to size and licensing restrictions, datasets are not included in this repository. You must download all datasets yourself from their respective sources.
├── results/                      # Training results, metrics, visualizations
├── images/                       # Sample images, predictions, training data
├── notebooks/                    # Jupyter notebooks (experiments, training)
└── videos/                       # Video clips for testing/demo
```

</details>

## 🚀 Roadmap

- [x] Sign classification & Detection (CNN)
- [x] Traffic light classification & Detection
- [x] Lane detection (SCNN, CV)
- [x] ⭐ Advanced lane detection using OpenCV (robust city/highway, lighting, outlier handling)
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA)
- [x] Tweak lane detection parameters and thresholds
- [ ] Integrate Radar
- [x] Integrate Lidar
- [ ] Lidar Object Detection (Maybe even train a model)
- [ ] Lidar lane boundry detection (Debug Not working, not implemented in pipeline)
- [x] Modularize and clean up BeamNG.tech pipeline
- [x] ⭐ Integrate vehicle control (autonomous driving logic)
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Test different weather and lighting conditions
- [x] ⭐ Begin integration of other models (sign, light, pedestrian, etc.)
- [ ] Adaptive Cruise Control
- [ ] Emergency Breaking / Collision Avoidance
- [ ] Weather condition detection

**Future / Stretch Goals**
- [ ] Docker containarization
- [ ] Blindspot Monitoring
- [ ] SLAM (simultaneous localization and mapping)
- [ ] GPS/IMU sensor
- [ ] Map Matching algorithm
- [ ] Global and Local path planning
- [ ] 💤 Behaviour planning and anticipation
- [ ] Test using actual RC car
- [ ] 💤 End-to-end driving policy learning (RL, imitation learning)
- [ ] 💤 Multi Camera
- [ ] 💤 Advanced traffic participant prediction (trajectory, intent)
- [ ] 💤 Interactive web dashboard for results/visualizations



> ⭐ = Complete but still being improved/tuned (not final version)

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
