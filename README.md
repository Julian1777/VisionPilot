
# 🚗 Self-Driving Car Simulation: Computer Vision, Deep Learning & Real-Time Perception (BeamNG.tech)

![GitHub stars](https://img.shields.io/github/stars/Julian1777/self-driving-project?style=social)


A modular Python project for autonomous driving research and prototyping, now fully integrated with the BeamNG.tech simulator. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) to tackle:

- 🛣️ Lane detection (Hough Transform, SCNN, city/highway scenarios)
- 🛑 Traffic sign classification & detection (CNN, YOLOv8, GTRSB, LISA, Mapillary)
- 🚦 Traffic light detection & classification (YOLOv8, DLDT, LISA)
- 🚗 Vehicle & pedestrian detection and recognition (YOLOv8, SCNN, BDD100K)
- 🧠 Multi-model inference, real-time simulation, and visualization (BeamNG.tech)

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

- Lane detection with SCNN and OpenCV (comparison)
- Traffic sign classification using CNN
- Traffic light detection (YOLO) + classification
- Video-based inference pipeline
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

## 📚 Datasets & Sources
- **Lane Detection:**
  - CU Lane Dataset (`datasets/lane-detection/`)
  - Processed Culane with sorted masks, images, and annotations (`lane-detection/processed/`, Raw Dataset `lane-detection/raw/`)
- **Traffic Sign Classification:**
  - GTSRB Dataset
- **Traffic Sign Detection:**
  - Unprocessed Mapillary Sign Dataset (`datasets/traffic-sign/raw`)
  - Processed dataset for yolov8 format (`datasets/traffic-sign/processed-yolo/`)
- **Traffic Light Detection & Classification:**
  - Unprocessed DLDT & LISA Datasets (`datasets/traffic-light/raw`)
  - Combined DLDT & LISA datasets sorted by light state(`datasets/traffic-light/processed/merged_dataset`)
  - Combined Dataset processed for YOLO training(`datasets/traffic-light/processed/yolo_dataset`)
- **Vehicle & Pedestrian Detection:**
  - BDD100K (Not in repo due to size, can be found on kaggle profile) (`datasets/vehicle-pedestrian/`)
- **Debug Visualizations:**
  - Traffic light debug visualizations (`datasets/traffic-light/debug_visualizations/`)
  - Results visualizations (`results/traffic-sign-classification/visualizations/`, `results/vehicle-pedestrian/visualizations/`)

## 📊 Results

| Model        | Task                               | Accuracy / IoU | Dataset   |    Size    | Epochs   |
|--------------|------------------------------------|----------------|-----------|------------|----------|
| CNN          | Sign Classification                | 89%            | GTRSB     |            |20        |
| YOLO       | Sign Detection                     | 89%            | Mapillary |            |50        |
| YOLO       | Traffic light Light Detection      | mAP x          |           |            |50        |
| SCNN         | Lane Clasification                     | IoU x          | Culane    |            |x         |
| CV         | Lane Detection                     | x          | N/A    |            |x         |
| YOLO         | Vehicle & Pedestrian detection     | IoU x          | BDD       | 100k       |30        |




## ⚡ Quickstart & Usage

1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
2. **Run a demo (BeamNG.tech):**
  ```bash
  python beamng_sim/beamng.py
  ```
  - Make sure you have BeamNG.tech installed and properly licensed. See [BeamNG.tech documentation](https://www.beamng.tech/) for setup instructions.
3. **Train a model:**
  See notebooks or scripts in each module folder.


## 📝 Setup & Installation
- Python 3.8+
- See `requirements.txt` for all dependencies
- Required: BeamNG.tech simulator for real-time testing ([Download & License](https://www.beamng.tech/))


## 🧠 Model Details
All models are located in the models folder
- **Lane Detection:** Hough Transform, SCNN (lane-detection-cnn/)
- **Traffic Sign Classification:** CNN classifier
- **Traffic Sign Detector:** YOLO detector (traffic_sign/)
- **Traffic Light Detect/Class:** YOLOv8 detector, classifier (traffic-lights/)
- **Vehicle/Pedestrian:** YOLOv8, SCNN (vehicle-pedestrian-detection/)

## 📊 Results
- All training results, metrics, and visualizations are in `results/`
- Example:
  - `results/traffic-sign-classification/metrics/` (JSON, curves)
  - `results/traffic-sign-detection/weights/` (YOLO checkpoints)
  - `results/vehicle-pedestrian/visualizations/` (confusion matrices, sample batches)


## 📂 Folder Structure

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
├── lane-detection/               # Lane detection (Hough, city/highway)
│   ├── city/
│   └── highway/
├── lane-detection-cnn/           # CNN/SCNN lane detection, model tests
├── traffic_sign/                 # Traffic sign detection/classification
├── traffic-lights/               # Traffic light detection/classification
├── vehicle-pedestrian-detection/ # Vehicle & pedestrian detection
├── models/                       # Pretrained models (YOLO, SCNN, CNN, etc)

├── datasets/                     # All datasets (see below)
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


**Completed**
- [x] Sign classification (CNN)
- [x] Traffic light classification
- [x] Lane detection (U-Net, SCNN, Hough)



**In Progress / Near-Term**
- [x] ⭐ Advanced lane detection using OpenCV (robust city/highway, lighting, outlier handling)
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA)
- [x] ⭐ Tweak lane detection parameters
- [ ] ⭐ Integrate radar sensor data (LiDAR)
- [x] Modularize and clean up BeamNG.tech pipeline
- [ ] ⭐ Integrate vehicle control (autonomous driving logic)
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Add evaluation scripts for all modules
- [ ] Documentation improvements (usage, troubleshooting)
- [ ] Begin integration of other models (sign, light, pedestrian, etc.)

**Future / Stretch Goals**
- [x] ⭐ Real-time sensor fusion (camera, radar, LiDAR)
- [ ] Multi-camera support (360° perception)
- [ ] End-to-end driving policy learning (RL, imitation learning)
- [ ] Advanced traffic participant prediction (trajectory, intent)
- [ ] Integration with ROS (Robot Operating System)
- [ ] Interactive web dashboard for results/visualizations

> ⭐ = Complete but still being improved/tuned (not final version)


## 🤝 Contributing
- Pull requests welcome!
- Please open issues for bugs, feature requests, or questions.


## 🙏 Credits
- Datasets: CU Lane, LISA, GTRSB, Mapillary, BDD100K
- Models: Ultralytics YOLOv8, custom CNNs
- Simulation: BeamNG.tech ([BeamNG GmbH](https://www.beamng.tech/))

### BeamNG.tech Citation

Use of BeamNG.tech in non-commercial, academic studies should be properly cited in any articles, conference papers, presentations made about research projects, etc. Please adhere to the citation format required by your institution or publication, using the information below:

> **Title:** BeamNG.tech  
> **Author:** BeamNG GmbH  
> **Address:** Bremen, Germany  
> **Year:** 2025  
> **Version:** 0.35.0.0  
> **URL:** https://www.beamng.tech/