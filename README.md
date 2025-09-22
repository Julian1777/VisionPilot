
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

> **Note:** The datasets folder is not included in the repository. You must download and prepare all datasets yourself. The following structure and subfolders are required for the code to work—please organize your downloaded and processed datasets to match this layout:

- **Lane Detection:**
  - Place CU Lane Dataset in `datasets/lane-detection/`
  - Processed Culane with sorted masks, images, and annotations in `lane-detection/processed/` and raw data in `lane-detection/raw/`
- **Traffic Sign Classification:**
  - Place GTSRB Dataset in the appropriate subfolder
- **Traffic Sign Detection:**
  - Place unprocessed Mapillary Sign Dataset in `datasets/traffic-sign/raw`
  - Processed dataset for yolov8 format in `datasets/traffic-sign/processed-yolo/`
- **Traffic Light Detection & Classification:**
  - Place unprocessed DLDT & LISA Datasets in `datasets/traffic-light/raw`
  - Combined DLDT & LISA datasets sorted by light state in `datasets/traffic-light/processed/merged_dataset`
  - Combined dataset processed for YOLO training in `datasets/traffic-light/processed/yolo_dataset`
- **Vehicle & Pedestrian Detection:**
  - Place BDD100K in `datasets/vehicle-pedestrian/` (download from Kaggle profile)
- **Debug Visualizations:**
  - Place traffic light debug visualizations in `datasets/traffic-light/debug_visualizations/`
  - Place results visualizations in `results/traffic-sign-classification/visualizations/` and `results/vehicle-pedestrian/visualizations/`



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
  python beamng_sim/beamng.py
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
- **Lane Detection:** Hough Transform, SCNN (lane-detection-cnn/)
- **Traffic Sign Classification:** CNN classifier
- **Traffic Sign Detector:** YOLOv8 detector (traffic_sign/)
- **Traffic Light Detect/Class:** YOLOv8 detector, classifier (traffic-lights/)
- **Vehicle/Pedestrian:** YOLOv8, SCNN (vehicle-pedestrian-detection/)
- 

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



**Roadmap**
- [x] Sign classification (CNN) *(completed)*
- [x] Traffic light classification *(completed)*
- [x] Lane detection (U-Net, SCNN, Hough) *(completed)*
- [x] ⭐ Advanced lane detection using OpenCV (robust city/highway, lighting, outlier handling) *(completed, still tuning)*
- [x] Integrate and test in BeamNG.tech simulation (replacing CARLA) *(completed)*
- [x] ⭐ Tweak lane detection parameters *(completed, still tuning)*
- [ ] ⭐ Integrate Radar
- [ ] Integrate Lidar
- [x] Modularize and clean up BeamNG.tech pipeline *(completed)*
- [ ] ⭐ Integrate vehicle control (autonomous driving logic)
- [ ] Traffic scenarios: driving in heavy, moderate, and light traffic
- [ ] Add evaluation scripts for all modules
- [ ] Documentation improvements (usage, troubleshooting)
- [x] ⭐ Begin integration of other models (sign, light, pedestrian, etc.)
- [ ] Adaptive Cruise Control utalizitng radar sensor

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
- Special thanks to [Kaggle](https://www.kaggle.com/) for providing free GPU resources for model training without them it would've been imposible to have such good models.

### BeamNG.tech Citation

Use of BeamNG.tech in non-commercial, academic studies should be properly cited in any articles, conference papers, presentations made about research projects, etc. Please adhere to the citation format required by your institution or publication, using the information below:

> **Title:** BeamNG.tech  
> **Author:** BeamNG GmbH  
> **Address:** Bremen, Germany  
> **Year:** 2025  
> **Version:** 0.35.0.0  
> **URL:** https://www.beamng.tech/