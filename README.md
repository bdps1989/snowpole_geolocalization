# Snow Pole Geo-Localization Framework

This repository focuses **exclusively on snow pole geo-localization**, a **core sub-module** of a broader **vehicle localization framework using georeferenced snow poles and LiDAR data**. The objective is to enable reliable vehicle localization in **GNSS-limited or GNSS-denied environments**, particularly under **harsh Nordic winter conditions**.

Snow poles are treated as **machine-perceivable infrastructure landmarks**, allowing vehicles to estimate their position when GNSS signals are unreliable or unavailable.

---

## Background and Motivation

Reliable vehicle localization remains a significant challenge in Nordic environments due to **GNSS signal degradation**, and **snow-covered infrastructure caused by harsh weather conditions**. These factors limit the effectiveness of conventional positioning and perception methods, particularly in winter scenarios where visual cues are sparse or unreliable.

This work investigates the use of **snow poles as stable, georeferenced roadside landmarks** that can be reliably detected using **LiDAR-based perception**. By leveraging snow poles as machine-perceivable infrastructure, the proposed approach enables robust vehicle localization in environments where GNSS performance is degraded or unavailable.


---

## Scope of This Repository


This repository implements the snow-pole-based vehicle localization framework proposed in the IEEE Transactions on Intelligent Transportation Systems paper “Vehicle Localization Framework Using Georeferenced Snow Poles and LiDAR in GNSS-Limited Environments under Nordic Conditions” 

The scope of this repository is to provide an end-to-end localization pipeline that enables robust vehicle positioning in GNSS-limited or GNSS-denied environments, particularly under harsh Nordic winter conditions, by leveraging georeferenced snow poles as machine-perceivable infrastructure landmarks.

Specifically, this repository includes:

Snow pole–based landmark utilization using pre-measured, georeferenced roadside snow poles

LiDAR-based perception and localization, including snow pole detection and relative pose estimation

GNSS–LiDAR fusion, dynamically alternating between GNSS-based positioning and LiDAR-based localization depending on GNSS availability

Incremental vehicle navigation (odometry) using point-cloud registration to maintain continuity during GNSS outages

Iterative vehicle pose refinement using snow pole geolocalization to reduce drift and improve accuracy

Quantitative and qualitative evaluation tools for comparing estimated vehicle positions against GNSS reference trajectories

---

## Snow Pole Geo-Localization Framework Overview

This framework provides a structured approach for **detecting and geolocating snow poles** using **continuous GNSS and LiDAR data**, specifically designed for **harsh Nordic winter environments** where visual cues are limited. GNSS measurements provide a global positioning reference, while high-resolution LiDAR data enable reliable detection and relative localization of snow poles in the vehicle’s surroundings.

Detected snow poles are first localized **relative to the vehicle** by estimating their distance and bearing from LiDAR observations. These relative measurements are then fused with GNSS positioning to compute **absolute geolocations** of the poles, which are aligned with pre-measured ground-truth coordinates at the test site. The framework is evaluated by comparing estimated pole positions against known reference locations, ensuring accuracy and robustness.

By tightly integrating GNSS and LiDAR sensing, this snow-pole geo-localization framework enables **reliable vehicle localization in GNSS-limited or winter-degraded conditions**, where conventional perception and positioning methods often fail.


### 1. Snow Pole Detection

To enable landmark-based localization in **GNSS-degraded or GNSS-denied environments**, snow pole detection is formulated as a **2D object detection problem** using **LiDAR-derived signal images**.

A **YOLOv5s-based convolutional neural network (CNN)** is initially employed to detect snow poles from LiDAR signal images. The model was trained on a curated dataset[2] of **1,954 manually annotated images**, split into **1,367 training**, **390 validation**, and **197 test images**. Initial annotations were created using **Roboflow** and subsequently refined in **CVAT** to correct labeling inaccuracies caused by JPEG compression artifacts, improving annotation fidelity.

To maximize the utilization of labeled data, the original training and test sets were combined[3], resulting in **1,564 images** used for training. The final model was evaluated on a held-out validation set of **390 images**, achieving a **precision of 0.873**, **recall of 0.826**, and **mAP@0.5 of 0.893**, demonstrating robust performance under visually sparse and snow-covered Nordic conditions.

Each detected bounding box is mapped back to the corresponding region in the **LiDAR range image** to retrieve spatial information. These pole detections serve as **key inputs to downstream data association and vehicle pose estimation modules** within the snow pole geo-localization pipeline.

---

#### Extended Model Evaluation

Building on the initial Snow Pole Detection dataset[2],[6] and prior LiDAR–GNSS localization work[3], we further conducted an extensive evaluation of **six YOLO-based object detection architectures[4]**:

- YOLOv5s  
- YOLOv7-tiny  
- YOLOv8n  
- YOLOv9t  
- YOLOv10n  
- YOLOv11n  

The models are benchmarked across multiple **LiDAR-derived input modalities**, including single-channel representations (**Reflectance**, **Signal**, **Near-Infrared**) and **six pseudo-color combinations** constructed from these channels.

Performance is evaluated using **Precision**, **Recall**, **mAP@50**, **mAP@50–95**, and **GPU inference latency**. To enable systematic comparison between accuracy and real-time capability, a **composite Rank Score** is defined. Results indicate that **YOLOv9t** achieves the highest detection accuracy, while **YOLOv11n** offered the best balance between accuracy and inference speed, making it well suited for real-time deployment.

Pseudo-color combinations—particularly those fusing **Near-Infrared, Signal, and Reflectance**—consistently outperformed single-channel modalities and yielded the highest Rank Scores. Based on these findings, **multimodal LiDAR configurations** (e.g., Combination 4 and Combination 5) are recommended to enhance detection robustness.

All datasets, trained models, and source code are publicly available via the  **GitHub repository:** (https://github.com/MuhammadIbneRafiq/Extended-evaluation-snowpole-lidar-dataset)  and the accompanying **Mendeley Data archive** [7], supporting full reproducibility and further research.




### 2. Georeferencing of Detected Poles

Detected snow poles are initially localized **relative to the vehicle** by estimating their distance and bearing from LiDAR observations. These relative measurements are then fused with **continuous GNSS positioning** to compute the **absolute (map-level) geolocation** of each pole, which is aligned with pre-measured ground-truth coordinates at the test site.

Experimental evaluation is performed using a **pretrained snow pole detection model** in combination with **ROS bag recordings containing synchronized LiDAR-derived images and GNSS measurements**. The dataset used for evaluation is publicly available on **Kaggle** [8].

This integrated **GNSS–LiDAR fusion framework** enables accurate snow pole geolocation under harsh winter conditions and forms a reliable foundation for vehicle localization. Once validated in GNSS-available scenarios, the approach can be extended to support **odometry-based vehicle localization** in **GNSS-limited or GNSS-denied environments**.

#### Using ROS Bag Data for Visualization and Analysis

Download **any one of the ROS bag files** associated with this project and place it inside the `snowpole_geolocalization/` directory (or update the file paths in the scripts accordingly). The ROS bag files are publicly available on **Kaggle**[8] under the folder `snow_pole_geo_localization_data`.

The following ROS bag files are provided:

- **`2024-02-28-12-59-51.bag` (41.24 GB)**  
  Contains the **complete raw dataset**, including all recorded sensor streams.

- **`2024-02-28-12-59-51_no_unwanted_topics.bag` (5.71 GB)**  
  A reduced version containing **only the LiDAR-derived images and GNSS data** required to conduct the experiments presented in this project.

To visualize and process various sensor data—such as **raw LiDAR point clouds**, **LiDAR-derived images**, and **GNSS information**—use the utilities provided in the `rosbag_utils/` folder of this repository. These scripts support data inspection, visualization, and preprocessing for reproducing the snow pole geo-localization experiments.




---


## Repository Structure

```text
snowpole_geolocalization/
│
├── snow_pole_geolocalization.py
│   Main pipeline for snow pole–based geo-localization
│
├── geoloc_utils.py
│   Utility functions for coordinate transformation, matching, and evaluation
│
├── rosbag_utils/
│   Utilities for ROS bag processing and visualization
│   │
│   ├── extract_rosbag_topics.py
│   │   Extracts required sensor topics from ROS bag files
│   │
│   ├── gnss_and_groundtruth_snowpole_visualization.py
│   │   Visualizes GNSS data alongside ground-truth snow pole locations
│   │
│   ├── lidar_image_visualization.py
│   │   Visualization of LiDAR-derived images
│   │
│   ├── range_image_to_pointcloud_visualization.py
│   │   Generates and visualizes point clouds from LiDAR range images
│   │
│   └── realpointcloud_vlsulaization.py
│       Visualizes raw LiDAR point clouds from ROS bag data
│
├── model/
│   Detection models and related components
│
├── Groundtruth_pole_location_at_test_site_E39_Hemnekjølen.csv
│   Georeferenced snow pole locations used for evaluation
│
├── Trip068.json
│   Sample trip metadata and configuration file
│
├── environment.yml
│   Conda environment specification for reproducibility
│
└── README.md
    Project description, usage instructions, and references
```
---

### 4. Data Description

The dataset used in this repository comprises **LiDAR sensor data** collected along Norwegian highways, with a particular focus on the **E39 – Hemnekjølen test site in Norway**. In addition to LiDAR and GNSS measurements, the dataset includes **georeferenced snow pole locations** obtained through infrastructure-level mapping, which serve as fixed landmarks for localization and evaluation.

Due to data size limitations and data-sharing constraints, **raw LiDAR datasets and other large artifacts are not included directly in this repository**. Instead, the repository provides the necessary tools, scripts, and metadata to support data processing, visualization, and reproducibility using externally hosted datasets[6-8].


---

## 5. Reproducibility

A Conda environment is provided to ensure full reproducibility of the experiments. The environment is explicitly configured using **Python 3.9.18** to maintain compatibility with all dependencies.

---

### Create the Conda Environment

Create a new Conda environment with the required Python version:

```bash
conda create -n snowpole_geolocalization python=3.9.18 -y
```
### Activate the Environment
```text
conda activate snowpole_geolocalization

```
### Install Dependencies

Install all required packages using the provided environment.yml file:
```bash
conda env update -f environment.yml --prune
```
### Verify Installation
```bash
python --version
```
## 6. Related Publications and Citation

If you use this code, dataset references, or methodological ideas, please cite the corresponding publications listed below.

### Journal Articles

1. **Bavirisetti, D. P., Berget, G. E., Kiss, G. H., Arnesen, P., Seter, H., & Lindseth, F. (2025). Vehicle Localization Framework Using Georeferenced Snow Poles and LiDAR in GNSS-Limited Environments Under Nordic Conditions. IEEE Transactions on Intelligent Transportation Systems.**

2. **Bavirisetti, D. P., Kiss, G. H., Arnesen, P., Seter, H., Tabassum, S., & Lindseth, F. (2025). SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions. Data in Brief, 59, 111403.**  
 

---

### Conference Papers

3. **Bavirisetti, D. P., Kiss, G. H., & Lindseth, F. (2024, July). A pole detection and geospatial localization framework using liDAR-GNSS data fusion. In 2024 27th International Conference on Information Fusion (FUSION) (pp. 1-8). IEEE.**
4. **Bavirisetti, D. P. (2025). Extended evaluation of SnowPole detection for machine-perceivable infrastructure for Nordic winter conditions: A comparative study of object detection models. Available at SSRN 5386946.**
5. **Bavirisetti, D. P. (2025). Enhancing vehicle navigation in GNSS-limited environments with georeferenced snow poles. In 2025 IEEE Symposium Series on Computational Intelligence (SSCI), Trondheim, Norway, 17-20 March 2025. IEEE. (Poster Presentation)**
   
### Datasets
6. **Bavirisetti, Durga Prasad; Kiss, Gabriel Hanssen ; Arnesen, Petter ; Seter, Hanne ; Tabassum, Shaira ; Lindseth, Frank  (2024), “SnowPole Detection: A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions”, Mendeley Data, V2, [https://doi.org/10.17632/tt6rbx7s3h.2](https://doi.org/10.17632/tt6rbx7s3h.2)**
7. **Bavirisetti, Durga Prasad; Rafiq, Muhammad; Kiss, Gabriel Hanssen ; Lindseth, Frank  (2025), “Extended Evaluation of SnowPole Detection for Machine-Perceivable Infrastructure for Nordic Winter Conditions: A Comparative Study of Object Detection Models”, Mendeley Data, V3, [https://doi.org/10.17632/tt6rbx7s3h.3](https://doi.org/10.17632/tt6rbx7s3h.3)**
8. **Durga Prasad Bavirisetti, Gabriel Hanssen Kiss, Frank Lindseth, Petter Arnesen, and Hanne Seter. (2025). Data for the snowpole based vehicle localization [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/14311103**
### 7. Project Context & Funding

This research was conducted as part of the project: “Machine Sensible Infrastructure under Nordic Conditions” with Project Number: 333875


### 8. Contact

Durga Prasad Bavirisetti,
Associate Professor – AI & Computer Vision,
University of Gävle, Sweden

For questions, collaborations, or extensions of this work, feel free to get in touch.
