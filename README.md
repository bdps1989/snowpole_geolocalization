<!-- comment section -->
# Snow Pole Geo-Localization Framework
---
This repository focuses **exclusively on snow pole geo-localization**, which constitutes a **core sub-module** of a broader **vehicle localization framework based on georeferenced snow poles and LiDAR data** [1]. The complete vehicle localization pipeline is available at:

https://github.com/bdps1989/Snow-pole-based-vehicle-localization-temporary

The snow pole geo-localization concept was **initially introduced in [3]** and has been **further extended and refined in [1]**. The primary objective of this module is to enable **robust and reliable vehicle localization in GNSS-limited or GNSS-denied environments**, with a particular focus on **harsh Nordic winter conditions**.

In this framework, snow poles are treated as **machine-perceivable roadside infrastructure landmarks**. By leveraging their stable geometry and georeferenced positions, the system enables accurate vehicle position estimation when GNSS signals are degraded, intermittent, or unavailable.



<!--
---

## Snow Pole Geo-Localization Framework Overview

This framework provides a structured approach for **detecting and geolocating snow poles** using **continuous GNSS and LiDAR data**, specifically designed for **harsh Nordic winter environments** where visual cues are limited. GNSS measurements provide a global positioning reference, while high-resolution LiDAR data enable reliable detection and relative localization of snow poles in the vehicle’s surroundings.

Detected snow poles are first localized **relative to the vehicle** by estimating their distance and bearing from LiDAR observations. These relative measurements are then fused with GNSS positioning to compute **absolute geolocations** of the poles, which are aligned with pre-measured ground-truth coordinates at the test site. The framework is evaluated by comparing estimated pole positions against known reference locations, ensuring accuracy and robustness.

By tightly integrating GNSS and LiDAR sensing, this snow-pole geo-localization framework enables **reliable vehicle localization in GNSS-limited or winter-degraded conditions**, where conventional perception and positioning methods often fail.
-->

### 1. Snow Pole Detection

To enable landmark-based localization in **GNSS-degraded or GNSS-denied environments**, snow pole detection is formulated as a **2D object detection problem** using **LiDAR-derived signal images**.

A **YOLOv5s-based convolutional neural network (CNN)** is initially employed to detect snow poles from LiDAR signal images. The model was trained on a curated dataset[2] of **1,954 manually annotated images**, split into **1,367 training**, **390 validation**, and **197 test images**. Initial annotations were created using **Roboflow** and subsequently refined in **CVAT** to correct labeling inaccuracies caused by JPEG compression artifacts, improving annotation fidelity.

To maximize the utilization of labeled data, the original training and test sets were combined[3], resulting in **1,564 images** used for training. The final model was evaluated on a held-out validation set of **390 images**, achieving a **precision of 0.873**, **recall of 0.826**, and **mAP@0.5 of 0.893**, demonstrating robust performance under visually sparse and snow-covered Nordic conditions.

Each detected bounding box is mapped back to the corresponding region in the **LiDAR range image** to retrieve spatial information. These pole detections serve as **key inputs to downstream data association and vehicle pose estimation modules** within the snow pole geo-localization pipeline.



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

This integrated **GNSS–LiDAR fusion framework** enables accurate snow pole geolocation under harsh winter conditions and forms a reliable foundation for vehicle localization. Once validated in GNSS-available scenarios, the approach can be extended to support **odometry-based vehicle localization** in **GNSS-limited or GNSS-denied environments[1]**.



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

## Data Description

The dataset used in this repository consists of **real-world LiDAR and GNSS sensor data** collected along Norwegian highways, with a primary focus on the **E39 – Hemnekjølen test site in Norway**. This test site spans approximately **4.2 km** and includes **mountainous terrain, forested regions, and open landscapes**, making it particularly suitable for evaluating vehicle localization performance under **GNSS-limited conditions and harsh Nordic winter environments**.

The dataset includes the following components:

- **High-resolution LiDAR data** captured using a **128-channel Ouster OS2-128 sensor**, providing full 360° environmental coverage  
- **Continuous GNSS measurements** from dual GNSS receivers mounted on the vehicle, used for initialization, reference positioning, and quantitative evaluation  
- **Georeferenced snow pole locations**, manually measured at the infrastructure level and used as **fixed, stable landmarks** for localization  
- **Synchronized ROS bag recordings** containing LiDAR-derived images, point clouds, and GNSS data required to reproduce the snow pole geo-localization and vehicle localization experiments  

Snow poles are distributed along both sides of the roadway and are designed to remain visible under heavy snow. Consequently, they serve as **reliable machine-perceivable infrastructure landmarks** when lane markings, traffic signs, and other visual cues are partially or fully obscured.



### Publicly Available Datasets and Usage

Due to data size limitations and data-sharing constraints, **raw LiDAR point clouds and full ROS bag recordings are not included directly in this repository**. Instead, all datasets required to reproduce the experiments are hosted externally and are publicly available via the following sources:

- **SnowPole Detection Dataset (LiDAR-derived images)**  
  *Mendeley Data, Version 2* [6]  
  https://doi.org/10.17632/tt6rbx7s3h.2  

- **Extended Evaluation of SnowPole Detection Dataset (LiDAR-derived images)**  
  *Mendeley Data, Version 3* [7]  
  https://doi.org/10.17632/tt6rbx7s3h.3  

- **Snow-Pole-Based Vehicle Localization Dataset (ROS bags)**  
  *Kaggle Dataset* [8]  
  https://doi.org/10.34740/KAGGLE/DSV/14311103  

Datasets **[6]** and **[7]** are used to **train the YOLOv5-based snow pole detection model[10]**. The resulting **pretrained snow pole detection model** is then employed—together with the **ROS bag data** from dataset **[8]**—to evaluate both the **snow pole geo-localization framework** and the **end-to-end snow-pole-based vehicle localization framework**.

The Kaggle dataset **[8]** provides the following ROS bag files:

- **`2024-02-28-12-59-51.bag` (41.24 GB)**  
  Contains the **complete raw dataset**, including all recorded sensor streams captured during the data collection campaign.

- **`2024-02-28-12-59-51_no_unwanted_topics.bag` (5.71 GB)**  
  A **reduced and experiment-ready version** containing only the **LiDAR-derived images and GNSS data** required to conduct the snow pole geo-localization and vehicle localization experiments presented in this project.
 

Together, these datasets provide a **complete and reproducible foundation** for training, evaluation, and benchmarking of snow-pole-based localization methods under **GNSS-limited and winter-degraded sensing conditions**.

#### Using ROS Bag Data for Visualization and Analysis

Download **any one of the ROS bag files** associated with this project and place it inside the `snowpole_geolocalization/` directory (or update the file paths in the scripts accordingly). The ROS bag files are publicly available on **Kaggle**[8] under the folder `snow_pole_geo_localization_data`.


To visualize and process various sensor data—such as **raw LiDAR point clouds**, **LiDAR-derived images**, and **GNSS information**—use the utilities provided in the `rosbag_utils/` folder of this repository. These scripts support data inspection, visualization, and preprocessing for reproducing the snow pole geo-localization experiments.


---

## Reproducibility

A Conda environment is provided to ensure full reproducibility of the experiments. The environment is explicitly configured using **Python 3.9.18** to maintain compatibility with all dependencies.

---

### 1. Create the Conda Environment

Create a new Conda environment with the required Python version:

```bash
conda create -n snowpole_geolocalization python=3.9.18 -y
```
### 2. Activate the Environment
```text
conda activate snowpole_geolocalization

```
### 3. Install Dependencies

Install all required packages using the provided environment.yml file:
```bash
conda env update -f environment.yml --prune
```
### 4. Verify Installation
```bash
python --version
```
---
## Related Publications and Citation

If you use this code, dataset references, or methodological ideas, please cite the corresponding publications listed below.

### Journal Articles

1. **Bavirisetti, D. P., Berget, G. E., Kiss, G. H., Arnesen, P., Seter, H., & Lindseth, F. (2025). Vehicle Localization Framework Using Georeferenced Snow Poles and LiDAR in GNSS-Limited Environments Under Nordic Conditions. IEEE Transactions on Intelligent Transportation Systems.**

2. **Bavirisetti, D. P., Kiss, G. H., Arnesen, P., Seter, H., Tabassum, S., & Lindseth, F. (2025). SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions. Data in Brief, 59, 111403.**  
 


### Conference Papers

3. **Bavirisetti, D. P., Kiss, G. H., & Lindseth, F. (2024, July). A pole detection and geospatial localization framework using liDAR-GNSS data fusion. In 2024 27th International Conference on Information Fusion (FUSION) (pp. 1-8). IEEE.**
4. **Bavirisetti, D. P. (2025). Extended evaluation of SnowPole detection for machine-perceivable infrastructure for Nordic winter conditions: A comparative study of object detection models. Available at SSRN 5386946.**
5. **Bavirisetti, D. P. (2025). Enhancing vehicle navigation in GNSS-limited environments with georeferenced snow poles. In 2025 IEEE Symposium Series on Computational Intelligence (SSCI), Trondheim, Norway, 17-20 March 2025. IEEE. (Poster Presentation)**
   
### Datasets
6. **Bavirisetti, Durga Prasad; Kiss, Gabriel Hanssen ; Arnesen, Petter ; Seter, Hanne ; Tabassum, Shaira ; Lindseth, Frank  (2024), “SnowPole Detection: A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions”, Mendeley Data, V2, [https://doi.org/10.17632/tt6rbx7s3h.2](https://doi.org/10.17632/tt6rbx7s3h.2)**
7. **Bavirisetti, Durga Prasad; Rafiq, Muhammad; Kiss, Gabriel Hanssen ; Lindseth, Frank  (2025), “Extended Evaluation of SnowPole Detection for Machine-Perceivable Infrastructure for Nordic Winter Conditions: A Comparative Study of Object Detection Models”, Mendeley Data, V3, [https://doi.org/10.17632/tt6rbx7s3h.3](https://doi.org/10.17632/tt6rbx7s3h.3)**
8. **Durga Prasad Bavirisetti, Gabriel Hanssen Kiss, Frank Lindseth, Petter Arnesen, and Hanne Seter. (2025). Data for the snowpole based vehicle localization [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/14311103**
---



## References

9. Arnold, E., Mozaffari, S., & Dianati, M. (2021). Fast and robust registration of partially overlapping point clouds. IEEE Robotics and Automation Letters, 7(2), 1502-1509.
10. Jocher, G. (2020). YOLOv5 by ultralytics (version 7.0)[computer software].


---
##  Project Context & Funding


This research was conducted as part of the project: “Machine Sensible Infrastructure under Nordic Conditions” with Project Number: 333875

---
## Contact

**Durga Prasad Bavirisetti**  

Senior Lecturer - Artificial Intelligence & Computer Vision

Department of Computer Science

University of Gävle, Sweden

For questions, collaborations, or extensions of this work, feel free to get in touch.

