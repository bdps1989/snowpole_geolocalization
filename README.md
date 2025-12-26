# Snow Pole Geo-Localization Framework

This repository focuses **exclusively on snow pole geo-localization**, which is a **core sub-module** of a broader **vehicle localization framework using georeferenced snow poles and LiDAR data**. The overall objective is to enable reliable vehicle localization in **GNSS-limited or GNSS-denied environments**, particularly under **harsh Nordic winter conditions**.

Snow poles are treated as **machine-perceivable infrastructure landmarks**, allowing vehicles to estimate their position when GNSS signals are unreliable or unavailable.

---

## 📌 Scope of This Repository

> ⚠️ **Important clarification**

This repository **does NOT implement the full vehicle localization stack**.  
Instead, it **implements and evaluates the snow pole geo-localization component**, which is later integrated into a complete vehicle localization framework.

---

## 🔍 Framework Overview

The snow pole geo-localization workflow consists of the following stages:

### 1. Snow Pole Detection
Snow poles are detected from LiDAR point clouds or LiDAR-derived representations using deep learning–based object detection models.

### 2. Georeferencing of Detected Poles
Detected poles are associated with known **global (map-level) pole coordinates** from a georeferenced database.

### 3. Vehicle Pose Estimation (via Poles)
Vehicle position is estimated by matching detected poles with their known geospatial locations, enabling localization in GNSS-challenged scenarios.

### 4. Evaluation Under Nordic Conditions
The framework is validated using **real-world Nordic winter datasets**, including snow-covered roads, reduced visibility, and harsh weather conditions.

---

## 📂 Repository Structure

```text
snowpole_geolocalization/
│
├── snow_pole_geolocalization.py
│   Main pipeline for snow pole–based geo-localization
│
├── geoloc_utils.py
│   Utility functions for coordinate transformation, matching, and evaluation
│
├── model/
│   Detection models and related components
│
├── Groundtruth_pole_location_at_test_site_E39_Hemnekjølen.csv
│   Georeferenced snow pole locations for evaluation
│
├── Trip068.json
│   Sample trip metadata
│
├── environment.yml
│   Conda environment for reproducibility
│
├── .gitignore
│   Excludes datasets, cache files, and large artifacts
│
└── README.md
```
---


### 5. Data Description

- **LiDAR data** collected along Norwegian highways  
- **Georeferenced snow pole locations** obtained from infrastructure-level mapping  
- **Test site**: E39 – Hemnekjølen, Norway  

⚠️ **Raw LiDAR datasets and large artifacts are not included in this repository** due to size limitations and data-sharing constraints.

---

### 6. Reproducibility

A Conda environment is provided to ensure reproducibility of the experiments.

```bash
conda env create -f environment.yml
conda activate snowpole_geolocalization
```
### 7. Related Publications

If you use this repository, please cite the following works.

Journal Articles

Vehicle Localization Framework Using Georeferenced Snow Poles and LiDAR in GNSS-Limited Environments Under Nordic Conditions
Durga Prasad Bavirisetti, Gabriel E. Berget, Gabriel H. Kiss, Petter Arnesen, Håvard Seter, Frank Lindseth
IEEE Transactions on Intelligent Transportation Systems

SnowPole Detection: A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions
Durga Prasad Bavirisetti, Gabriel H. Kiss, Petter Arnesen, Håvard Seter, S. Tabassum, Frank Lindseth
Data in Brief, Vol. 59, 111403

Conference Papers

Enhancing Vehicle Navigation in GNSS-Limited Environments with Georeferenced Snow Poles
Durga Prasad Bavirisetti
IEEE Symposium Series on Computational Intelligence (SSCI 2025), Trondheim, Norway

A Pole Detection and Geospatial Localization Framework Using LiDAR–GNSS Data Fusion
Durga Prasad Bavirisetti, Gabriel Hanssen Kiss, Frank Lindseth
27th International Conference on Information Fusion (FUSION 2024)

### 8. Project Context & Funding

This research was conducted as part of the project:

“Machine Sensible Infrastructure under Nordic Conditions”
Project Number: 333875

### 9. Citation

If you use this code, dataset references, or methodological ideas, please cite the corresponding publications listed above.

### 10. Contact

Durga Prasad Bavirisetti
Associate Professor – AI & Computer Vision
University of Gävle, Sweden

For questions, collaborations, or extensions of this work, feel free to get in touch.
