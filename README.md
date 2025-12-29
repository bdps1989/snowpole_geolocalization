# Snow Pole Geo-Localization Framework

This repository focuses **exclusively on snow pole geo-localization**, a **core sub-module** of a broader **vehicle localization framework using georeferenced snow poles and LiDAR data**. The objective is to enable reliable vehicle localization in **GNSS-limited or GNSS-denied environments**, particularly under **harsh Nordic winter conditions**.

Snow poles are treated as **machine-perceivable infrastructure landmarks**, allowing vehicles to estimate their position when GNSS signals are unreliable or unavailable.

---

## Background and Motivation

Reliable vehicle localization remains challenging in Nordic environments due to:
- GNSS signal degradation
- Snow-covered infrastructure
- Reduced visibility and harsh weather conditions

This work investigates **snow poles as stable, georeferenced roadside landmarks** that can be leveraged for localization using LiDAR-based perception.

---

## Scope of This Repository


This repository **does NOT implement the full vehicle localization stack**.  
Instead, it implements and evaluates **only the snow pole geo-localization component**, which is later integrated into a complete vehicle localization framework.

---

## Framework Overview

The snow pole geo-localization workflow consists of the following stages:

### 1. Snow Pole Detection
Snow poles are detected from LiDAR point clouds or LiDAR-derived representations using deep learning–based object detection models.

### 2. Georeferencing of Detected Poles
Detected poles are associated with known **global (map-level) snow pole coordinates** from a georeferenced database.

### 3. Evaluation Under Nordic Conditions
The framework is validated using **real-world Nordic winter datasets**, including snow-covered roads, reduced visibility, and challenging weather conditions.

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

### 4. Data Description

- **LiDAR data** collected along Norwegian highways  
- **Georeferenced snow pole locations** obtained from infrastructure-level mapping  
- **Test site**: E39 – Hemnekjølen, Norway  

⚠️ **Raw LiDAR datasets and large artifacts are not included in this repository** due to size limitations and data-sharing constraints.

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

- **Bavirisetti, D. P., Berget, G. E., Kiss, G. H., Arnesen, P., Seter, H., & Lindseth, F. (2025). Vehicle Localization Framework Using Georeferenced Snow Poles and LiDAR in GNSS-Limited Environments Under Nordic Conditions. IEEE Transactions on Intelligent Transportation Systems.**

- **Bavirisetti, D. P., Kiss, G. H., Arnesen, P., Seter, H., Tabassum, S., & Lindseth, F. (2025). SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions. Data in Brief, 59, 111403.**  
 

---

### Conference Papers

- **Bavirisetti, D. P., Kiss, G. H., & Lindseth, F. (2024, July). A pole detection and geospatial localization framework using liDAR-GNSS data fusion. In 2024 27th International Conference on Information Fusion (FUSION) (pp. 1-8). IEEE.**
- **Bavirisetti, D. P. (2025). Enhancing vehicle navigation in GNSS-limited environments with georeferenced snow poles. In 2025 IEEE Symposium Series on Computational Intelligence (SSCI), Trondheim, Norway, 17-20 March 2025. IEEE. (Poster Presentation)**


### 7. Project Context & Funding

This research was conducted as part of the project:

“Machine Sensible Infrastructure under Nordic Conditions”
Project Number: 333875


### 8. Contact

Durga Prasad Bavirisetti
Associate Professor – AI & Computer Vision
University of Gävle, Sweden

For questions, collaborations, or extensions of this work, feel free to get in touch.
