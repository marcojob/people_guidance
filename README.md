# EGS: Computationally Efficient Guidance System for Visually Impaired Individuals

<center>
<img src="https://drive.google.com/uc?export=view&id=1E7igx7hbhx5tFuHK6OEQ9EvgjPHCU3xs" alt="" width="100%">
</center>

## System Overview
EGS is a modular system. In its current state the following modules are used by default:
- [Drivers Module](/people_guidance/modules/drivers_module): Interfaces with the hardware
- [Feature Tracking Module](/people_guidance/modules/feature_tracking_module): Visual Odometry
- [Position Estimation Module](/people_guidance/modules/position_module): IMU integration and homography scaling
- [Reprojection Module](/people_guidance/modules/reprojection_module): 3D reconstruction and collision probability

Moreover, you can enable the optional [visualization module](/people_guidance/modules/visualization_module).

The modules are connected using queues and managed by the [Pipeline](/people_guidance/pipeline.py) class. Logfiles are saved in the logs directory in the project root. There are logfiles for each module and the pipeline itself.

## Installation
1. Clone this repo.
2. Install the requirements using `pip install -r requirements.txt`. If you would like to use SIFT features you need to build opencv-python from sources with the enable-nonfree flag set (as described [here](https://github.com/skvark/opencv-python/issues/126)). If you do not need SIFT features you can install opencv-python by using the requirements file which downloads a prebuilt wheel directly from [PyPi](https://pypi.org/).
3. Download one or multiple datasets from XXXXXXX.

## Usage
### Playback
You can evaluate the pipeline on a pre-recorded dataset with visualization using:
``` shell
python main.py --playback /path/to/your/dataset --visualize
```
### Deployment
If you have replicated our hardware you can deploy the pipeline in real-time. After installing all requirements simply run:
``` shell
python main.py --deploy
```


## Hardware
<center>
<img src="https://drive.google.com/uc?export=view&id=1E4m8Vy020IXHZr6dpMHGo7GPaHATFAa5" alt="alt text" width="50%">
</center>
