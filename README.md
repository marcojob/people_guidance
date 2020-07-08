# EGS: Computationally Efficient Guidance System for Visually Impaired Individuals

<center>
<img src="/hardware/demo.png " alt="" width="100%">
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
3. Download one or multiple [datasets](https://drive.google.com/drive/folders/1wUy62vMHkyyuOigcdbWZYMblkrIDHQn1?usp=sharing) and unzip them.

## Usage
### Playback
You can evaluate the pipeline on a pre-recorded dataset with visualization using:
``` shell
python main.py --replay /path/to/your/dataset --visualize
```
### Deployment
If you have replicated our hardware you can deploy the pipeline in real-time. After installing all requirements simply run:
``` shell
python main.py --deploy
```


## Hardware
<center>
<img src="/hardware/mount.JPG" alt="alt text" width="50%">
</center>

Our system is based on readily available and low cost components. Additionally to the materials listed below you will need a battery bank to power the Raspberry Pi and some cables to connect everything together. You can 3D print your own chest mount using our [stl file](/hardware/baseplate.stl).
<br />

| Component                     | Price |
|-------------------------------|-------|
| Raspberry Pi 3 Model B        | ~35$  |
| MPU-6050 IMU                  | ~3$   |
| Raspberry Pi Camera Module V2 | ~25$  |


## Workload
The workload was split evenly across our team throughout the whole project. However, each module was assigned a product owner who was responsible for any major design decision and approved improvements to their modules made by the other contributors. The module owners were assigned as follows:
- Hardware and Drivers: Marco Job
- Inertial Position: Théophile Messin-Roizard
- Visual Odometry: Adrian Schneebeli
- Software Architecture and 3D Reconstruction: Lorenz Hetzel
