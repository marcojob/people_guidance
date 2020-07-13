# EGS: Computationally Efficient Guidance System for Visually Impaired Individuals

<center>
<img src="/hardware/demo.png " alt="" width="100%">
</center>

## System Overview
EGS is a guidance system aimed at visually impaired individuals. It is meant to compliment the capabilities of the traditional white cane. It can provide earlier warnings, which allow the user to navigate more safely in potentially challenging situations. Moreover, our system can offer warnings for obstacles above waist height, which are difficult to detect using a white cane. **However, it is important to note that this system is still in early development and should only be tested in controlled environments.**
The system is designed to be highly modular. In its current state the following modules are used by default:
- [Drivers Module](/people_guidance/modules/drivers_module): Interfaces with the hardware
- [Feature Tracking Module](/people_guidance/modules/feature_tracking_module): Visual Odometry
- [Position Estimation Module](/people_guidance/modules/position_module): IMU integration and homography scaling
- [Reprojection Module](/people_guidance/modules/reprojection_module): 3D reconstruction and collision probability

Moreover, you can enable the optional [visualization module](/people_guidance/modules/visualization_module).

The modules are connected using queues and managed by the [Pipeline](/people_guidance/pipeline.py) class. Logfiles are saved in the logs directory in the project root. There are logfiles for each module and the pipeline itself.

## Installation
This repo requires Python 3.7 or higher and was tested on Windows 10, Ubuntu 18.04 and Raspbian Stretch.
1. Clone this repo.
2. Install the requirements using `pip install -r requirements.txt`. If you would like to use SIFT features you need to build opencv-python from sources with the enable-nonfree flag set (as described [here](https://github.com/skvark/opencv-python/issues/126)). If you do not need SIFT features you can install opencv-python by using the requirements file which downloads a prebuilt wheel directly from [PyPi](https://pypi.org/).
3. Download one or multiple datasets (for link contact one of the contributors) and unzip them.

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

Our system is based on readily available and low cost components. Additionally, to the materials listed below you will need a battery bank to power the Raspberry Pi and some cables to connect everything together. You can 3D print your own chest mount using our [stl file](/hardware/baseplate.stl). We recommed a layer height of 0.3mm and an infill of 20%.
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
