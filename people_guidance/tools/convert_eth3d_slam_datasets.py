from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np

from .download_eth3d_slam_datasets import download_dataset_for_conversion
from ..utils import ROOT_DATA_DIR

def convert_s_to_ms(timestamp):
    return int(timestamp * 1000)


def convert_rad_to_degree(rads):
    return rads * (180 / np.pi)


def download_and_convert_dataset(dataset_name):

    dataset_dir = ROOT_DATA_DIR / dataset_name

    if not dataset_dir.is_dir():
        print("Downloading Dataset!")
        download_dataset_for_conversion(dataset_name, ROOT_DATA_DIR)

    converted_dataset_dir = ROOT_DATA_DIR / f"converted_eth_slam_{dataset_name}"
    converted_dataset_dir.mkdir(exist_ok=True)
    converted_dataset_img_dir = converted_dataset_dir / "imgs"
    converted_dataset_img_dir.mkdir(exist_ok=True)

    rgb_paths = []

    # parse paths to rgb images and their respective timestamps
    with open(str(dataset_dir / "rgb.txt"), "r") as fp:
        for line in fp:
            line = line.replace("\n", "")
            timestamp, fpath = line.split(" ")
            rgb_paths.append((float(timestamp), fpath))

    rgb_paths.sort(key=lambda tup: tup[0])  # sort by timestamps (just to be sure)

    fname_mapping = {}
    # convert the images into jpg and save them into the converted dataset folder
    with open(str(converted_dataset_dir / "img_data.txt"), "w") as out_fp:
        print("Converting png images to jpg")
        for i, (timestamp, fpath) in tqdm(enumerate(rgb_paths), total=len(rgb_paths)):
            im = Image.open(dataset_dir / fpath)
            # save the filename mapping so that we can find the corresponding depth images later
            fname_mapping[fpath] = f"img_{str(i).zfill(4)}.jpg"
            im.save(converted_dataset_img_dir / fname_mapping[fpath])
            out_fp.write(f"{i}: {convert_s_to_ms(timestamp)}\n")

    # parse the imu data and write it to the new file in the new format simultaneously
    imu_file: Path = dataset_dir / "imu.txt"
    with open(str(converted_dataset_dir / "imu_data.txt"), "w") as out_fp:
        with open(str(imu_file), "r") as fp:
            for line in fp:
                if line[0] != "#":
                    line = line.replace("\n", "")
                    frame = [float(elem) for elem in line.split(" ")]
                    line = f"{convert_s_to_ms(frame[0])}: accel_x: {frame[4]}, accel_y: {frame[5]}, " \
                           f"accel_z: {frame[6]}, gyro_x: {convert_rad_to_degree(frame[1])}, " \
                           f"gyro_y: {convert_rad_to_degree(frame[2])}, gyro_z: {convert_rad_to_degree(frame[3])}\n"
                    out_fp.write(line)

    # copy the depth images
    shutil.copytree(dataset_dir / "depth", converted_dataset_dir / "depth")
    shutil.copy(dataset_dir / "depth.txt", converted_dataset_dir / "depth.txt")
    shutil.copy(dataset_dir / "calibration.txt", converted_dataset_dir / "calibration.txt")

    print("Deleting raw dataset...")
    shutil.rmtree(dataset_dir)
    print("Finished...")

