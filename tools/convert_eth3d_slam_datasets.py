from pathlib import Path
from PIL import Image
from tqdm import tqdm

from tools.download_eth3d_slam_datasets import download_dataset_for_conversion

ROOT_DIR = Path(__file__).parent.resolve()


def convert_s_to_ms(timestamp):
    return int(timestamp * 1000)


if __name__ == '__main__':
    dataset_name = "cables_2"

    dataset_dir = ROOT_DIR / dataset_name

    if not dataset_dir.is_dir():
        print("Downloading Dataset!")
        download_dataset_for_conversion(dataset_name)

    converted_dataset_dir = ROOT_DIR / f"converted_eth_slam_{dataset_name}"
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
        print("Converting rgb images to jpg")
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
                           f"accel_z: {frame[6]}, gyro_x: {frame[1]}, gyro_y: {frame[2]}, gyro_z: {frame[3]}\n"
                    out_fp.write(line)
