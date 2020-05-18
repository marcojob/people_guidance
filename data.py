import argparse

from people_guidance.tools.convert_eth3d_slam_datasets import download_and_convert_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    args = parser.parse_args()
    download_and_convert_dataset(args.dataset_name)