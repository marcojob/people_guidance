import pathlib
from typing import Optional, List

import cv2


class VoTestDataset:
    def __init__(self, data_dir: Optional[pathlib.Path] = None):
        self.data_dir = data_dir if data_dir is not None else \
            pathlib.Path(__file__).parent.parent / "data" / "shaky_corridor" / "frames"
        self.filenames = [child for child in self.data_dir.iterdir() if child.is_file() and child.suffix == ".png"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.filenames[item]
        img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE

        _, position, rotation = self.decode_filename(filename.stem)

        return img, position, rotation

    @staticmethod
    def decode_filename(filename: str):
        parts = filename.split("_")

        position = tuple(float(dim) for dim in parts[2].replace("(", "").replace(")", "").replace(",", ".").split("!"))
        rotation = tuple(float(dim) for dim in parts[3].replace("(", "").replace(")", "").replace(",", ".").split("!"))
        return int(parts[1]), position, rotation


if __name__ == '__main__':
    dset = VoTestDataset()

    for (image, absolute_position, absolute_rotation) in dset:
        print(image.shape)