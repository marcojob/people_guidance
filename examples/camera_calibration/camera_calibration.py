import re
import cv2
import numpy as np

from pathlib import Path

DEFAULT_DATASET = Path("../../data/indoor_dataset_7")

# Defining the dimensions of checkerboard
CHECKERBOARD = (9,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


def process_image(img_filename):
    img_file_path = DEFAULT_DATASET / "imgs" / img_filename

    with open(img_file_path, 'rb') as fp:
        img_data_file = fp.read()

    img = cv2.imdecode(np.frombuffer(img_data_file, dtype=np.int8), flags=cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    print(f"{img_filename}: {ret}")

    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        gray = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)

    cv2.imshow("img", gray)
    cv2.waitKey(500)

if __name__ == '__main__':
    # Get all the imgs
    img_data = (DEFAULT_DATASET / "img_data.txt").open(mode='r')
    img_str_lines = img_data.readlines()
    img_list = list()

    for line in img_str_lines:
        out = re.search(r'([0-9]*): ([0-9]*)', line)
        if out:
            img_id = int(out.group(1))
            if img_id > 0 and img_id < 1000:
                img_list.append(f"img_{img_id:04d}.jpg")

    for cnt, file in enumerate(img_list):
        if cnt%5 == 0:
            process_image(file)
            print(f"Progress: {cnt} / {len(img_list)}\r", end='')

    img_shape = (1640, 1232)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape,None,None)
    print(mtx)