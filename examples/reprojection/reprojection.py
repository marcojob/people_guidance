from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DEFAULT_DATASET = Path("../../data/outdoor_dataset_08")
RESIZED_IMAGE = (820, 616)
FAST_THRESHOLD = 40
INTRINSIC_MATRIX = np.array([[1.29168322e+03, 0.0, 8.10433936e+02], [0.0, 1.29299333e+03, 6.15008893e+02], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array([[ 0.1952957 , -0.48124548, -0.00223218, -0.00106617,  0.2668875]])
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
shi_tomasi_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

# Config
method = 'FAST'
max_num_features = 5000
USE_ONLY_E = False

def main(img_pair):
    img_data = list()
    for i in img_pair:
        img_file_path = DEFAULT_DATASET / "imgs" / i

        with open(img_file_path, 'rb') as fp:
            img_data.append(fp.read())

    # Preprocess image
    img1 = cv2.imdecode(np.frombuffer(img_data[0], dtype=np.int8), flags=cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img_data[1], dtype=np.int8), flags=cv2.IMREAD_COLOR)

    # Copy images
    img1_rgb = img1.copy()
    img2_rgb = img2.copy()
    #img1_rgb = cv2.undistort(img1_rgb, INTRINSIC_MATRIX, DISTORTION_COEFFS)
    #img2_rgb = cv2.undistort(img2_rgb, INTRINSIC_MATRIX, DISTORTION_COEFFS)

    # Grayscale imgs
    img1, img2 = grayscale_imgs(img1, img2)

    # Resize images
    img1, img2 = resize_imgs(img1, img2)
    img1_rgb, img2_rgb = resize_imgs(img1_rgb, img2_rgb)

    #Detect keypoints
    detector = get_detector()
    prev_kps = detect_keypoints(method, detector, img1)

    # Undistort points
    prev_kps = cv2.undistortPoints(prev_kps, INTRINSIC_MATRIX, DISTORTION_COEFFS, R=None, P=INTRINSIC_MATRIX).reshape(-1, 2)

    # Lukas Kanade optical flow
    prev_kps, cur_kps, diff = KLT_featureTracking(img1, img2, prev_kps)

    # Fit essential matrix
    if USE_ONLY_E:
        E, mask = cv2.findEssentialMat(prev_kps, cur_kps, INTRINSIC_MATRIX, cv2.RANSAC, 0.999, 1, None)
        _, rot, tran, _ = cv2.recoverPose(E, prev_kps, cur_kps, INTRINSIC_MATRIX, mask)
    else:
        # Assume planar motion model
        H, mask = cv2.findHomography(prev_kps, cur_kps, cv2.RANSAC, 1.0)

        # Find points that belong to that model
        prev_kps_hom = prev_kps[mask.ravel().astype(bool)]
        cur_kps_hom = cur_kps[mask.ravel().astype(bool)]

        # Find essential matrix of this model
        E, mask = cv2.findEssentialMat(prev_kps_hom, cur_kps_hom, INTRINSIC_MATRIX, cv2.RANSAC, 0.999, 1, mask)

        # Recover pose
        _, rot, tran, _ = cv2.recoverPose(E, prev_kps_hom, cur_kps_hom, INTRINSIC_MATRIX, mask)

    # Stack to homography
    homography = np.hstack((rot, tran))

    # Projection matrices
    P0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
    P1 = np.dot(INTRINSIC_MATRIX, homography)

    # Triangulate with all the keypoints
    points3d = cv2.triangulatePoints(P0, P1, np.transpose(prev_kps), np.transpose(cur_kps))
    points3d = cv2.convertPointsFromHomogeneous(points3d.T)

    # Show images
    img1_rgb = visualize_matches(img1_rgb, prev_kps, cur_kps)
    img2_rgb = visualize_matches(img2_rgb, cur_kps, prev_kps)

    # Stack images
    img_vis = np.hstack((img1_rgb, img2_rgb))

    plot_3d(points3d, img_vis)

def detect_keypoints(method, detector, img):
    if method == 'SHI-TOMASI':
        keypoints = cv2.goodFeaturesToTrack(img, **shi_tomasi_params)
    elif method == 'ORB':
        keypoints = detector.detect(img, None)
        keypoints, descriptors = detector.compute(img, keypoints)
    elif method == 'FAST':
        keypoints = detector.detect(img, None)
    elif method == 'REGULAR_GRID':
        keypoints = regular_grid_detector(img)
    else:
        keypoints, descriptors = detector.detectAndCompute(img, None)

    if not method == 'SHI-TOMASI':
        keypoints = np.array([x.pt for x in keypoints], dtype=np.float32).reshape((-1, 2))

    return keypoints

def resize_imgs(img1, img2):
    img1 = cv2.resize(img1, RESIZED_IMAGE)
    img2 = cv2.resize(img2, RESIZED_IMAGE)
    return img1, img2

def grayscale_imgs(img1, img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return img1, img2

def visualize_matches(img, prev_kps, cur_kps):
    THICKNESS = 1
    GREEN = (0, 255, 0)
    for i in range(min(cur_kps.shape[0], prev_kps.shape[0])):
        img = cv2.arrowedLine(img, tuple(cur_kps[i]), tuple(prev_kps[i]), GREEN, THICKNESS)
        #img = cv2.circle(img, tuple(cur_kps[i]), 1, (0,0,255), -1)
        #img = cv2.circle(img, tuple(prev_kps[i]), 1, (255, 0, 0), -1)

    return img

def get_detector():
    if method == 'FAST':
        detector = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_num_features)
    elif method == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create(max_num_features)
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(max_num_features)
    elif method == 'SHI-TOMASI':
        detector = None
    elif method == 'REGULAR_GRID':
        regular_grid_max_pts = max_num_features
    else:
        print("Unknown method")
    return detector

def plot_3d(points3d, img_vis):
    FIGSIZE = (15,12)
    DPI = 100
    PLOT_LIM = 0.5
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    p4 = fig.add_subplot(2, 1, 1)
    p1 = fig.add_subplot(2, 3, 4)
    p2 = fig.add_subplot(2, 3, 5)
    p3 = fig.add_subplot(2, 3, 6, projection='3d')

    x = list()
    y = list()
    z = list()
    d = list()
    for i in range(points3d.shape[0]):
        x.append(points3d[i][0][2])
        y.append(-points3d[i][0][0])
        z.append(-points3d[i][0][1])
        d.append(np.sqrt(points3d[i][0][0]**2 + points3d[i][0][1]**2 + points3d[i][0][2]**2))

    p1.scatter(y, z, c=x)
    p1.set_title("front")

    p2.scatter(x, y, c=z)
    p2.set_title("top")

    p3.scatter(x, y, z, c=x)

    p4.imshow(img_vis[...,::-1])

    plt.show()


def KLT_featureTracking(prev_img, cur_img, prev_fts):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    """

    # Feature Correspondence with Backtracking Check
    MIN_MATCHING_DIFF = 1
    kp2, status, error = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, prev_fts, None, **lk_params)
    kp1, status, error = cv2.calcOpticalFlowPyrLK(cur_img, prev_img, kp2, None, **lk_params)

    d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
    good = d < MIN_MATCHING_DIFF

    # Error Management
    if len(d) == 0:
        print('No point correspondance.')
    elif list(good).count(True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
        print('Few point correspondances')
        return kp1, kp2, MIN_MATCHING_DIFF

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    # Verify if the point correspondence points are in the same pixel coordinates
    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    # The mean of the differences is used to determine the amount of distance between the pixels
    diff_mean = np.mean(d)

    return n_kp1, n_kp2, diff_mean


if __name__ == '__main__':
    img_id = 185
    incr = 1
    img_pair = [f"img_{img_id:04d}.jpg", f"img_{img_id+incr:04d}.jpg"]
    main(img_pair)