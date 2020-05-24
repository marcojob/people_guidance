from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DEFAULT_DATASET = Path("../../data/outdoor_dataset_19")
RESIZED_IMAGE = (820, 616)
FAST_THRESHOLD = 40
INTRINSIC_MATRIX = np.array([[644.90127548, 0.0, 406.99519054], [0.0, 644.99811417, 307.06244081], [0.0, 0.0, 1.0]])

DISTORTION_COEFFS = np.array([[0.19956839, -0.49217089, -0.00235192, -0.00051292, 0.28251577]])
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
shi_tomasi_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

# Config
method = 'FAST'
max_num_features = 5000
USE_ONLY_E = False


def main():
    for iidx in range(200, 700, 10):
        fname1 = f"img_{iidx:04d}.jpg"
        fname2 = f"img_{iidx + 1:04d}.jpg"

        with open(DEFAULT_DATASET / "imgs" / fname1, 'rb') as fp:
            img1data = fp.read()

        with open(DEFAULT_DATASET / "imgs" / fname2, 'rb') as fp:
            img2data = fp.read()

        # Preprocess image
        img1 = cv2.imdecode(np.frombuffer(img1data, dtype=np.int8), flags=cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2data, dtype=np.int8), flags=cv2.IMREAD_COLOR)
        # Copy images
        img1_rgb = img1.copy()
        img2_rgb = img2.copy()
        # Grayscale imgs
        img1, img2 = grayscale_imgs(img1, img2)
        # Resize images
        img1, img2 = resize_imgs(img1, img2)
        img1_rgb, img2_rgb = resize_imgs(img1_rgb, img2_rgb)

        # Detect keypoints
        detector = get_detector()

        prev_kps = detect_keypoints(method, detector, img1)
        prev_kps = cv2.undistortPoints(prev_kps, INTRINSIC_MATRIX, DISTORTION_COEFFS,
                                       R=None, P=INTRINSIC_MATRIX).reshape(-1, 2)
        prev_kps, cur_kps, diff = KLT_featureTracking(img1, img2, prev_kps)

        E, inlier_mask = cv2.findEssentialMat(prev_kps, cur_kps, cameraMatrix=INTRINSIC_MATRIX, method=cv2.RANSAC,
                                              threshold=0.2, prob=0.999)

        homography, chirality_mask = get_homography(E, prev_kps, cur_kps)

        cur_kps = cur_kps[chirality_mask.ravel().astype(bool)]
        prev_kps = prev_kps[chirality_mask.ravel().astype(bool)]

        PM0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
        PM1 = np.dot(INTRINSIC_MATRIX, homography)

        points3d = cv2.triangulatePoints(PM0, PM1, prev_kps.T, cur_kps.T)
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)

        img1_rgb = visualize_matches(img1_rgb, prev_kps, cur_kps)
        img2_rgb = visualize_matches(img2_rgb, prev_kps, cur_kps)
        img1_rgb = visualize_reprojection(points3d, homography, img1_rgb)
        img2_rgb = visualize_reprojection(points3d, homography, img2_rgb)

        """
        # Stack images
        img_vis = np.hstack((img1_rgb, img2_rgb))
        cv2.imshow("img", img_vis)
        cv2.waitKey(1)
        plt.scatter(points3d[:, 0, 0], points3d[:, 0, 2])
        plt.scatter(points3d[:, 0, 1], points3d[:, 0, 2])
        plt.show()
        """


def get_homography(E, prev_kps, cur_kps):
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # this gives us four canidate homgographies [R1,t], [R1,−t], [R2,t], [R2,−t].
    # we create only two canidates. if the best absolute score is negative, we now we must use
    # the negative translation vector

    candidates = (np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)), np.hstack((R2, -t)))
    scores = []
    masks = []
    forward_axis = 2
    for candidate_homo in candidates:
        points3d = cv2.triangulatePoints(np.eye(3, 4), candidate_homo, prev_kps.T, cur_kps.T)
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)
        scores.append((points3d[:, 0, forward_axis] > 0.).sum())
        masks.append(points3d[:, 0, forward_axis] > 0.)

    idx = int(np.argmax(scores))

    points3d = cv2.triangulatePoints(np.eye(3, 4), candidates[idx], prev_kps.T, cur_kps.T)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T)
    mean = ((points3d[:, 0, 2] > 0.) * 2.).mean()
    if mean < 1.9:
        print(mean, scores)
        for axis in (0, 1, 2):
            print(points3d[:, 0, axis].min(), "max", points3d[:, 0, axis].max())

    return candidates[idx], masks[idx]


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
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2


def visualize_matches(img, prev_kps, cur_kps):
    THICKNESS = 1
    GREEN = (0, 255, 0)
    for i in range(min(cur_kps.shape[0], prev_kps.shape[0])):
        img = cv2.arrowedLine(img, tuple(cur_kps[i]), tuple(prev_kps[i]), GREEN, THICKNESS)
        # img = cv2.circle(img, tuple(kp1[i]), 1, (0,0,255), -1)
        # img = cv2.circle(img, tuple(kp0[i]), 1, (255, 0, 0), -1)

    return img


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def visualize_reprojection(points3d, homography, img):
    point_vectors: np.array = points3d.reshape((points3d.shape[0], 3))
    point_vectors = np.abs(point_vectors)
    distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)
    distances = np.abs(distances)
    distances -= reject_outliers(distances).min()
    distances /= reject_outliers(distances).max()

    points2d = project3dto2d(homography, points3d)
    for i in range(points2d.shape[0]):
        c = int(distances[i] * 255.0)
        if c > 255:
            c = 255
        if c < 0:
            c = 0
        point = points2d[i, 0, :]

        img = cv2.circle(img, tuple((int(point[0]), int(point[1]))), 3, (0, c, 255 - c), -1)
    return img


def project3dto2d(homography: np.array, points3d: np.array):
    rot_vec = cv2.Rodrigues(homography[:, :3])[0]
    trans_vec = homography[:, 3:]
    try:
        points2d = cv2.projectPoints(points3d, rot_vec, trans_vec, INTRINSIC_MATRIX, distCoeffs=None)[0]
    except Exception as e:
        points2d = np.array([])
    return points2d


def project3dTo2dArray(points3d, K, rotation, translation):
    imagePoints, _ = cv2.projectPoints(points3d,
                                       rotation,
                                       translation,
                                       K,
                                       np.array([]))
    p2d = imagePoints.reshape((imagePoints.shape[0], 2))
    return p2d


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
    FIGSIZE = (15, 12)
    DPI = 100
    PLOT_LIM = 0.5
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    preview = plt.figure(figsize=FIGSIZE, dpi=DPI)

    p1 = fig.add_subplot(1, 3, 1)
    p2 = fig.add_subplot(1, 3, 2)
    p3 = fig.add_subplot(1, 3, 3, projection='3d')
    p4 = preview.add_subplot(1, 1, 1)

    x = list()
    y = list()
    z = list()
    for i in range(points3d.shape[0]):
        x.append(points3d[i][0][2])
        y.append(-points3d[i][0][0])
        z.append(-points3d[i][0][1])

    p1.scatter(y, z, c=x)
    p1.set_title("front")

    p2.scatter(x, y, c=z)
    p2.set_title("top")

    p3.scatter(x, y, z)

    p4.imshow(img_vis[..., ::-1])

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
    elif list(good).count(
            True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
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
    img_id = 10
    incr = 1
    main()
