from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DEFAULT_DATASET = Path("../../data/outdoor_dataset_19")
RESIZED_IMAGE = (820, 616)
INTRINSIC_MATRIX = np.array([[644.90127548, 0.0, 406.99519054], [0.0, 644.99811417, 307.06244081], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array([[0.19956839, -0.49217089, -0.00235192, -0.00051292, 0.28251577]])


class LucasKanadeTracker:
    #https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py
    def __init__(self, visualize = True):
        self.track_len = 10
        self.detect_interval = 10
        self.tracks = []
        self.frame_idx = 0
        self.visualize = visualize
        self.prev_frame = None

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def track(self, frame0, frame1):
        self.track_frame(frame0)
        self.track_frame(frame1)

        return np.array([tr[-2] for tr in self.tracks]), np.array([tr[-1] for tr in self.tracks])

    def track_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.tracks) > 0:
            img0, img1 = self.prev_frame, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                if self.visualize:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            self.tracks = new_tracks

            if self.visualize:
                cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_frame = frame_gray


def load_img_from_file(fpath):
    with open(DEFAULT_DATASET / "imgs" / fpath, 'rb') as fp:
        data = fp.read()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.int8), flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, RESIZED_IMAGE)
    return img


def project3dto2d(points3d, homography):
    points2d, _ = cv2.projectPoints(points3d,
                                       homography[:, :3],
                                       homography[:, 3:],
                                       INTRINSIC_MATRIX,
                                       np.array([]))
    points2d = points2d.reshape((points2d.shape[0], 2))
    return points2d


def reject_outliers(data, m=6):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def get_homography(E, kp0, kp1):
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # this gives us four canidate homgographies [R1,t], [R1,−t], [R2,t], [R2,−t].
    # we create only two canidates. if the best absolute score is negative, we now we must use
    # the negative translation vector

    candidates = (np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)), np.hstack((R2, -t)))
    scores = []
    masks = []
    forward_axis = 2
    for candidate_homo in candidates:
        points3d = cv2.triangulatePoints(np.eye(3, 4), candidate_homo, kp0.T, kp1.T)
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)
        scores.append((points3d[:, 0, forward_axis] > 0.).sum())
        masks.append(points3d[:, 0, forward_axis] > 0.)

    idx = int(np.argmax(scores))

    points3d = cv2.triangulatePoints(np.eye(3, 4), candidates[idx], kp0.T, kp1.T)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T)
    mean = ((points3d[:, 0, 2] > 0.) * 2.).mean()
    if mean < 1.9:
        for axis in (0, 1, 2):
            print(mean, points3d[:, 0, axis].min(), "max", points3d[:, 0, axis].max())

    return candidates[idx], masks[idx]



def lucas_kandade_approach():
    feature_tracker = LucasKanadeTracker()
    for iidx in range(100, 550, 1):
        img0 = load_img_from_file(f"img_{iidx:04d}.jpg")
        img1 = load_img_from_file(f"img_{iidx + 1:04d}.jpg")

        kp0, kp1 = feature_tracker.track(img0, img1)

        E, mask = cv2.findEssentialMat(kp0, kp1, INTRINSIC_MATRIX, method=cv2.RANSAC,
                                       threshold=0.99, prob=0.999)

        # homography, mask = get_homography(E, kp0, kp1)
        retval, R, t, mask, points3d = cv2.recoverPose(E, kp0, kp1, INTRINSIC_MATRIX, distanceThresh=100)

        points3d = cv2.convertPointsFromHomogeneous(points3d.T)

        point_vectors: np.array = points3d.reshape((points3d.shape[0], 3))
        distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)
        distances -= distances.min()
        distances /= distances.max()

        homography = np.hstack((R, t))
        points2d = project3dto2d(points3d, homography)

        for i in range(points2d.shape[0]):
            c = int(distances[i] * 255.0)
            if c > 255:
                c = 255
            if c < 0:
                c = 0
            cv2.circle(img0, tuple((int(points2d[i, 0]), int(points2d[i, 1 ]))), 4, (0, c, 255 - c), -1)

        if (iidx % 10 == 0):
            cv2.imshow("img", np.hstack((img0, img1)))
            cv2.waitKey(1)



if __name__ == '__main__':
    lucas_kandade_approach()