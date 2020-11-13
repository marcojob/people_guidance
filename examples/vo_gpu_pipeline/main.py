import cv2
import time
import numpy as np

SOURCE = 'KITTI'
DATASET_MOV = 'datasets/test.MOV'
DATASET_KITTI = 'datasets/kitti00/kitti/00/image_0/'
RESIZED_FRAME_SIZE_MOV = (960, 540)
RESISED_FRAME_SIZE_KITTI = None  # (1241, 376)
RESIZED_FRAME_SIZE = None
DETECTOR = 'SURF'
MAX_NUM_FEATURES = 1000
MIN_NUM_FEATURES = 100
USE_CLAHE = False
MIN_MATCHING_DIFF = 1


class VisualOdometry:
    def __init__(self, detector=DETECTOR):
        # Save detector type
        self.DETECTOR = DETECTOR

        # Initialize detector
        if detector == 'FAST':
            self.detector = cv2.cuda_FastFeatureDetector.create(
                threshold=75, nonmaxSuppression=True)
        elif detector == 'SIFT':
            self.detector = cv2.cuda.xfeatures2d.SIFT_create(MAX_NUM_FEATURES)
        elif detector == 'SURF':
            self.detector = cv2.cuda.SURF_CUDA_create(300, _nOctaveLayers=2)
        elif detector == 'ORB':
            self.detector = cv2.cuda_ORB.create(nfeatures=MAX_NUM_FEATURES)
        elif detector == 'SHI-TOMASI':
            feature_params = dict( maxCorners = 1000,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )
            self.detector = cv2.cuda.createGoodFeaturesToTrackDetector(cv2.CV_8UC1, feature_params['maxCorners'], \
                                           feature_params['qualityLevel'], feature_params['minDistance'], \
                                           feature_params['blockSize'])

        # Initialize clahe filter
        self.clahe = cv2.cuda.createCLAHE(clipLimit=5.0)

        # LK
        lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,)
                  # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create(**lk_params)

        # Frames
        self.cur_c_frame = None
        self.pre_c_frame = None

        self.cur_g_frame = None
        self.pre_g_frame = None

        # Features
        self.cur_c_fts = None  # Current CPU features
        self.pre_c_fts = None  # Previous CPU features

        self.cur_g_fts = None  # Current GPU features
        self.pre_g_fts = None  # Previous GPU features

    def init_dataset(self):
        if SOURCE == 'MOV':
            # Init video capture with video
            self.cap = cv2.VideoCapture(DATASET_MOV)

            # Resize size
            RESIZED_FRAME_SIZE = RESIZED_FRAME_SIZE_MOV

        elif SOURCE == 'KITTI':
            self.img_cnt = 0

            # Resize size
            RESIZED_FRAME_SIZE = RESISED_FRAME_SIZE_KITTI

    def get_frame(self):
        if SOURCE == 'MOV':
            # Read from capture
            ret, frame = self.cap.read()
        elif SOURCE == 'KITTI':
            # Read img
            ret, frame = 1, cv2.imread(
                DATASET_KITTI + "{:06d}.png".format(self.img_cnt))

            # Increase counter
            self.img_cnt += 1

        return ret, frame

    def init(self):
        # Init datasets
        self.init_dataset()

        # Handle first frame
        ret, frame = self.get_frame()

        # Process frame
        self.process_first_frame(frame)

        while True:
            # Read the frames
            ret, frame = self.get_frame()
            if ret:
                # Main frame processing
                self.process_frame(frame)

                # Create copy for plotting
                frame = self.cur_c_frame

                # Draw matches
                frame = self.draw_fts(frame, self.cur_c_fts)

                # Draw framerate
                frame = self.draw_framerate(frame, self.framerate)

                # Show img
                cv2.imshow("Video", frame)

            if cv2.waitKey(1) == 27:
                break

        # Release the capture
        cap.release()

        # Destroy all windows
        cv2.destroyAllWindows()

    def draw_fts(self, frame, fts):
        size = 3
        col = (255, 0, 0)
        th = 1
        for f in fts:
            x, y = int(f[0]), int(f[1])
            cv2.circle(frame, (x, y), size, col, thickness=th)
        return frame

    def draw_framerate(self, frame, framerate):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (880, 15)
        fontScale = 0.5
        col = (255, 0, 0)
        th = 2
        return cv2.putText(frame, "FPS: "+str(framerate), org, font,
                           fontScale, col, th, cv2.LINE_AA)

    def process_frame(self, frame):
        # Start timer
        process_frame_start = time.monotonic()

        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        if not RESIZED_FRAME_SIZE is None:
            self.gf = cv2.cuda.resize(self.gf, RESIZED_FRAME_SIZE)

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Update CPU frame
        self.pre_c_frame = self.cur_c_frame
        self.cur_c_frame = self.gf.download()

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update prev and curr img
        self.pre_g_frame = self.cur_g_frame
        self.cur_g_frame = self.gf

        # Detect new features if we don't have enough
        if len(self.pre_c_fts) < MIN_NUM_FEATURES or True:
            self.cur_g_fts = self.detect_new_features(self.cur_g_frame)
            self.pre_g_fts = self.detect_new_features(self.pre_g_frame)

            # Convert keypoints to CPU
            self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
            self.pre_c_fts = self.convert_fts_gpu_to_cpu(self.pre_g_fts)

            # Fix some shit in OpenCV
            tmp = cv2.cuda_GpuMat()
            tmp_re = self.pre_c_fts.reshape((1, -1, 2))
            tmp.upload(tmp_re)
            self.pre_g_fts = tmp

        # Sparse OF
        self.pre_c_fts, self.cur_c_fts, diff = self.KLT_featureTracking(self.pre_g_frame, self.cur_g_frame, self.pre_g_fts)

        #Reupload to GPU
        self.cur_g_fts = cv2.cuda_GpuMat()
        self.cur_g_fts.upload(self.cur_c_fts)

        self.pre_g_fts = cv2.cuda_GpuMat()
        self.pre_g_fts = cv2.cuda_GpuMat()

        # Download frame
        self.d_frame = self.gf.download()

        # End timer and compute framerate
        self.framerate = round(1.0 / (time.monotonic() - process_frame_start))

    def KLT_featureTracking(self, prev_img, cur_img, prev_fts):
        """Feature tracking using the Kanade-Lucas-Tomasi tracker.
        """

        # Feature Correspondence with Backtracking Check
        kp2_g, status, error = self.lk.calc(prev_img, cur_img, prev_fts, None)
        kp1_g, status, error = self.lk.calc(cur_img, prev_img, kp2_g, None)

        # Get CPU kp
        kp2 = kp2_g.download()[0]
        kp1 = kp1_g.download()[0]

        # import pdb; pdb.set_trace()

        # Verify the absolute difference between feature points
        d = abs(self.pre_c_fts - kp1).reshape(-1, 2).max(-1)
        good = d < MIN_MATCHING_DIFF

        # Error Management
        if len(d) == 0:
            print('Error: No point correspondance.')
        # If less than 5 good points, it uses the features obtain without the backtracking check
        elif list(good).count(True) <= 5:
            print('Warning: Few point correspondances')
            return kp1, kp2, MIN_MATCHING_DIFF

        # Create new lists with the good features
        n_kp1, n_kp2 = [], []
        for i, good_flag in enumerate(good):
            if good_flag:
                n_kp1.append(kp1[i])
                n_kp2.append(kp2[i])

        # Format the features into float32 numpy arrays
        n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(
            n_kp2, dtype=np.float32)

        # Verify if the point correspondence points are in the same pixel coordinates
        d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

        # The mean of the differences is used to determine the amount of distance between the pixels
        diff_mean = np.mean(d)

        return n_kp1, n_kp2, diff_mean

    def process_first_frame(self, frame):
        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        if not RESIZED_FRAME_SIZE is None:
            self.gf = cv2.cuda.resize(self.gf, RESIZED_FRAME_SIZE)

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Update CPU frame
        self.cur_c_frame = self.gf.download()

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update curr img
        self.cur_g_frame = self.gf

        # Detect initial features
        self.cur_g_fts = self.detect_new_features(self.cur_g_frame)
        self.pre_g_fts = self.cur_g_fts

        # Convert keypoints to CPU
        self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
        self.pre_c_fts = self.cur_c_fts

    def detect_new_features(self, img):
        """ Detect features using selected detector
        """
        if self.DETECTOR == 'FAST' or self.DETECTOR == 'ORB':
            g_kps = self.detector.detectAsync(img, None)
        elif self.DETECTOR == 'SURF':
            g_kps = self.detector.detect(img, None)
        elif self.DETECTOR == 'SHI-TOMASI':
            g_kps = self.detector.detect(img)

        return g_kps

    def convert_fts_gpu_to_cpu(self, g_fts):
        if self.DETECTOR == 'FAST' or self.DETECTOR == 'ORB':
            c_fts = self.detector.convert(g_fts)
            c_fts = np.array([x.pt for x in c_fts], dtype=np.float32)
        elif self.DETECTOR == 'SURF':
            c_fts = cv2.cuda_SURF_CUDA.downloadKeypoints(self.detector, g_fts)
            c_fts = np.array([x.pt for x in c_fts], dtype=np.float32)
        elif self.DETECTOR == 'SHI-TOMASI':
            c_fts = g_fts.download()
            c_fts = c_fts[0]


        return c_fts


def main():
    # VO class instance
    vo = VisualOdometry(DETECTOR)

    # VO init with video
    vo.init()


if __name__ == '__main__':
    main()
