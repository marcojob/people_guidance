import cv2
import time
import numpy as np

MOV_FILE = 'datasets/test.MOV'
FRAME_SIZE = (1920, 1080)
RESIZED_FRAME_SIZE = (960, 540)
DETECTOR = 'ORB'
MAX_NUM_FEATURES = 1000
MIN_NUM_FEATURES = 100
USE_CLAHE = True

class VisualOdometry:
    def __init__(self, detector=DETECTOR):
        # Save detector type
        self.DETECTOR = DETECTOR

        # Initialize detector
        if detector == 'FAST':
            self.detector = cv2.cuda_FastFeatureDetector.create(threshold=75, nonmaxSuppression=True)
        elif detector == 'SIFT':
            self.detector = cv2.cuda.xfeatures2d.SIFT_create(MAX_NUM_FEATURES)
        elif detector == 'SURF':
            self.detector = cv2.cuda.SURF_CUDA_create(300, _nOctaveLayers=2)
        elif detector == 'ORB':
            self.detector = cv2.cuda_ORB.create(nfeatures=MAX_NUM_FEATURES)

        # Initialize clahe filter
        self.clahe = cv2.cuda.createCLAHE(clipLimit=5.0)

        # Frames
        self.cur_frame = None
        self.pre_frame = None

        # Features
        self.cur_c_fts = None # Current CPU features
        self.pre_c_fts = None # Previous CPU features

        self.cur_g_fts = None # Current GPU features
        self.pre_g_fts = None # Previous GPU features

    def init(self, file_path):
        # Init video capture with video
        cap = cv2.VideoCapture(file_path)

        # Handle first frame
        ret, frame = cap.read()
        self.process_first_frame(frame)

        while True:
            # Read the frames
            ret, frame = cap.read()
            if ret:
                # Main frame processing
                self.process_frame(frame)

                # Create copy for plotting
                frame = self.r_frame

                # Draw matches
                frame = self.draw_fts(frame, self.cur_c_fts)

                # Draw framerate
                frame = self.draw_framerate(frame, self.framerate)

                # Show img
                cv2.imshow("Video", self.r_frame)

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
            x, y = int(f.pt[0]), int(f.pt[1])
            cv2.circle(frame, (x,y), size, col, thickness=th)
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
        process_frame_start = time.monotonic();

        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        self.gf = cv2.cuda.resize(self.gf, RESIZED_FRAME_SIZE)
        self.r_frame = self.gf.download()

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update prev and curr img
        self.pre_frame = self.cur_frame
        self.cur_frame = self.gf

        # Detect new features if we don't have enough
        if len(self.cur_c_fts) < MIN_NUM_FEATURES or True:
            self.cur_g_fts = self.detect_new_features(self.cur_frame)
            self.pre_g_fts = self.detect_new_features(self.pre_frame)

            # Convert keypoints to CPU
            self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
            self.pre_c_fts = self.convert_fts_gpu_to_cpu(self.pre_g_fts)


        # Download frame
        self.d_frame = self.gf.download()

        # End timer and compute framerate
        self.framerate = round(1.0 / (time.monotonic() - process_frame_start))

    def process_first_frame(self, frame):
        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        self.gf = cv2.cuda.resize(self.gf, RESIZED_FRAME_SIZE)
        self.r_frame = self.gf.download()

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update curr img
        self.cur_frame = self.gf

        # Detect initial features
        self.cur_g_fts = self.detect_new_features(self.cur_frame)

        # Convert keypoints to CPU
        self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)

    def detect_new_features(self, img):
        """ Detect features using selected detector
        """
        if self.DETECTOR == 'FAST' or self.DETECTOR == 'ORB':
            g_kps = self.detector.detectAsync(img, None)
        elif self.DETECTOR == 'SURF':
            g_kps = self.detector.detect(img, None)

        return g_kps

    def convert_fts_gpu_to_cpu(self, g_fts):
        if self.DETECTOR == 'FAST' or self.DETECTOR == 'ORB':
            c_fts = self.detector.convert(g_fts)
        elif self.DETECTOR == 'SURF':
            c_fts = cv2.cuda_SURF_CUDA.downloadKeypoints(self.detector, g_fts)

        return c_fts


def main():
    # VO class instance
    vo = VisualOdometry(DETECTOR)

    # VO init with video
    vo.init(MOV_FILE)

if __name__ == '__main__':
    main()