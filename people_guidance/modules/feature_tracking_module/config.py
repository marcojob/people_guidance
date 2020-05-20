import cv2

DETECTOR = 'FAST' # Can be either FAST, SIFT, SURF, SHI-TOMASI, ORB, REGULAR_GRID
USE_OPTICAL_FLOW = True
USE_E = True
USE_CLAHE = False

# Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Params for Shi-Tomasi
shi_tomasi_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

OF_MIN_MATCHING_DIFF = 1  # Minimum difference in the KLT point correspondence
OF_MIN_NUM_FEATURES = 100 # If features fall below this threshold we detect new ones
OF_MAX_NUM_FEATURES = 1000 # Maximum number of features
OF_DIFF_THRESHOLD = 1
FAST_THRESHOLD = 40