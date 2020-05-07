import cv2
import numpy as np
import math

"""
General helper definition
"""

# VisualOdemetry class helpers

DETECTOR = 'SIFT' # Can be either FAST, SIFT, SURF, SHI-TOMASI

STAGE_FIRST = 0 # First stage: first image
STAGE_SECOND = 1 # Second stage: second image found
STAGE_DEFAULT = 2 # Third stage: pipeline full

# Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Params for Shi-Tomasi
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
MIN_MATCHING_DIFF = 1  # Minimum difference in the KLT point correspondence
MIN_NUM_FEATURES = 100 # If features fall below this threshold we detect new ones
MAX_NUM_FEATURES = 5000 # Maximum number of features
DIFF_THRESHOLD = 0.5

def drawOpticalFlowField(img, ref_pts, cur_pts):
    """ Shows a window which shows the optical flow of detected features. """

    # Draw the tracks
    for i, (new, old) in enumerate(zip(cur_pts, ref_pts)):

        a,b = new.ravel()
        c,d = old.ravel()
        v1 = tuple((new - old)*2.5 + old)
        d_v = [new-old][0]*0.75
        arrow_color = (28,24,178)
        arrow_t1 = rotateFunct([d_v], 0.5)
        arrow_t2 = rotateFunct([d_v], -0.5)
        tip1 = tuple(np.float32(np.array([c, d]) + arrow_t1)[0])
        tip2 = tuple(np.float32(np.array([c, d]) + arrow_t2)[0])
        cv2.line(img, v1,(c,d), (0,255,0), 2)
        cv2.line(img, (c,d), tip1, arrow_color, 2)
        cv2.line(img, (c,d), tip2, arrow_color, 2)
        cv2.circle(img, v1,1,(0,255,0),-1)

    cv2.imshow('Optical Flow Field', img)
    cv2.waitKey(1)

    return

def rotateFunct(pts_l, angle, degrees=False):
    """ Returns a rotated list(function) by the provided angle."""
    if degrees == True:
        theta = math.radians(angle)
    else:
        theta = angle

    R = np.array([ [math.cos(theta), -math.sin(theta)],
                   [math.sin(theta), math.cos(theta)] ])
    rot_pts = []
    for v in pts_l:
        v = np.array(v).transpose()
        v = R.dot(v)
        v = v.transpose()
        rot_pts.append(v)

    return rot_pts

def RT_trajectory_window(window, x, y, z, img_id):
    """ Real-time trajectory window. Draws the VO trajectory
        while the images are being processed. """
    # Drawing the points and creating the Real-time trajectory window
    draw_x, draw_y = int(x) + 290, int(z) + 90
    cv2.circle(window, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
    cv2.rectangle(window, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
    cv2.putText(window, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    cv2.imshow('Real-Time Trajectory', window)
    cv2.waitKey(1)

    return window