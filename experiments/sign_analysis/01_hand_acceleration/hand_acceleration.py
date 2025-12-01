import matplotlib
matplotlib.use('qt5agg')

from src.mediapipe.parse import mp_to_arr
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
from src.utils.video import camera_reader
from src import Landmarks, nn_parser
import numpy as np

pos = []
vel = []
accel = []

lm = Landmarks()
MAX_POINTS = 80
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(7, 7))

with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2) as hol:
    for frame in camera_reader(fps=12):
        hol_res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hol_res.pose_landmarks is None or hol_res.left_hand_landmarks is None or hol_res.right_hand_landmarks is None:
            continue
        pose = mp_to_arr(hol_res.pose_landmarks.landmark)
        left = mp_to_arr(hol_res.left_hand_landmarks.landmark)
        right = mp_to_arr(hol_res.right_hand_landmarks.landmark)
        arr = nn_parser(pose, left, right)
        pos.append(arr)

        if len(pos) > 1:
            ax[0].clear()
            vel.append((pos[-1] - pos[-2]) / (1 / 12))
            vel_mat = np.array(vel)

            idx = np.argmax(np.abs(vel_mat), axis=1)
            ax[0].plot(vel_mat[:, idx])
        if len(vel) > 1:
            ax[1].clear()
            accel.append((vel[-1] - vel[-2]) / (1 / 12))
            accel_mat = np.array(accel)
            norms = np.max(accel_mat, axis=1)
            ax[1].plot(norms)

        if len(pos) > MAX_POINTS:
            pos = pos[-MAX_POINTS:]
        if len(vel) > MAX_POINTS:
            vel = vel[-MAX_POINTS:]
        if len(accel) > MAX_POINTS:
            accel = accel[-MAX_POINTS:]

        plt.pause(0.05)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)