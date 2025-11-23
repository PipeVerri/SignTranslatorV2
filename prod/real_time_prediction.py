import cv2
from utils.video import camera_reader
import mediapipe as mp
from utils.landmarks.landmarks import Landmarks, nn_parser
from gtts import gTTS
import pandas as pd

signs = pd.read_csv("../data/LSA64/meta.csv")
print()

lm = Landmarks()

with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
    for frame in camera_reader():
        hol_res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm.add(hol_res.pose_landmarks, hol_res.left_hand_landmarks, hol_res.right_hand_landmarks)
        for pose, left, right in lm.get_landmarks():
            parsed = nn_parser(pose, left, right)