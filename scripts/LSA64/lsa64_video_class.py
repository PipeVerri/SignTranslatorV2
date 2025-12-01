import os
import cv2
import numpy as np
import mediapipe as mp
from src.mediapipe.render import render_frame, draw_landmarks_from_array
from src import Landmarks, nn_parser
from src.utils.video import frame_reader

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

class VideoLSA64:
    def __init__(self, path, meta_path, save_path, sign_counter, fps=12):
        self.landmarks = Landmarks()
        self.fps = fps
        self.save_path = save_path
        self.sign_counter = sign_counter

        # Read the video
        self.cap = cv2.VideoCapture(path)

        # Set the name and id
        file = path.name.split(".")[0].split("_")
        self.sign = str(int(file[0]))
        id = self.sign_counter.get(self.sign, 0)
        self.id = str(id)
        self.sign_counter[self.sign] = id + 1

    def generate_landmarks(self, render=False):
        with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
            for frame in frame_reader(self.cap, fps=self.fps):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hol_res = holistic.process(rgb_frame)

                if render:
                    render_frame(frame, hol_res)

                self.landmarks.add(hol_res.pose_landmarks, hol_res.left_hand_landmarks, hol_res.right_hand_landmarks)

        self.cap.release()

    def render(self):
        parsed_landmarks = self.landmarks.get_landmarks()
        for frame in frame_reader(self.cap, fps=self.fps):
            img = cv2.resize(frame, (640, 480))
            pose, left, right = next(parsed_landmarks)
            img = draw_landmarks_from_array(img, pose, connections=POSE_CONNECTIONS)
            img = draw_landmarks_from_array(img, left, connections=HAND_CONNECTIONS)
            img = draw_landmarks_from_array(img, right, connections=HAND_CONNECTIONS)
            cv2.imshow("frame", img)
            cv2.waitKey(200)

    def save(self):
        parsed_landmarks = self.landmarks.get_landmarks()
        lm_list = []
        for pose, left, right in parsed_landmarks:
            x = nn_parser(pose, left, right)
            lm_list.append(x)

        np.save(self.save_path / (self.sign + "_" + self.id), np.array(lm_list))

    def exists(self):
        return os.path.exists(self.save_path / (self.sign + "_" + self.id + ".npy"))

    def close(self):
        """Explicitly free heavy resources."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

        # If Landmarks holds big lists/arrays, clear them
        if hasattr(self, "landmarks") and self.landmarks is not None:
            try:
                self.landmarks.clear()  # if you have such a method
            except AttributeError:
                pass
            self.landmarks = None

    # optionally:
    def __del__(self):
        # Failsafe: if GC collects this object, make sure native resources are closed.
        self.close()

if __name__ == "__main__":
    rh_path = "../../data/LSA64/video/001_001_001.mp4"
    bh_path = "../../data/LSA64/video/031_001_001.mp4"
    meta_path = "../../data/LSA64/meta.csv"
    rec = VideoLSA64(rh_path, meta_path)
    rec.generate_landmarks()
    rec.render()