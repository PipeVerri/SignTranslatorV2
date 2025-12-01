from utils.video import frame_reader
import os
import math
import mediapipe as mp
import cv2
from utils.mp_utils.parse import mp_to_arr
from src import nn_parser
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LSA64_dir = Path(__file__).parent.parent.parent.parent
downloads_dir = LSA64_dir / "TED_videos"
landmarks_dir = LSA64_dir / "landmarks"
landmarks_dir.mkdir(exist_ok=True)

videos = os.listdir(downloads_dir)
landmark_count = 0

def process_video(video_path: Path, landmark_count_start: int, clip_len: int = 12) -> int:
    """Process a single video using multi-threaded Holistic workers."""
    cap = cv2.VideoCapture(str(video_path))

    # Read all frames first (at 12 fps, as in your original code)
    frames = [frame for frame in frame_reader(cap, fps=6)]
    cap.release()

    if not frames:
        return landmark_count_start

    n_frames = len(frames)
    # Number of workers – tweak as you like
    max_workers = min(os.cpu_count() or 1, n_frames)
    if max_workers < 2:
        max_workers = 1

    # Split into contiguous chunks
    chunk_size = math.ceil(n_frames / max_workers)
    chunks = []
    for i in range(max_workers):
        start = i * chunk_size
        if start >= n_frames:
            break
        end = min(n_frames, start + chunk_size)
        chunks.append((start, frames[start:end]))

    def worker(start_idx, frames_chunk):
        """Worker: new Holistic instance, processes a chunk of frames."""
        results = []
        with mp.solutions.holistic.Holistic(model_complexity=2) as holistic:
            for offset, frame in enumerate(frames_chunk):
                frame_idx = start_idx + offset
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hol_res = holistic.process(rgb)

                if hol_res.pose_landmarks and \
                   hol_res.left_hand_landmarks and \
                   hol_res.right_hand_landmarks:
                    pose_arr = mp_to_arr(hol_res.pose_landmarks.landmark)
                    left_arr = mp_to_arr(hol_res.left_hand_landmarks.landmark)
                    right_arr = mp_to_arr(hol_res.right_hand_landmarks.landmark)
                    arr = nn_parser(pose_arr, left_arr, right_arr)
                else:
                    arr = None  # invalid frame

                results.append((frame_idx, arr))
        return results

    # Collect per-frame results
    frame_results = [None] * n_frames
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(worker, start_idx, frames_chunk)
            for (start_idx, frames_chunk) in chunks
        ]
        for fut in as_completed(futures):
            for frame_idx, arr in fut.result():
                frame_results[frame_idx] = arr

    # Now frame_results[i] is either an array or None, in frame order.
    # Reuse your original "48 consecutive valid frames" logic:
    frames_window = []
    landmark_count = landmark_count_start

    for arr in frame_results:
        if arr is None:
            frames_window = []
        else:
            frames_window.append(arr)

        if len(frames_window) == clip_len:
            print("Found clip")
            frames_arr = np.array(frames_window)
            np.save(
                landmarks_dir / f"65_{landmark_count}.npy",
                frames_arr
            )
            landmark_count += 1
            # NOTE: your original code does NOT reset frames here.
            # If you want overlapping clips, do:
            # frames_window.pop(0)

    return landmark_count


for video in videos:
    video_path = downloads_dir / video
    landmark_count = process_video(video_path, landmark_count)
    print(f"Video done")