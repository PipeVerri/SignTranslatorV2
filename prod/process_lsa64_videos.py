import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from utils.LSA64_parser import VideoLSA64
from tqdm import tqdm

root_dir = Path(__file__).parent.parent
videos = os.listdir(root_dir / "data" / "LSA64" / "video")
meta_path = root_dir / "data" / "LSA64" / "meta.csv"
save_path = root_dir / "data" / "LSA64" / "landmarks"

# Usar multiprocessing para evitar el GIL y aunque mediapipe no sea pickle-able, no tengo que pasarlo entre procesos
def process_video(video_path):
    vid = VideoLSA64(root_dir / "data" / "LSA64" / "video" / video_path, meta_path, save_path)
    if not vid.exists():
        vid.generate_landmarks()
        vid.save()
    else:
        print("VIDEO EXISTS")
        vid.cap.release()

    #print(f"{vid.sign}, {vid.id} done")

process_video(videos[0])

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        list(tqdm(executor.map(process_video, videos),
                 total=len(videos),
                 desc="Processing videos"))