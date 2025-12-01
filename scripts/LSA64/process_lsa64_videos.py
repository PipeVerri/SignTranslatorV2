import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scripts.LSA64.lsa64_video_class import VideoLSA64
from tqdm import tqdm
from multiprocessing import Manager

root_dir = Path(__file__).parent.parent
videos = os.listdir(root_dir / "data" / "LSA64" / "video")
meta_path = root_dir / "data" / "LSA64" / "meta.csv"
save_path = root_dir / "data" / "LSA64" / "landmarks"

manager = Manager()
sign_counter = manager.dict()

# Usar multiprocessing para evitar el GIL y aunque mediapipe no sea pickle-able, no tengo que pasarlo entre procesos
def process_video(video_path):
    try:
        vid = VideoLSA64(root_dir / "data" / "LSA64" / "video" / video_path, meta_path, save_path, sign_counter, fps=6)
        if not vid.exists():
            vid.generate_landmarks()
            vid.save()
        else:
            print("VIDEO EXISTS")
    finally:
        vid.close()

    #print(f"{vid.sign}, {vid.id} done")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(process_video, videos),
                 total=len(videos),
                 desc="Processing videos"))