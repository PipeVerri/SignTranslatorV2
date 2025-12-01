from yt_dlp import YoutubeDL
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent

ydl_ops = {
    'format': 'bestvideo[height=240][vcodec!=none][acodec=none]',
    'outtmpl': root_dir / "data" / "LSA64" / "TED_videos" / "%(title)s.%(ext)s"
}

with open("videos", "r") as f:
    videos = f.read().splitlines()

with YoutubeDL(ydl_ops) as ydl:
    for vid in videos:
        ydl.download([vid])