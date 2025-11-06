import torch
import os
import cv2
import numpy as np
from collections import deque
import tempfile
import uuid
import yt_dlp

# Konfigurasi global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
CLASSES_LIST = ['NonViolence', 'Violence']
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64


def save_uploaded_file(uploaded_file):
    """Simpan file Streamlit ke file sementara dan kembalikan pathnya."""
    file_ext = os.path.splitext(uploaded_file.name)[-1]
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex}{file_ext}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def get_video_info(video_path):
    """Ambil info dasar video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return total_frames, fps, duration, width, height


def download_youtube_video(yt_url):
    """Unduh video dari YouTube dan simpan ke file sementara."""
    temp_path = os.path.join(tempfile.gettempdir(), f"yt_{uuid.uuid4().hex}.mp4")
    ydl_opts = {
        'outtmpl': temp_path,
        'format': 'mp4',
        'quiet': True,
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'postprocessors': [],  # Hapus ffmpeg jika belum tersedia
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])
    return temp_path
