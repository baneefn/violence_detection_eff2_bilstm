# Violence Detection System (EfficientFormerV2 + BiLSTM + Attention)

A Streamlit-based application to detect violent actions in videos using an EfficientFormerV2 backbone combined with a BiLSTM sequence model and optional Attention. The app supports local video upload, generates prediction-overlaid videos, and produces Grad-CAM visualizations to highlight salient regions contributing to the model’s decision.


## ✨ Features
- Upload local videos for inference
- Frame-wise prediction overlay: "Violence" vs "NonViolence"
- Grad-CAM visualization video to interpret the model
- Basic video metadata preview (duration, resolution, frame count)
- Downloadable outputs (prediction video and Grad-CAM video)


## 📂 Project Structure
```
E:/KULIAH/TA_SI_GUE/deteksi_kekerasan_ta
├─ app.py                           # Streamlit UI
├─ checkpoints/
│  ├─ Eff2Bilstm_all_model.pth     # (example) trained weights
│  ├─ Eff2Bilstm_checkpoint.pth     # default checkpoint used by utils/model_loader.py
│  ├─ eff2_model.py                 # model architecture (EfficientFormerV2 + BiLSTM + Attention)
│  ├─ model_att.pth                 # (example) trained weights with attention
│  └─ model_noatt.pth               # (example) trained weights without attention
├─ create_model/
│  └─ EFF2_BLSTM_ATT_STAGEALL.ipynb # training/experimentation notebook
├─ temp_videos/                     # sample output previews (for reference)
│  ├─ grad/*.mp4
│  └─ pred/*.mp4
└─ utils/
   ├─ gradcam_utils.py              # Grad-CAM generation
   ├─ inference.py                  # video inference utilities
   ├─ model_loader.py               # checkpoint loading wrapper
   ├─ video_utils.py                # video I/O helpers
   └─ youtube_utils.py              # (optional) YouTube downloader helpers
```


## 🧠 Model Overview
- Backbone: EfficientFormerV2 (variant `efficientformerv2_s0`)
- Temporal modeling: BiLSTM over sequences of frames (default sequence length: 16)
- Classification: 2 classes — `NonViolence`, `Violence`
- Optional attention layer after BiLSTM

Model definition is implemented in `checkpoints/eff2_model.py` and loaded via `utils/model_loader.py`.


## 🔧 Requirements
- Python 3.9+ (recommended)
- OS: Windows (paths in repo use Windows-style separators) — works on Linux/Mac with small path adjustments

Python packages (install via pip):
- streamlit
- torch (install a CUDA or CPU build appropriate for your system)
- torchvision (if required by your PyTorch install)
- opencv-python
- numpy
- yt-dlp (optional; only if enabling YouTube download)

Example installation:
```
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
pip install --upgrade pip
pip install streamlit torch torchvision opencv-python numpy yt-dlp
```

Note: For PyTorch, prefer the official selector to choose the right command for your CUDA/CPU setup:
https://pytorch.org/get-started/locally/


## 🗂️ Checkpoints
Default checkpoint path is defined in `utils/model_loader.py`:
```
CHECKPOINT_PATH = r"E:\KULIAH\TA_SI_GUE\deteksi_kekerasan_ta\checkpoints\Eff2Bilstm_checkpoint.pth"
```
If you clone/move this project, update the path or make it relative. Two easy options:
1) Keep absolute path correct for your machine.
2) Change to a relative path and place the checkpoint file accordingly, e.g.:
```
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "Eff2Bilstm_checkpoint.pth")
CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)
```

Put your trained weights in the `checkpoints/` folder and ensure the file name matches the configured `CHECKPOINT_PATH`.


## ▶️ Run the App
1) Activate your virtual environment and ensure dependencies are installed.
2) Ensure the checkpoint path in `utils/model_loader.py` points to a valid `.pth` file.
3) Run Streamlit:
```
streamlit run app.py
```
4) In your browser, choose "📤 Upload Lokal" and upload an `.mp4` or `.avi` file.
5) Click "🚀 Lakukan Deteksi Kekerasan" to start inference.
6) When finished, you can preview and download the prediction and Grad-CAM videos.


## 🎛️ App Inputs (current state)
- Upload Lokal: Enabled.
- YouTube Link: The code is present but commented out in `app.py`. To enable, uncomment the relevant block and ensure `yt-dlp` is installed.
- Kamera: The code is present but commented out. Requires Streamlit `st.camera_input` support and testing on your setup.


## 📏 Inference Details
- Frames are resized to 64×64 and normalized to [0, 1].
- A sliding window of `SEQUENCE_LENGTH=16` frames is fed into the model.
- The predicted class is overlaid on each output frame.
- Grad-CAM video is generated to visualize important regions.


## 🧪 Sample Outputs
See `temp_videos/pred/*.mp4` and `temp_videos/grad/*.mp4` for example result videos.


## 🐛 Troubleshooting
- Streamlit can’t preview very small resolution videos
  - The app automatically falls back to showing the first-frame thumbnail (see `utils/video_utils.py`).
- CUDA not available / PyTorch errors
  - Install the correct PyTorch build; fall back to CPU if needed. The app will auto-detect device: `cuda` if available else `cpu`.
- Checkpoint not found / wrong path
  - Update `CHECKPOINT_PATH` in `utils/model_loader.py` and ensure the file exists.
- Codec issues writing videos
  - OpenCV uses `mp4v` in `utils/inference.py`. Ensure the codec is supported on your OS; consider trying a different fourcc if needed.


## 📜 License
This project is for academic/educational purposes. Add a suitable license (e.g., MIT) if you plan to distribute.


## 🙏 Acknowledgements
- EfficientFormerV2 authors and community implementations
- PyTorch, OpenCV, Streamlit
- yt-dlp for optional YouTube downloading


## Dataset Extracted 
https://drive.google.com/drive/folders/1Pt8zauHg4W_zbMuUEOB25BRnLFCwAJ_a?usp=sharing
