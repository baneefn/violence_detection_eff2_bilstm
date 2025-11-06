import streamlit as st
import os
import tempfile
import uuid
from utils.model_loader import load_model_from_checkpoint
from utils.inference import predict_video_with_progress
from utils.gradcam_utils import generate_gradcam_video
from utils.video_utils import save_uploaded_file, get_video_info
from utils.youtube_utils import download_youtube_video

st.set_page_config(page_title="Violence Detection", layout="wide")
st.title("🚨 Violence Detection System")
st.markdown("Upload, record, or use a YouTube link to detect violent actions using EfficientFormerV2 + BiLSTM + Attention model.")

# Select input source
# input_source = st.radio("Pilih sumber video:", ["📤 Upload Lokal", "🔗 Link YouTube", "📷 Kamera"], horizontal=True)
input_source = st.radio("Pilih sumber video:", ["📤 Upload Lokal"], horizontal=True)
video_bytes = None
video_path = None
video_preview_path = None

# Upload
if input_source == "📤 Upload Lokal":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi"])
    if uploaded_file is not None:
        video_path = save_uploaded_file(uploaded_file)

# # YouTube
# elif input_source == "🔗 Link YouTube":
#     yt_url = st.text_input("Masukkan URL YouTube")
#     if yt_url:
#         with st.spinner("Mengunduh video dari YouTube..."):
#             video_path = download_youtube_video(yt_url)

# # Kamera
# elif input_source == "📷 Kamera":
#     camera_input = st.camera_input("Rekam dari Kamera")
#     if camera_input is not None:
#         video_path = save_uploaded_file(camera_input)

# Jika sudah ada video
if video_path:
    # st.video(video_path)
    from utils.video_utils import show_video_preview_or_fallback

    show_video_preview_or_fallback(video_path)

    total_frames, fps, duration, width, height = get_video_info(video_path)
    st.info(f"Durasi: {duration:.2f}s | Resolusi: {width}x{height} | Total Frame: {total_frames}")

    if st.button("🚀 Lakukan Deteksi Kekerasan"):
        st.subheader("🔍 Proses Inferensi... Harap Tunggu")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎬 Video Prediksi")
            pred_placeholder = st.empty()

        with col2:
            st.markdown("#### 🔥 Grad-CAM Overlay")
            gradcam_placeholder = st.empty()

        # Temp folder for output
        temp_output_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.makedirs(temp_output_dir, exist_ok=True)

        pred_path = os.path.join(temp_output_dir, "pred_result.mp4")
        gradcam_path = os.path.join(temp_output_dir, "gradcam_result.mp4")

        # Load model
        model = load_model_from_checkpoint()

        # Predict with UI updates
        predict_video_with_progress(model, video_path, pred_path, progress_placeholder=pred_placeholder)
        generate_gradcam_video(model, video_path, gradcam_path, progress_placeholder=gradcam_placeholder)

        # Show results
        with col1:
            # st.video(pred_path)

            show_video_preview_or_fallback(pred_path)

            with open(pred_path, "rb") as f:
                st.download_button("💾 Download Prediksi", f, file_name="prediction_result.mp4")

        with col2:
            # st.video(gradcam_path)
            # from utils.video_utils import show_video_preview_or_fallback

            show_video_preview_or_fallback(gradcam_path)

            with open(gradcam_path, "rb") as f:
                st.download_button("💾 Download Grad-CAM", f, file_name="gradcam_result.mp4")
