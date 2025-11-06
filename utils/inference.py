import torch
import os
import cv2
import numpy as np
from collections import deque
from checkpoints.eff2_model import EfficientFormerV2WithBiLSTM

# Konfigurasi global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
CLASSES_LIST = ['NonViolence', 'Violence']
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64


def load_model_from_checkpoint(path="checkpoints/EFF2_BLSTM_ATT_StageAll_6dataset_model.pth", use_attention=True):
    model = EfficientFormerV2WithBiLSTM(
        base_model_name='efficientformerv2_s0',
        num_classes=NUM_CLASSES,
        use_attention=use_attention
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_video_with_progress(model, input_path, output_path, progress_placeholder=None):
    video_reader = cv2.VideoCapture(input_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    current_frame = 0

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        current_frame += 1

        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            input_frames = np.array(frames_queue, dtype=np.float32)
            input_frames = np.transpose(input_frames, (0, 3, 1, 2))
            input_tensor = torch.tensor(input_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_label = torch.argmax(prediction, dim=1).cpu().numpy()[0]
                predicted_class_name = CLASSES_LIST[predicted_label]

        # Overlay
        color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
        cv2.putText(frame, f"{predicted_class_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        video_writer.write(frame)

        if progress_placeholder:
            progress_placeholder.text(f"Processing Frame: {current_frame}/{total_frames}")

    video_reader.release()
    video_writer.release()
