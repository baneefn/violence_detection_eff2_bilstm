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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) != 0 else np.zeros_like(cam)
        cam = cv2.resize(cam, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return cam


def generate_gradcam_video(model, input_path, output_path, progress_placeholder=None):
    video_reader = cv2.VideoCapture(input_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    gradcam = GradCAM(model, target_layer=model.base_model.stages[3].blocks[-1])
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

            cam = gradcam.generate_cam(input_tensor, predicted_label)
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            overlayed = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
            cv2.putText(overlayed, f"{predicted_class_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(overlayed, f"Frame: {current_frame}/{total_frames}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            video_writer.write(overlayed)

            if progress_placeholder:
                progress_placeholder.text(f"Generating GradCAM: {current_frame}/{total_frames}")

    video_reader.release()
    video_writer.release()