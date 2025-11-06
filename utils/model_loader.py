import torch
import os
from checkpoints.eff2_model import EfficientFormerV2WithBiLSTM

# Konfigurasi global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # NonViolence dan Violence
CHECKPOINT_PATH = r"E:\KULIAH\TA_SI_GUE\deteksi_kekerasan_ta\checkpoints\Eff2Bilstm_checkpoint.pth"


def load_model_from_checkpoint(path=CHECKPOINT_PATH, use_attention=True):
    """
    Memuat model dari checkpoint dan mengembalikannya dalam mode evaluasi.
    """
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
