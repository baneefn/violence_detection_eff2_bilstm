import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import timm
from timm import create_model  # Untuk EfficientFormerV2
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchinfo import summary
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import json
import torch.optim as optim
import torch.nn.functional as F
import time
from collections import deque
import random
import sys
from ptflops import get_model_complexity_info
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import label_binarize
from glob import glob
from collections import defaultdict
import datetime

CLASSES_LIST = ['NonViolence', 'Violence']
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_CLASSES = len(CLASSES_LIST)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)             # Linear projection W
        self.b = nn.Parameter(torch.zeros(input_dim))        # Bias vector b
        self.u = nn.Linear(input_dim, 1)                      # Scoring vector u
        self.attention_weights = None                        # Optional: save α_t for analysis

    def forward(self, h_bilstm):
        # Step 1: Alignment vector u_it = tanh(W * h_t + b)
        u_it = torch.tanh(self.W(h_bilstm) + self.b)          # [B, T, d]

        # Step 2: Attention score α_t = softmax(u^T * u_it)
        score = self.u(u_it)                                  # [B, T, 1]
        α_t = torch.softmax(score, dim=1)                     # [B, T, 1]
        self.attention_weights = α_t                          # (Optional: for visualization)

        # Step 3: Weighted sum context vector
        v = torch.sum(α_t * h_bilstm, dim=1)                  # [B, d]

        # Step 4: Residual connection: v + ∑ h_t (optional, as in your paper)
        residual = torch.sum(h_bilstm, dim=1)                 # [B, d]
        v = v + residual

        return v


class EfficientFormerV2WithBiLSTM(nn.Module):
    def __init__(self, base_model_name, num_classes, hidden_dim=176, lstm_units=32, weight_decay=1e-4, use_attention=True):
        super(EfficientFormerV2WithBiLSTM, self).__init__()
        self.use_attention = use_attention
        self.base_model = create_model(base_model_name, pretrained=True, num_classes=0, in_chans=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_units, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(lstm_units * 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_units * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),

        )
        # Apply L2 regularization (weight decay)
        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                module.weight_decay = weight_decay

    def forward(self, x):
        # print("Input shape to model:", x.shape)  # Debugging
        batch_size, seq_len, c, h, w = x.size()
        # Cek apakah channel pertama adalah 3 (RGB)
        if c != 3:
            raise ValueError(f"Expected 3 channels, but got {c}")
        x = x.view(batch_size * seq_len, c, h, w)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize
        features = self.base_model(x)
        features = self.flatten(features)
        features = features.view(batch_size, seq_len, -1)
        features = self.dropout(features)
        lstm_out, _ = self.lstm(features)
        if self.use_attention:
            context = self.attention(lstm_out)  # Attention mechanism
        else:
            context = torch.mean(lstm_out, dim=1)  # Global Average Pooling jika tanpa Attention
        output = self.fc_layers(context)  # Klasifikasi akhir
        return output

use_attention = False  # Ganti ke False untuk pengujian tanpa Attention
model = EfficientFormerV2WithBiLSTM('efficientformerv2_s0', num_classes=NUM_CLASSES, use_attention=use_attention).to(DEVICE)
summary(model, input_size=(BATCH_SIZE, SEQUENCE_LENGTH, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)