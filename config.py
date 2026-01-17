"""
Configuration and constants
"""

import torch
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
CLASSES = ['music', 'noise']

# Audio configuration
SAMPLE_RATE = 44100
DURATION = 5.0
SILENCE_THRESHOLD = 0.005

# Model path
MODEL_PATH = "sound_model.pth"

# Spectrogram settings
SPECTROGRAM_DIR = "spectrograms"