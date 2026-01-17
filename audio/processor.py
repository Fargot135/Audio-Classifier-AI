"""
Audio processing and prediction
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import io
from pathlib import Path

from model.classifier import SoundClassifier
from config import DEVICE, CLASSES, SAMPLE_RATE, MODEL_PATH, SPECTROGRAM_DIR


class AudioProcessor:
    """Handles audio processing and prediction"""
    
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((155, 154)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def load_model(self):
        """Load trained model"""
        model = SoundClassifier().to(DEVICE)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            self.model = model
            return True
        except FileNotFoundError:
            return False
    
    def predict_audio(self, audio):
        """Classify audio sample"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Generate spectrogram
        fig = plt.figure(figsize=(2, 2))
        S = librosa.feature.melspectrogram(y=audio.flatten(), sr=SAMPLE_RATE)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=SAMPLE_RATE)
        plt.axis('off')
        
        # Save debug image
        Path(SPECTROGRAM_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{SPECTROGRAM_DIR}/latest.png", bbox_inches='tight', pad_inches=0)
        
        # Convert to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB')
        buf.close()
        
        # Predict
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return CLASSES[predicted.item()], confidence.item() * 100