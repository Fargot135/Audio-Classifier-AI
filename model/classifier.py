"""
CNN for audio classification
"""

import torch.nn as nn
from config import CLASSES


class SoundClassifier(nn.Module):
    """CNN for audio classification"""
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 38 * 38, 128), 
            nn.ReLU(),
            nn.Linear(128, len(CLASSES))
        )
    
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))