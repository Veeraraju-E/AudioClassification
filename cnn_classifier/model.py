"""
Let's try with conv1d blocks
"""
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch

sys.path.append('../')
sys.path.append('../../')

class AudioClassifierTimeDomain(nn.Module):
    """
    Classifier built for log-spectrograms in the time-domain
    Expected input tensors of [B, 1, num_timesteps]
    """
    def __init__(self, num_classes=50):
        super(AudioClassifierTimeDomain, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 128, 80, 4),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 128, 3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(4),
            nn.Conv1d(256, 512, 3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x shape: [B, 1, time_steps] 
        """
        out = self.block1(x)
        # print(f'out.shape: {out.shape}')    # B, 512, 1
        out = out.view(x.size(0), -1)
        # print(f'out.shape: {out.shape}')    # B, 512
        out = self.classifier(out)         
        # print(f'out.shape: {out.shape}')    # [B, num_classes]
        return F.log_softmax(out, dim=1)        


if __name__ == "__main__":
    batch_size = 8
    time_steps = 44100  # 1 second of audio at 44.1kHz
    x = torch.randn(batch_size, 1, time_steps)
    model = AudioClassifierTimeDomain()
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [B, 50]

