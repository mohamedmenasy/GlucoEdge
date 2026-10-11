import torch.nn as nn


class TrendCNN(nn.Module):
    def __init__(self, num_classes: int = 5, channels=(24, 36)):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
