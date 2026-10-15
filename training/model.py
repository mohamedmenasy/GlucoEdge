import torch.nn as nn


class TrendCNN(nn.Module):
    def __init__(self, num_classes: int = 5, channels=(24, 36),
                 in_channels: int = 1, head: str = "gap", length: int = 12):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.head = head
        if head == "flat":
            # position-preserving: classify the full [c2, length] feature map
            self.pool = None
            self.fc = nn.Linear(c2 * length, num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.pool is None:
            return self.fc(x.flatten(1))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = (3 - 1) * dilation  # causal left pad, kernel 3
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.relu(y) + x


class TrendTCN(nn.Module):
    """Causal dilated TCN; receptive field 31 steps (sweep r3 spec section 5)."""

    def __init__(self, num_classes: int = 5, channels: int = 72, in_channels: int = 1):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[_TCNBlock(channels, d) for d in (1, 2, 4, 8)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.blocks(self.inp(x))
        return self.fc(self.pool(x).squeeze(-1))
