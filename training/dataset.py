import numpy as np
import torch
from torch.utils.data import Dataset

from training.labeling import label_trend


class GlucoseTrendDataset(Dataset):
    def __init__(
        self,
        df,
        classes,
        collapse_map=None,
        id_col="id",
        segment_col="id_segment",
        time_col="time",
        target_col="gl",
        input_length=12,
        horizon=3,
        horizon_minutes=15.0,
    ):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.collapse_map = collapse_map
        self.input_length = input_length
        self.horizon = horizon
        self.horizon_minutes = horizon_minutes

        self.windows = []
        self.labels = []
        # Index labels of each window's input + horizon rows in `df`, so a
        # window can be traced back to its source rows (split audits).
        self.row_ids = []
        # Per-window provenance, aligned with windows/labels: participant,
        # segment, anchor time (last input sample) and target time.
        self.participants = []
        self.segments = []
        self.anchor_times = []
        self.target_times = []

        span = input_length + horizon
        for (pid, seg), group in df.groupby([id_col, segment_col]):
            group = group.sort_values(time_col)
            values = group[target_col].to_numpy(dtype=np.float32)
            times = group[time_col].to_numpy()
            index = group.index.to_numpy()
            n_valid = len(values) - span + 1
            for i in range(max(n_valid, 0)):
                self.row_ids.append(index[i:i + span])
                self.participants.append(pid)
                self.segments.append(seg)
                self.anchor_times.append(times[i + input_length - 1])
                self.target_times.append(times[i + span - 1])
                window = values[i:i + input_length]
                last_value = window[-1]
                future_value = values[i + input_length + horizon - 1]
                raw_label = label_trend(float(last_value), float(future_value), horizon_minutes)
                label = collapse_map[raw_label] if collapse_map else raw_label
                self.windows.append(window)
                self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).float().unsqueeze(0)
        y = self.labels[idx]
        return x, y
