"""CLI entry point for training the glucose trend classifier."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from training.dataset import GlucoseTrendDataset
from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP
from training.model import TrendCNN

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
GLUCOBENCH_ROOT = GLUCOEDGE_ROOT / "GlucoBench"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def load_formatter(dataset: str) -> "DataFormatter":
    # Imported lazily: GlucoBench is an external clone, not a package
    # dependency, so only the code path that actually reads its data
    # needs it importable (keeps `pytest` runnable without cloning it).
    sys.path.insert(0, str(GLUCOBENCH_ROOT))
    from data_formatter.base import DataFormatter

    config_path = GLUCOBENCH_ROOT / "config" / f"{dataset}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["data_csv_path"] = str(GLUCOBENCH_ROOT / "raw_data" / f"{dataset}.csv")
    return DataFormatter(config)


def class_weights(train_ds: GlucoseTrendDataset, num_classes: int) -> torch.Tensor:
    labels = np.array(train_ds.labels)
    present = np.unique(labels)
    weights = np.ones(num_classes, dtype=np.float64)
    if len(present) > 0:
        present_weights = compute_class_weight("balanced", classes=present, y=labels)
        weights[present] = present_weights
    return torch.tensor(weights, dtype=torch.float32)


def train_model(train_ds, val_ds, num_classes, epochs, device):
    model = TrendCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_ds, num_classes).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        denom = max(len(val_loader), 1)
        print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss / denom:.4f}")

    return model


def evaluate(model, test_ds, classes, device):
    loader = DataLoader(test_ds, batch_size=64)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    return classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        output_dict=True,
        zero_division=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    formatter = load_formatter(args.dataset)

    if args.classes == "5":
        classes, collapse_map = FIVE_CLASSES, None
    else:
        classes, collapse_map = THREE_CLASSES, THREE_CLASS_MAP

    train_ds = GlucoseTrendDataset(formatter.train_data, classes=classes, collapse_map=collapse_map)
    val_ds = GlucoseTrendDataset(formatter.val_data, classes=classes, collapse_map=collapse_map)
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=classes, collapse_map=collapse_map)

    print(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    train_label_counts = {
        c: int(np.sum(np.array(train_ds.labels) == i)) for i, c in enumerate(classes)
    }
    print(f"train class counts: {train_label_counts}")

    model = train_model(train_ds, val_ds, len(classes), args.epochs, device)

    results_dir = GLUCOEDGE_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if args.save_checkpoint:
        checkpoint_path = results_dir / f"{args.dataset}_{args.classes}class_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"wrote {checkpoint_path}")

    report = evaluate(model, test_ds, classes, device)

    out_path = results_dir / f"{args.dataset}_{args.classes}class_report.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "num_classes": len(classes),
                "classes": classes,
                "train_label_counts": train_label_counts,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
