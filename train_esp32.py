from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

from UT_HAR_model import UT_HAR_CNN_LSTM
from build_new_test_dataset import read_recording, make_windows, get_activity

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "live_recordings"
BASE_CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"
OUTPUT_CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_esp32.pth"

CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]
SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_recordings():
    paths = sorted(RECORDINGS_DIR.glob("*.csv"))
    by_class = {label: [] for label in CLASSES}

    for path in paths:
        label = get_activity(path)
        frames = read_recording(path)
        windows = make_windows(frames)
        if len(windows) == 0:
            print(f"WARNING: no windows, skipping {path.name}")
            continue
        by_class[label].append((path.name, windows))
        print(f"{label}: {path.name} -> {len(windows)} windows")

    for label in CLASSES:
        if len(by_class[label]) < 2:
            raise RuntimeError(
                f"Need at least 2 recordings for {label}; found {len(by_class[label])}"
            )
    return by_class


def main():
    set_seed(SEED)
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_CHECKPOINT}")
    if not RECORDINGS_DIR.exists():
        raise FileNotFoundError(f"Recordings folder not found: {RECORDINGS_DIR}")

    by_class = load_recordings()
    train_records, val_records = [], []
    rng = random.Random(SEED)

    # Hold out one entire recording per class for validation.
    for label in CLASSES:
        records = list(by_class[label])
        rng.shuffle(records)
        val_records.append((label, *records[0]))
        train_records.extend((label, name, windows) for name, windows in records[1:])

    # Fit scaling on training recordings only; use it unchanged for validation.
    train_min = min(float(w.min()) for _, _, w in train_records)
    train_max = max(float(w.max()) for _, _, w in train_records)
    if train_max <= train_min:
        raise RuntimeError("Training recordings have a constant value range.")

    def normalize(windows):
        return ((windows - train_min) / (train_max - train_min)).astype(np.float32)

    X_train = np.concatenate([normalize(w) for _, _, w in train_records], axis=0)
    y_train = np.concatenate([
        np.full(len(w), CLASSES.index(label), dtype=np.int64)
        for label, _, w in train_records
    ])
    X_val = np.concatenate([normalize(w) for _, _, w in val_records], axis=0)
    y_val = np.concatenate([
        np.full(len(w), CLASSES.index(label), dtype=np.int64)
        for label, _, w in val_records
    ])

    print("\nRecording split:")
    for label, name, _ in train_records:
        print(f"  TRAIN {label}: {name}")
    for label, name, _ in val_records:
        print(f"  VALID {label}: {name}")
    print(f"\nTrain windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")
    print(f"Training normalization range: {train_min:.6f} to {train_max:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = UT_HAR_CNN_LSTM(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).unsqueeze(1), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_x = torch.from_numpy(X_val).unsqueeze(1).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, best_epoch, stale_epochs = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x).argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()
        train_loss = total_loss / len(y_train)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"validation_window_accuracy={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": "CNN+BiLSTM",
            "num_classes": len(CLASSES),
            "input_shape": [1, 250, 90],
            "classes": CLASSES,
            "seed": SEED,
            "base_checkpoint": BASE_CHECKPOINT.name,
            "best_epoch": best_epoch,
            "validation_window_accuracy": best_val_acc,
            "train_min": train_min,
            "train_max": train_max,
            "train_recordings": [name for _, name, _ in train_records],
            "validation_recordings": [name for _, name, _ in val_records],
        },
        OUTPUT_CHECKPOINT,
    )
    print("\nTraining finished.")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation window accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved checkpoint: {OUTPUT_CHECKPOINT}")
    print("Validation uses only 5 recordings; treat the result as preliminary.")


if __name__ == "__main__":
    main()
