from pathlib import Path
import numpy as np
import torch
from UT_HAR_model import UT_HAR_CNN_LSTM

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_recordings"
CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"
TRAIN_DATA = ROOT / "Data" / "MyCSI" / "data" / "X_train.npy"
CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


def normalize_per_recording(windows):
    lo, hi = windows.min(), windows.max()
    if hi <= lo:
        return np.zeros_like(windows, dtype=np.float32)
    return ((windows - lo) / (hi - lo)).astype(np.float32)


def normalize_using_training_minmax(windows, train_min, train_max):
    # Same min/max constants as the original training split, without
    # refitting them on each new recording.
    if train_max <= train_min:
        raise ValueError("Training maximum must exceed training minimum.")
    return ((windows - train_min) / (train_max - train_min)).astype(np.float32)


def evaluate(model, normalization_name, normalize_fn, device):
    correct = total = 0
    per_class = {name: [0, 0] for name in CLASSES}

    with torch.no_grad():
        for label in CLASSES:
            class_dir = DATA_DIR / label
            if not class_dir.exists():
                print(f"WARNING: missing class folder: {class_dir}")
                continue

            for file in sorted(class_dir.glob("*.npy")):
                windows = np.load(file)
                if len(windows) == 0:
                    continue

                normalized = normalize_fn(windows)
                x = torch.from_numpy(normalized).unsqueeze(1).to(device)
                predictions = model(x).argmax(dim=1).cpu().numpy()

                hits = int(np.sum(predictions == CLASSES.index(label)))
                count = len(predictions)
                correct += hits
                total += count
                per_class[label][0] += hits
                per_class[label][1] += count

    print(f"\n=== {normalization_name} ===")
    print(f"Total windows: {total}")
    if total:
        print(f"Window accuracy: {100 * correct / total:.2f}%")
    for label, (hits, count) in per_class.items():
        if count:
            print(f"{label}: {hits}/{count} ({100 * hits / count:.2f}%)")
        else:
            print(f"{label}: no windows")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed recordings not found: {DATA_DIR}")
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(f"Training array not found: {TRAIN_DATA}")

    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = UT_HAR_CNN_LSTM(num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    training = np.load(TRAIN_DATA, mmap_mode="r")
    train_min = float(training.min())
    train_max = float(training.max())
    print(f"Training split min: {train_min:.8f}")
    print(f"Training split max: {train_max:.8f}")
    print("Comparing normalization only; checkpoint and processed recordings are unchanged.")

    evaluate(
        model,
        "Per-recording min-max (current method)",
        normalize_per_recording,
        device,
    )
    evaluate(
        model,
        "Training-split min-max (fixed training min/max)",
        lambda windows: normalize_using_training_minmax(windows, train_min, train_max),
        device,
    )


if __name__ == "__main__":
    main()
