
from pathlib import Path
import numpy as np
import torch
from UT_HAR_model import UT_HAR_CNN_LSTM

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_recordings"
CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"

CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )

    model = UT_HAR_CNN_LSTM(num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_counts = {name: [0, 0] for name in CLASSES}

    with torch.no_grad():
        for label in CLASSES:
            class_dir = DATA_DIR / label

            for file in sorted(class_dir.glob("*.npy")):
                windows = np.load(file)

                if len(windows) == 0:
                    print(f"SKIP: {file.name} has no windows")
                    continue

                # Apply min-max normalization separately to each recording,
                # matching the normalization style in dataset.py.
                # lo = windows.min()
                # hi = windows.max()

                # if hi > lo:
                #     windows = (windows - lo) / (hi - lo)
                # else:
                #     windows = np.zeros_like(windows)

                
                # Use the fixed min/max from the original training split.
                TRAIN_MIN = 1.0
                TRAIN_MAX = 25.8827829587677

                windows = (windows - TRAIN_MIN) / (TRAIN_MAX - TRAIN_MIN)



                # Model input: (batch, 1, 250, 90)
                x = torch.from_numpy(
                    windows.astype(np.float32)
                ).unsqueeze(1).to(device)

                logits = model(x)
                predictions = logits.argmax(dim=1).cpu().numpy()

                file_correct = int(np.sum(predictions == CLASSES.index(label)))
                file_total = len(predictions)

                correct += file_correct
                total += file_total
                class_counts[label][0] += file_correct
                class_counts[label][1] += file_total

                predicted_names, counts = np.unique(
                    [CLASSES[p] for p in predictions],
                    return_counts=True
                )
                distribution = dict(zip(predicted_names, counts))

                print(
                    f"{label}/{file.name}: "
                    f"correct={file_correct}/{file_total}, "
                    f"predictions={distribution}"
                )

    print("\n=== RESULTS ON GENERATED CSI WINDOWS ===")
    print(f"Total windows: {total}")

    if total:
        print(f"Window accuracy: {100 * correct / total:.2f}%")

    print("\nPer-class results:")
    for label, (hits, count) in class_counts.items():
        if count:
            print(f"{label}: {hits}/{count} ({100 * hits / count:.2f}%)")
        else:
            print(f"{label}: no windows")

    print(
        "\nWARNING: These results are preliminary. The CSI feature conversion "
        "and normalization have not been verified against training-time "
        "preprocessing, so this is not yet a trustworthy live-performance score."
    )


if __name__ == "__main__":
    main()
