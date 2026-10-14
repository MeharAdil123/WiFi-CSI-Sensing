
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from UT_HAR_model import UT_HAR_CNN_LSTM


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_new_test_recordings"
CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"

CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]

TRAIN_MIN = 1.0
TRAIN_MAX = 25.8827829587677


def evaluate_normalization(model, method):
    total_correct = 0
    total_windows = 0
    class_results = {
        label: {"correct": 0, "total": 0}
        for label in CLASSES
    }
    recording_results = []

    with torch.no_grad():
        for label in CLASSES:
            class_dir = DATA_DIR / label

            if not class_dir.exists():
                print(f"Missing activity folder: {label}")
                continue

            for file in sorted(class_dir.glob("*.npy")):
                raw_windows = np.load(file).astype(np.float32)

                if (
                    raw_windows.ndim != 3
                    or raw_windows.shape[1:] != (250, 90)
                    or len(raw_windows) == 0
                ):
                    print(f"Skipping invalid file: {file.name}")
                    continue

                windows = raw_windows.copy()

                if method == "fixed":
                    windows = (
                        windows - TRAIN_MIN
                    ) / (TRAIN_MAX - TRAIN_MIN)

                elif method == "per_recording":
                    minimum = windows.min()
                    maximum = windows.max()

                    if maximum > minimum:
                        windows = (
                            windows - minimum
                        ) / (maximum - minimum)
                    else:
                        windows.fill(0)

                x = torch.from_numpy(windows).unsqueeze(1)
                predictions = model(x).argmax(dim=1).cpu().numpy()

                expected = CLASSES.index(label)
                correct = int(np.sum(predictions == expected))
                count = len(predictions)

                total_correct += correct
                total_windows += count
                class_results[label]["correct"] += correct
                class_results[label]["total"] += count

                distribution = Counter(
                    CLASSES[int(p)] for p in predictions
                )
                majority_label = distribution.most_common(1)[0][0]

                recording_results.append(
                    (label, file.name, majority_label)
                )

    print(f"\n--- {method} normalization ---")
    print(f"Windows: {total_windows}")

    if total_windows:
        print(
            f"Window accuracy: "
            f"{100 * total_correct / total_windows:.2f}%"
        )

    print("\nPer-class window accuracy:")
    for label, result in class_results.items():
        count = result["total"]
        correct = result["correct"]

        if count:
            print(
                f"{label}: {correct}/{count} "
                f"({100 * correct / count:.2f}%)"
            )
        else:
            print(f"{label}: no windows")

    recording_correct = sum(
        predicted == actual
        for actual, _, predicted in recording_results
    )
    recording_total = len(recording_results)

    print("\nRecording-level results:")
    print(f"Recordings: {recording_total}")

    if recording_total:
        print(
            f"Majority-vote accuracy: "
            f"{100 * recording_correct / recording_total:.2f}%"
        )

    for actual, filename, predicted in recording_results:
        status = "CORRECT" if actual == predicted else "INCORRECT"
        print(
            f"{filename}: actual={actual}, "
            f"predicted={predicted}, {status}"
        )


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    device = torch.device("cpu")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = UT_HAR_CNN_LSTM(num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Checkpoint:", CHECKPOINT.name)
    print("Test data:", DATA_DIR)

    evaluate_normalization(model, "fixed")
    evaluate_normalization(model, "per_recording")


if __name__ == "__main__":
    main()
