
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from UT_HAR_model import UT_HAR_CNN_LSTM

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_new_test_recordings"
CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"

CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]

# Same fixed normalization values as test_recorded_csi.py
TRAIN_MIN = 1.0
TRAIN_MAX = 25.8827829587677


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test data folder not found: {DATA_DIR}"
        )

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model = UT_HAR_CNN_LSTM(num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
                windows = np.load(file)

                if windows.ndim != 3 or windows.shape[1:] != (250, 90):
                    print(
                        f"SKIP {file.name}: unexpected shape "
                        f"{windows.shape}"
                    )
                    continue

                if len(windows) == 0:
                    print(f"SKIP {file.name}: no windows")
                    continue

                # Apply the same fixed normalization as the
                # existing test_recorded_csi.py script.
                windows = (
                    windows - TRAIN_MIN
                ) / (TRAIN_MAX - TRAIN_MIN)

                x = torch.from_numpy(
                    windows.astype(np.float32)
                ).unsqueeze(1).to(device)

                logits = model(x)
                predictions = logits.argmax(dim=1).cpu().numpy()

                expected_index = CLASSES.index(label)
                correct = int(np.sum(predictions == expected_index))
                count = len(predictions)

                total_correct += correct
                total_windows += count

                class_results[label]["correct"] += correct
                class_results[label]["total"] += count

                distribution = Counter(
                    CLASSES[p] for p in predictions
                )

                # One majority-vote prediction per recording.
                majority_label, majority_count = distribution.most_common(1)[0]
                recording_correct = majority_label == label

                recording_results.append(
                    (label, file.name, majority_label, recording_correct)
                )

                print(
                    f"{label}/{file.name}: "
                    f"windows={correct}/{count} correct, "
                    f"majority_prediction={majority_label}, "
                    f"distribution={dict(distribution)}"
                )

    print("\n========== WINDOW-LEVEL RESULTS ==========")
    print(f"Total windows: {total_windows}")

    if total_windows:
        print(
            f"Window accuracy: "
            f"{100 * total_correct / total_windows:.2f}%"
        )

    print("\nPer-activity window accuracy:")
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

    print("\n========== RECORDING-LEVEL RESULTS ==========")
    correct_recordings = sum(
        result[3] for result in recording_results
    )
    total_recordings = len(recording_results)

    print(f"Recordings evaluated: {total_recordings}")

    if total_recordings:
        print(
            f"Recording accuracy: "
            f"{100 * correct_recordings / total_recordings:.2f}%"
        )

    for label, filename, prediction, correct in recording_results:
        status = "CORRECT" if correct else "INCORRECT"
        print(
            f"{filename}: actual={label}, "
            f"predicted={prediction}, {status}"
        )

    print("\nNOTE: These are preliminary results. The feature conversion")
    print("and normalization still need verification against the")
    print("checkpoint's exact training preprocessing.")


if __name__ == "__main__":
    main()
