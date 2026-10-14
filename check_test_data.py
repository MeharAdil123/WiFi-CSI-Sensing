
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

FOLDERS = {
    "Original processed data": ROOT / "processed_recordings",
    "New ESP32 test data": ROOT / "processed_new_test_recordings",
}


def main():
    for title, folder in FOLDERS.items():
        print(f"\n===== {title} =====")

        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        files = sorted(folder.glob("*/*.npy"))

        if not files:
            print("No .npy files found.")
            continue

        arrays = []

        for file in files:
            data = np.load(file)

            if data.size:
                arrays.append(data.reshape(-1))

        if not arrays:
            print("No windows available.")
            continue

        values = np.concatenate(arrays)

        print(f"Files: {len(files)}")
        print(f"Values: {values.size}")
        print(f"Minimum: {values.min():.6f}")
        print(f"Maximum: {values.max():.6f}")
        print(f"Mean: {values.mean():.6f}")
        print(f"Standard deviation: {values.std():.6f}")

        normalized = (values - 1.0) / (
            25.8827829587677 - 1.0
        )

        print(
            "After fixed normalization — "
            f"min: {normalized.min():.6f}, "
            f"max: {normalized.max():.6f}, "
            f"mean: {normalized.mean():.6f}"
        )

        print(
            "Fraction outside [0, 1]: "
            f"{100 * np.mean((normalized < 0) | (normalized > 1)):.2f}%"
        )


if __name__ == "__main__":
    main()
