import numpy as np
import os

base_path = "Data/MyCSI"

splits = ["train", "val", "test"]

for split in splits:
    path = os.path.join(
        base_path, "label", f"y_{split}.npy"
    )

    labels = np.load(path)

    print(f"\n--- {split.upper()} LABELS ---")
    print("Shape:", labels.shape)
    print("Unique labels:", np.unique(labels))
    print(
        "Label counts:",
        dict(zip(*np.unique(labels, return_counts=True)))
    )