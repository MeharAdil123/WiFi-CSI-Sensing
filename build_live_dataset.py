"""
Convert ESP32-S3 CSI receiver CSV recordings into model-ready windows.

Input: recordings/*.csv
Output: processed_recordings/<Class>/<recording_stem>.npy, each shaped (N, 250, 90).

This is a candidate pipeline: adjacent CSI values are treated as I/Q pairs, converted
to magnitudes, and resampled from 192 values to 90 features. Confirm this matches
the preprocessing used by the existing checkpoint before relying on its predictions.
"""

import ast
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
OUTPUT_DIR = ROOT / "processed_recordings"
WINDOW_SIZE = 250
STRIDE = 50
CSI_FIELD_INDEX = 24
EXPECTED_FIELDS = 25
EXPECTED_CSI_VALUES = 384
TARGET_FEATURES = 90

LABEL_ALIASES = {
    "empty": "Empty",
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
    "falling": "Falling",
}


def class_from_filename(path):
    match = re.match(r"([A-Za-z]+)", path.stem)
    if not match or match.group(1).lower() not in LABEL_ALIASES:
        raise ValueError(f"Cannot infer activity label from filename: {path.name}")
    return LABEL_ALIASES[match.group(1).lower()]


def parse_csi_list(raw, path, row_number):
    try:
        values = ast.literal_eval(raw.strip())
        array = np.asarray(values, dtype=np.float32)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}, row {row_number}: invalid CSI list") from exc
    if array.ndim != 1 or array.size != EXPECTED_CSI_VALUES:
        raise ValueError(
            f"{path.name}, row {row_number}: expected {EXPECTED_CSI_VALUES} "
            f"values, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}, row {row_number}: non-finite CSI values")
    return array


# Count packets that cannot be interpolated after zero-amplitude removal.
SKIPPED_LOW_AMPLITUDE_PACKETS = 0


def csi_to_features(csi):
    # Match the original interpolate_packets.py preprocessing.
    global SKIPPED_LOW_AMPLITUDE_PACKETS

    real = csi[0::2]
    imag = csi[1::2]

    magnitudes = np.sqrt(real**2 + imag**2)

    # The original pipeline removes zero amplitudes before interpolation.
    magnitudes = magnitudes[magnitudes > 0]

    if len(magnitudes) < 2:
        SKIPPED_LOW_AMPLITUDE_PACKETS += 1
        return None

    old_axis = np.linspace(0, 1, len(magnitudes))
    new_axis = np.linspace(0, 1, TARGET_FEATURES)

    return np.interp(
        new_axis, old_axis, magnitudes
    ).astype(np.float32)


def read_recording(path):
    frames = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if len(row) != EXPECTED_FIELDS:
                raise ValueError(
                    f"{path.name}, row {row_number}: expected {EXPECTED_FIELDS} "
                    f"fields, got {len(row)}"
                )
            if row[0].strip() != "CSI_DATA":
                raise ValueError(
                    f"{path.name}, row {row_number}: expected CSI_DATA marker"
                )
            csi = parse_csi_list(row[CSI_FIELD_INDEX], path, row_number)
            features = csi_to_features(csi)
            # Match the original preprocessing: skip packets with fewer than
            # two nonzero amplitudes instead of inserting an artificial zero row.
            if features is not None:
                frames.append(features)
    if not frames:
        raise ValueError(f"{path.name}: no CSI packets found")
    return np.stack(frames).astype(np.float32)


def make_windows(frames):
    if len(frames) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, TARGET_FEATURES), dtype=np.float32)
    starts = range(0, len(frames) - WINDOW_SIZE + 1, STRIDE)
    return np.stack([frames[s:s + WINDOW_SIZE] for s in starts]).astype(np.float32)


def main():
    if not RECORDINGS_DIR.is_dir():
        raise FileNotFoundError(f"Recordings folder not found: {RECORDINGS_DIR}")
    paths = sorted(RECORDINGS_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV recordings found in {RECORDINGS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_packets = total_windows = 0
    for path in paths:
        label = class_from_filename(path)
        frames = read_recording(path)
        windows = make_windows(frames)
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.npy"
        np.save(out_path, windows)
        total_packets += len(frames)
        total_windows += len(windows)
        print(
            f"{path.name}: label={label}, packets={len(frames)}, "
            f"windows={len(windows)}, saved={out_path.relative_to(ROOT)}"
        )

    print("\n=== SUMMARY ===")
    print(f"Recordings processed: {len(paths)}")
    print(f"Packets retained after feature extraction: {total_packets}")
    print(f"Packets skipped (fewer than 2 nonzero amplitudes): {SKIPPED_LOW_AMPLITUDE_PACKETS}")
    print(f"Windows created: {total_windows}")
    print(f"Output: {OUTPUT_DIR}")
    print("Each recording remains a separate file to prevent window-level data leakage.")
    print("Confirm feature extraction matches training preprocessing before inference.")


if __name__ == "__main__":
    main()
