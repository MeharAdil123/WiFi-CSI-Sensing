"""
Compare the original preprocess.py padding/truncation method against the
current interpolation pipeline using the same checkpoint.

Writes only to processed_recordings_padded/. Existing recordings, processed data,
and checkpoint are not modified. Results are diagnostic, not a validated score.
"""
import ast
import csv
from pathlib import Path
import numpy as np
import torch
from UT_HAR_model import UT_HAR_CNN_LSTM

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
OUTPUT_DIR = ROOT / "processed_recordings_padded"
CHECKPOINT = ROOT / "checkpoints" / "cnn_bilstm_seed_456.pth"
WINDOW_SIZE, STRIDE, TARGET_FEATURES = 250, 50, 90
EXPECTED_FIELDS, EXPECTED_CSI_VALUES = 25, 384
CLASSES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


def label_from_filename(filename):
    prefix = filename.split("_")[0].lower()
    for label in CLASSES:
        if prefix == label.lower():
            return label
    raise ValueError(f"Unknown class in filename: {filename}")


def packet_to_features(raw_csi, filename, row_number):
    values = ast.literal_eval(raw_csi.strip())
    csi = np.asarray(values, dtype=np.float32)
    if csi.ndim != 1 or csi.size != EXPECTED_CSI_VALUES:
        raise ValueError(f"{filename}, row {row_number}: expected 384 CSI values")
    if not np.isfinite(csi).all():
        raise ValueError(f"{filename}, row {row_number}: non-finite CSI values")

    real, imag = csi[::2], csi[1::2]
    amplitude = np.sqrt(real ** 2 + imag ** 2)
    amplitude = amplitude[amplitude > 0]

    # Match active preprocess.py: keep first 90 values or pad remaining slots with zero.
    if len(amplitude) >= TARGET_FEATURES:
        return amplitude[:TARGET_FEATURES].astype(np.float32)
    result = np.zeros(TARGET_FEATURES, dtype=np.float32)
    result[:len(amplitude)] = amplitude
    return result


def read_recording(path):
    packets = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row_number, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) != EXPECTED_FIELDS or row[0].strip() != "CSI_DATA":
                raise ValueError(f"{path.name}, row {row_number}: invalid CSV structure")
            packets.append(packet_to_features(row[-1], path.name, row_number))
    if not packets:
        raise ValueError(f"No packets found in {path.name}")
    return np.stack(packets).astype(np.float32)


def make_windows(packets):
    if len(packets) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, TARGET_FEATURES), dtype=np.float32)
    return np.stack([
        packets[start:start + WINDOW_SIZE]
        for start in range(0, len(packets) - WINDOW_SIZE + 1, STRIDE)
    ]).astype(np.float32)


def main():
    if not RECORDINGS_DIR.is_dir():
        raise FileNotFoundError(f"Missing recordings folder: {RECORDINGS_DIR}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    files = sorted(RECORDINGS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {RECORDINGS_DIR}")

    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = UT_HAR_CNN_LSTM(num_classes=5)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint has no model state. Keys: {list(checkpoint.keys())}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {label: [0, 0] for label in CLASSES}
    total_packets = total_windows = 0

    with torch.no_grad():
        for path in files:
            label = label_from_filename(path.name)
            packets = read_recording(path)
            windows = make_windows(packets)
            out_dir = OUTPUT_DIR / label
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{path.stem}.npy", windows)
            total_packets += len(packets)
            total_windows += len(windows)
            if len(windows) == 0:
                print(f"{path.name}: no complete windows")
                continue

            # Keep the same per-recording min-max normalization as the prior test.
            lo, hi = windows.min(), windows.max()
            normalized = (windows - lo) / (hi - lo) if hi > lo else np.zeros_like(windows)
            x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(1)
            predictions = model(x).argmax(dim=1).cpu().numpy()
            correct = int(np.sum(predictions == CLASSES.index(label)))
            totals[label][0] += correct
            totals[label][1] += len(predictions)
            unique, counts = np.unique(
                [CLASSES[int(pred)] for pred in predictions], return_counts=True
            )
            print(f"{label}/{path.name}: correct={correct}/{len(predictions)}, "
                  f"predictions={dict(zip(unique.tolist(), counts.tolist()))}")

    correct_all = sum(v[0] for v in totals.values())
    count_all = sum(v[1] for v in totals.values())
    print("\n=== PADDING/TRUNCATION PIPELINE SUMMARY ===")
    print(f"Recordings processed: {len(files)}")
    print(f"Packets processed: {total_packets}")
    print(f"Windows evaluated: {total_windows}")
    print(f"Overall accuracy: {100 * correct_all / count_all:.2f}%" if count_all else "No windows evaluated")
    print("\nPer-class results:")
    for label in CLASSES:
        hits, count = totals[label]
        print(f"{label}: {hits}/{count} ({100 * hits / count:.2f}%)" if count else f"{label}: no windows")
    print(f"\nSaved comparison dataset to: {OUTPUT_DIR}")
    print("Diagnostic comparison only; this is not a validated independent test score.")


if __name__ == "__main__":
    main()
