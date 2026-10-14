
import ast
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "live_recordings"
OUTPUT_DIR = ROOT / "processed_new_test_recordings"

WINDOW_SIZE = 250
STRIDE = 50
EXPECTED_CSI_VALUES = 384
EXPECTED_FIELDS = 25
TARGET_FEATURES = 90

ACTIVITIES = {
    "empty": "Empty",
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
    "falling": "Falling",
}

skipped_packets = 0


def get_activity(path):
    match = re.match(r"([A-Za-z]+)", path.stem)

    if not match or match.group(1).lower() not in ACTIVITIES:
        raise ValueError(
            f"Cannot identify activity from filename: {path.name}"
        )

    return ACTIVITIES[match.group(1).lower()]


def parse_csi_values(raw):
    global skipped_packets

    try:
        values = np.asarray(
            ast.literal_eval(raw.strip()),
            dtype=np.float32
        )
    except (ValueError, SyntaxError, TypeError):
        skipped_packets += 1
        return None

    if (
        values.ndim != 1
        or len(values) != EXPECTED_CSI_VALUES
        or not np.isfinite(values).all()
    ):
        skipped_packets += 1
        return None

    return values


def parse_csi_line(line):
    try:
        fields = next(csv.reader([line]))

        if (
            len(fields) != EXPECTED_FIELDS
            or fields[0].strip() != "CSI_DATA"
        ):
            return None

        return parse_csi_values(fields[24])

    except (csv.Error, IndexError, StopIteration):
        return None


def csi_to_features(csi):
    real = csi[0::2]
    imag = csi[1::2]

    magnitudes = np.sqrt(real**2 + imag**2)
    magnitudes = magnitudes[magnitudes > 0]

    if len(magnitudes) < 2:
        return None

    old_axis = np.linspace(0, 1, len(magnitudes))
    new_axis = np.linspace(0, 1, TARGET_FEATURES)

    return np.interp(
        new_axis,
        old_axis,
        magnitudes
    ).astype(np.float32)


def read_recording(path):
    frames = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)
        first_row = next(reader, None)

        if first_row is None:
            raise ValueError(f"{path.name}: file is empty")

        # Format 1: original 25-column CSI CSV.
        if first_row[0].strip() == "CSI_DATA":

            rows = enumerate(
                [first_row, *reader],
                start=1
            )

            for row_number, row in rows:
                if (
                    len(row) != EXPECTED_FIELDS
                    or row[0].strip() != "CSI_DATA"
                ):
                    global skipped_packets
                    skipped_packets += 1
                    continue

                csi = parse_csi_values(row[24])

                if csi is None:
                    continue

                features = csi_to_features(csi)

                if features is not None:
                    frames.append(features)

        # Format 2: activity,timestamp,raw_csi_line.
        elif (
            len(first_row) >= 3
            and first_row[0].strip().lower() == "activity"
        ):

            for row_number, row in enumerate(reader, start=2):
                if not row:
                    continue

                if len(row) != 3:
                    skipped_packets += 1
                    continue

                csi = parse_csi_line(row[2])

                if csi is None:
                    continue

                features = csi_to_features(csi)

                if features is not None:
                    frames.append(features)

        else:
            raise ValueError(
                f"{path.name}: unrecognized CSV format"
            )

    if not frames:
        raise ValueError(
            f"{path.name}: no valid CSI packets were found"
        )

    return np.stack(frames).astype(np.float32)


def make_windows(frames):
    if len(frames) < WINDOW_SIZE:
        return np.empty(
            (0, WINDOW_SIZE, TARGET_FEATURES),
            dtype=np.float32
        )

    starts = range(
        0,
        len(frames) - WINDOW_SIZE + 1,
        STRIDE
    )

    return np.stack([
        frames[start:start + WINDOW_SIZE]
        for start in starts
    ]).astype(np.float32)


def main():
    global skipped_packets

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_DIR}"
        )

    paths = sorted(INPUT_DIR.glob("*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No CSV recordings found in {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_packets = 0
    total_windows = 0
    successful = 0
    failed = 0

    for path in paths:
        try:
            activity = get_activity(path)
            frames = read_recording(path)
            windows = make_windows(frames)

            output_folder = OUTPUT_DIR / activity
            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = output_folder / f"{path.stem}.npy"
            np.save(output_path, windows)

            total_packets += len(frames)
            total_windows += len(windows)
            successful += 1

            print(
                f"{path.name}: activity={activity}, "
                f"valid_packets={len(frames)}, "
                f"windows={len(windows)}, "
                f"saved={output_path.relative_to(ROOT)}"
            )

        except (ValueError, OSError, csv.Error) as exc:
            failed += 1
            print(f"FAILED: {path.name}: {exc}")

    print("\n========== SUMMARY ==========")
    print(f"CSV files found: {len(paths)}")
    print(f"Recordings processed: {successful}")
    print(f"Recordings failed: {failed}")
    print(f"Valid packets retained: {total_packets}")
    print(f"Malformed CSI packets skipped: {skipped_packets}")
    print(f"Windows created: {total_windows}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original training recordings were not modified.")
    print("================================")


if __name__ == "__main__":
    main()
