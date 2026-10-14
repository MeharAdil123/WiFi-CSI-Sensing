
import csv
import io
import time

import numpy as np
import serial

COM_PORT = "COM8"
BAUD_RATE = 921600
RECORDS_TO_CHECK = 10


def parse_amplitude(line):
    row = next(csv.reader(io.StringIO(line)))

    if len(row) != 25:
        return None

    csi_text = row[-1].strip().strip('"').strip("[]")
    values = np.asarray(
        [int(v.strip()) for v in csi_text.split(",") if v.strip()],
        dtype=np.float32,
    )

    if values.size != 384 or values.size % 2:
        return None

    # Candidate I/Q interpretation; verify before production use.
    i_values = values[0::2]
    q_values = values[1::2]
    amplitude = np.hypot(i_values, q_values)

    # Resample the candidate amplitude vector to 90 positions.
    source_x = np.linspace(0.0, 1.0, amplitude.size)
    target_x = np.linspace(0.0, 1.0, 90)
    features = np.interp(target_x, source_x, amplitude)

    return features


def main():
    count = 0
    deadline = time.time() + 30

    print(f"Reading live CSI from {COM_PORT} at {BAUD_RATE} baud...")

    with serial.Serial(COM_PORT, BAUD_RATE, timeout=2) as ser:
        while count < RECORDS_TO_CHECK and time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("CSI_DATA,"):
                continue

            try:
                features = parse_amplitude(line)
                if features is None:
                    print("Skipped unexpected CSI record.")
                    continue

                count += 1
                print(
                    f"Record {count}: shape={features.shape}, "
                    f"min={features.min():.2f}, "
                    f"max={features.max():.2f}, "
                    f"mean={features.mean():.2f}"
                )
                print("First 10 features:", np.round(features[:10], 2))

            except (ValueError, csv.Error, IndexError, StopIteration) as exc:
                print("Could not parse record:", exc)

    print(f"Preview complete: {count} records processed.")


if __name__ == "__main__":
    main()
