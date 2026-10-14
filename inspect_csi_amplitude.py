
import csv
import io
import time

import numpy as np
import serial

COM_PORT = "COM8"
BAUD_RATE = 921600
RECORDS_TO_CHECK = 10


def main():
    count = 0

    with serial.Serial(COM_PORT, BAUD_RATE, timeout=3) as ser:
        print(f"Reading CSI from {COM_PORT} at {BAUD_RATE} baud...")

        deadline = time.time() + 30

        while count < RECORDS_TO_CHECK and time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("CSI_DATA,"):
                continue

            try:
                row = next(csv.reader(io.StringIO(line)))
                if len(row) != 25:
                    print("Skipping unexpected CSV field count:", len(row))
                    continue

                values = np.asarray(
                    [int(v) for v in row[-1].strip().strip('"').strip("[]").split(",")],
                    dtype=np.float32
                )

                if len(values) != 384:
                    print("Skipping unexpected CSI length:", len(values))
                    continue

                # Candidate interpretation: alternating I and Q components.
                i_values = values[0::2]
                q_values = values[1::2]
                amplitude = np.sqrt(i_values**2 + q_values**2)

                count += 1
                print(
                    f"Record {count}: "
                    f"components={len(values)}, "
                    f"candidate magnitudes={len(amplitude)}, "
                    f"min={amplitude.min():.2f}, "
                    f"max={amplitude.max():.2f}, "
                    f"mean={amplitude.mean():.2f}"
                )

            except (ValueError, csv.Error, IndexError) as exc:
                print("Skipping malformed record:", exc)

    print(f"Diagnostic complete: {count} records inspected.")


if __name__ == "__main__":
    main()
