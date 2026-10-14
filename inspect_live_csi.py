
import csv
import io
import serial

COM_PORT = "COM8"
BAUD_RATE = 921600
MAX_RECORDS = 10


def main():
    count = 0

    with serial.Serial(COM_PORT, BAUD_RATE, timeout=3) as ser:
        print(f"Listening on {COM_PORT} at {BAUD_RATE} baud...")

        try:
            while count < MAX_RECORDS:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("CSI_DATA,"):
                    continue

                try:
                    row = next(csv.reader(io.StringIO(line)))

                    # The ESP32-S3 CSV has 23 fields before the CSI array.
                    if len(row) < 24:
                        print(f"Record skipped: only {len(row)} CSV fields")
                        continue

                    csi_text = row[-1].strip().strip('"').strip("[]")
                    values = [
                        value.strip()
                        for value in csi_text.split(",")
                        if value.strip()
                    ]
                    csi_values = [int(value) for value in values]

                    count += 1
                    print(
                        f"Record {count}: CSV fields={len(row)}, "
                        f"CSI integers={len(csi_values)}, "
                        f"RSSI={row[3]}, channel={row[16]}, "
                        f"sig_len={row[20]}, CSI len={row[22]}"
                    )

                except (ValueError, csv.Error, IndexError, StopIteration) as exc:
                    print("Could not parse record:", exc)

        except KeyboardInterrupt:
            print("\nStopped by user.")

    print(f"Finished. Parsed {count} records.")


if __name__ == "__main__":
    main()
