
import csv
import time
from datetime import datetime
from pathlib import Path

import serial

COM_PORT = "COM8"
BAUD_RATE = 921600
ACTIVITIES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


def main():
    print("Choose the activity being recorded:")
    for index, name in enumerate(ACTIVITIES):
        print(f"{index}: {name}")

    choice = input("Activity number: ").strip()
    if not choice.isdigit() or int(choice) not in range(len(ACTIVITIES)):
        print("Invalid activity number.")
        return

    activity = ACTIVITIES[int(choice)]
    duration_text = input("Recording duration in seconds (e.g. 60): ").strip()

    try:
        duration = int(duration_text)
        if duration <= 0:
            raise ValueError
    except ValueError:
        print("Enter a positive whole number of seconds.")
        return

    output_dir = Path("live_recordings")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{activity}_{timestamp}.csv"

    count = 0
    deadline = time.time() + duration

    print(f"Recording {activity} for {duration} seconds.")
    print("Keep the selected activity consistent throughout the recording.")

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser, \
             output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["activity", "timestamp", "raw_csi_line"])

            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("CSI_DATA,"):
                    continue

                writer.writerow([activity, time.time(), line])
                count += 1

                if count % 100 == 0:
                    print(f"Captured {count} CSI records...")

    except serial.SerialException as exc:
        print("Serial port error:", exc)
        return

    print(f"Saved {count} records to: {output_path}")


if __name__ == "__main__":
    main()
