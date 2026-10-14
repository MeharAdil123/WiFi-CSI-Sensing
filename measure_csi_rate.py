
import csv
import io
import time
import serial

COM_PORT = "COM8"
BAUD_RATE = 921600
DURATION_SECONDS = 10

count = 0
start = time.time()

print(f"Measuring CSI records on {COM_PORT} for {DURATION_SECONDS} seconds...")

try:
    with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
        while time.time() - start < DURATION_SECONDS:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("CSI_DATA,"):
                continue

            try:
                row = next(csv.reader(io.StringIO(line)))
                if len(row) >= 25:
                    count += 1
            except (csv.Error, StopIteration):
                pass

except serial.SerialException as exc:
    print("Serial port error:", exc)
    raise SystemExit(1)

elapsed = time.time() - start

print("Valid CSI records:", count)
print("Elapsed seconds:", round(elapsed, 2))

if count:
    rate = count / elapsed
    print("Observed rate:", round(rate, 2), "records/second")
    print("Estimated time for 250 records:", round(250 / rate, 2), "seconds")
else:
    print("No valid CSI records received. Check the receiver and COM port.")
