import serial
import serial.tools.list_ports
import time

# Change this to the COM port shown on your computer.
COM_PORT = "COM8"

# Change this if your ESP32 firmware uses another baud rate.
BAUD_RATE = 921600


def list_ports():
    print("Available serial ports:")

    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        print(f"{port.device}: {port.description}")


def main():
    list_ports()

    print(f"\nConnecting to {COM_PORT}...")

    try:
        with serial.Serial(
            port=COM_PORT,
            baudrate=BAUD_RATE,
            timeout=2
        ) as ser:

            print("Connected. Waiting for serial data...")
            time.sleep(2)

            while True:
                line = ser.readline()

                if line:
                    text = line.decode(
                        "utf-8",
                        errors="replace"
                    ).strip()

                    if text:
                        print(text)

    except serial.SerialException as error:
        print("\nCould not open serial port.")
        print("Error:", error)
        print("Check the COM port, USB connection, and baud rate.")

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()