# data_saver.py
import csv
import time

class DataSaver:
    def __init__(self, filename="sensor_log.csv"):
        self.filename = filename
        self.file = open(self.filename, mode="a", newline="")
        self.writer = csv.writer(self.file)

        # Write header only if file is empty
        if self.file.tell() == 0:
            self.writer.writerow([
                "timestamp",
                "voltage",
                "current",
                "power",
                "energy",
                "frequency",
                "power_factor"
            ])

        print(f"[DataSaver] Initialized CSV file: {self.filename}")

    def save(self, voltage, current, power, energy, frequency, pf):
        timestamp = time.time()
        self.writer.writerow([timestamp, voltage, current, power, energy, frequency, pf])
        self.file.flush()  # ensure immediate write
        print(f"[DataSaver] Saved row: {voltage}, {current}, {power}")

    def __del__(self):
        try:
            self.file.close()
            print("[DataSaver] CSV file closed.")
        except Exception:
            pass
