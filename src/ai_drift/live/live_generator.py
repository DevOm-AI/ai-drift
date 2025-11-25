import os
import csv
import json
import time
import random
import threading
from datetime import datetime
import pandas as pd
from pathlib import Path

from ..monitoring.drift_detector import DriftDetector
from .drift_patterns import PATTERNS

# Path for live artifacts
LIVE_FOLDER = "artifacts/live_stream"
LOG_ROW_JSON = os.path.join(LIVE_FOLDER, "live_row.json")
BATCH_CSV = os.path.join(LIVE_FOLDER, "live_batch.csv")
REPORTS_FOLDER = "artifacts/reports"

# default header logo path (not required)
LOGO_PATH = "/mnt/data/21623917-d285-44b7-8c1e-9814b3f3e67f.png"

class LiveGenerator:
    """
    Generates streaming rows based on a base template (either from reference.csv or synthetic),
    applies active patterns, writes a single JSON row and appends to a CSV batch file.
    """

    def __init__(self, reference_csv="data/reference.csv", interval=2.0, batch_size=1, detector_cls=DriftDetector):
        self.reference_csv = reference_csv
        self.interval = float(interval)
        self.batch_size = int(batch_size)
        self.active_patterns = []  # list of pattern names (strings)
        self._stop = False
        self._lock = threading.Lock()
        self._thread = None

        # load a small dataframe to sample base values
        if Path(reference_csv).exists():
            self.ref_df = pd.read_csv(reference_csv)
        else:
            # fallback: create a tiny synthetic df
            self.ref_df = pd.DataFrame([{
                'age': 35, 'gender': 'male', 'geo_city': 'Mumbai', 'membership_years': 2,
                'session_duration': 300, 'page_views': 4, 'device_type': 'mobile',
                'cart_value': 120.0, 'spend_score': 60.0, 'income': 50000.0,
                'purchased': 0, 'payment_method': 'Card'
            }])

        os.makedirs(LIVE_FOLDER, exist_ok=True)
        os.makedirs(REPORTS_FOLDER, exist_ok=True)

        # ensure CSV header exists
        if not os.path.exists(BATCH_CSV):
            self._write_header()

    def _write_header(self):
        cols = ['timestamp','age','gender','geo_city','membership_years','session_duration','page_views','device_type','cart_value','spend_score','income','purchased','payment_method']
        with open(BATCH_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)

    def sample_base_row(self):
        # sample a row from reference or create one with reasonable noise
        row = self.ref_df.sample(1).to_dict(orient='records')[0]
        # extend with additional fields if missing
        defaults = {
            'gender': random.choice(['male','female','other']),
            'geo_city': random.choice(['Mumbai','Pune','Delhi','Bangalore']),
            'device_type': random.choice(['mobile','desktop','tablet']),
            'session_duration': float(max(1, random.gauss(300, 80))),
            'page_views': int(max(1, int(abs(random.gauss(4, 2))))),
            'cart_value': float(abs(random.gauss(120, 90))),
            'membership_years': float(abs(random.gauss(2, 2))),
            'payment_method': random.choice(['UPI','Card','COD','Netbanking'])
        }
        for k,v in defaults.items():
            if k not in row or pd.isna(row.get(k)):
                row[k] = v
        # ensure proper types
        row['age'] = int(max(16, int(row.get('age', 35))))
        row['purchased'] = int(row.get('purchased', 0))
        return row

    def apply_patterns(self, row):
        # apply active patterns in order
        for pat in list(self.active_patterns):
            fn = PATTERNS.get(pat)
            if fn:
                try:
                    row = fn(row)
                except TypeError:
                    # some patterns expect default args - call directly
                    row = fn(row)
        return row

    def _emit_row(self, row):
        # write JSON of last row
        with open(LOG_ROW_JSON, 'w', encoding='utf-8') as f:
            json.dump(row, f, default=str, indent=2)
        # append to CSV
        cols = ['timestamp','age','gender','geo_city','membership_years','session_duration','page_views','device_type','cart_value','spend_score','income','purchased','payment_method']
        with open(BATCH_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([row.get(c, "") for c in cols])

    def generate_once(self):
        # generate one batch (batch_size rows)
        for _ in range(self.batch_size):
            row = self.sample_base_row()
            row['timestamp'] = datetime.utcnow().isoformat()
            row = self.apply_patterns(row)
            # basic sanity fixes
            row['purchased'] = int(min(1, int(row.get('purchased', 0))))
            self._emit_row(row)

    def start(self):
        self._stop = False
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while not self._stop:
            with self._lock:
                self.generate_once()
            time.sleep(self.interval)

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)

    # control API for patterns
    def add_pattern(self, name):
        if name not in PATTERNS:
            raise KeyError(f"Unknown pattern: {name}")
        if name not in self.active_patterns:
            self.active_patterns.append(name)

    def remove_pattern(self, name):
        if name in self.active_patterns:
            self.active_patterns.remove(name)

    def list_patterns(self):
        return list(PATTERNS.keys())

    def list_active(self):
        return list(self.active_patterns)