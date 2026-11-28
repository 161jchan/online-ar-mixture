"""Cached ETTh1 (seven channels) and monthly SILSO V2.0 streams."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SOURCES = {
    "etth1": ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
              "ETDataset ETTh1; downloaded bytes pinned by SHA-256"),
    "sunspots": ("https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
                 "WDC-SILSO, Royal Observatory of Belgium, Brussels; V2.0; "
                 "https://doi.org/10.24414/qnza-ac80; CC BY-NC 4.0"),
}
ETT_CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_fill(values):
    """Forward-fill each channel; use zero until its first observed value.

    The returned mask marks actual finite observations. Filled values may enter
    lag histories, but targets lacking any channel are excluded from row losses.
    No interpolation, backward fill, or full-series statistic is used.
    """
    x = np.asarray(values, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Expected a (time,channel) array")
    observed = np.isfinite(x)
    previous = np.zeros(x.shape[1])
    for t in range(len(x)):
        x[t] = np.where(observed[t], x[t], previous)
        previous = x[t].copy()
    return x, observed


def _number(value, *, sunspot=False):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return np.nan
    return np.nan if sunspot and result == -1 else result


def parse_dataset(name, text):
    """Keep source order; reject duplicates/out-of-order timestamps.

    Hourly ETT timestamps are timezone-naive labels exactly as supplied.
    Sunspots use YYYY-MM-01 as a monthly label, not an observation day.
    Missing grid timestamps are inserted and causally filled like missing cells.
    SILSO documents February 1824 as interpolated from adjacent months; mask
    that upstream value as missing before applying our past-only fill.
    """
    if name == "etth1":
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != ["date", *ETT_CHANNELS]:
            raise ValueError("Unexpected ETTh1 columns")
        records = [(datetime.fromisoformat(row["date"]),
                    [_number(row[c]) for c in ETT_CHANNELS]) for row in reader]
        channels = list(ETT_CHANNELS)
        advance = lambda d: d + timedelta(hours=1)
    elif name == "sunspots":
        records = []
        for row in csv.reader(io.StringIO(text), delimiter=";"):
            if not row or not row[0].strip():
                continue
            records.append((datetime(int(row[0]), int(row[1]), 1),
                            [np.nan if (int(row[0]), int(row[1])) == (1824, 2)
                             else _number(row[3], sunspot=True)]))
        channels = ["monthly_mean_total_sunspot_number"]
        advance = lambda d: datetime(d.year+(d.month == 12), d.month % 12+1, 1)
    else:
        raise ValueError("Choose etth1 or sunspots")
    if not records:
        raise ValueError("Empty dataset")
    stamps, rows = [], []
    for stamp, row in records:
        if stamps and stamp <= stamps[-1]:
            raise ValueError("Timestamps must be strictly increasing and unique")
        if name == "etth1" and (stamp.minute or stamp.second or stamp.microsecond):
            raise ValueError("ETTh1 timestamp is off the hourly grid")
        while stamps and advance(stamps[-1]) < stamp:
            stamps.append(advance(stamps[-1]))
            rows.append([np.nan]*len(channels))
        if stamps and advance(stamps[-1]) != stamp:
            raise ValueError("Timestamp is off the expected grid")
        stamps.append(stamp)
        rows.append(row)
    values, observed = causal_fill(rows)
    timestamps = np.array([d.isoformat() for d in stamps])
    return values, timestamps, observed, channels


def load_real(name, cache_dir=Path("data/cache"), *, download=False):
    """No network access on cache hits; reject changed bytes/provenance."""
    cache_dir = Path(cache_dir)
    path, manifest = cache_dir/f"{name}.csv", cache_dir/f"{name}.json"
    if name not in SOURCES:
        raise ValueError("Choose etth1 or sunspots")
    url, credit = SOURCES[name]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Run python -m data.real --download to cache {name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(Request(url, headers={"User-Agent": "ECE685D-academic-project"}), timeout=60) as response:
            payload = response.read()
            headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified")}
            final_url = response.geturl()
        # Parse before accepting the download.
        parse_dataset(name, payload.decode("utf-8-sig"))
        path.write_bytes(payload)
        meta = dict(dataset=name, source_url=url, resolved_url=final_url,
                    source_credit=credit, sha256=sha256(path), bytes=len(payload),
                    retrieved_utc=datetime.now(timezone.utc).isoformat(), headers=headers)
        manifest.write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    if not manifest.exists():
        raise ValueError(f"Missing provenance manifest: {manifest}")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256(path) != meta["sha256"]:
        raise ValueError(f"Cached bytes do not match recorded SHA-256: {path}")
    values, timestamps, observed, channels = parse_dataset(name, path.read_text(encoding="utf-8-sig"))
    return values, timestamps, observed, dict(meta, channels=channels,
        missing_policy="Per-channel forward fill; leading missing=0; exclude incomplete target rows",
        timestamp_policy="Source order; unique regular hourly/monthly grid; insert missing grid rows",
        missing_cells=int((~observed).sum()), rows=len(values),
        upstream_interpolation_policy="SILSO February 1824 marked missing; forward-filled from January" if name == "sunspots" else None,
        first_timestamp=str(timestamps[0]), last_timestamp=str(timestamps[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for name in SOURCES:
        values, _, _, metadata = load_real(name, download=args.download)
        print(name, values.shape, metadata["sha256"], flush=True)


if __name__ == "__main__":
    main()
