#!/usr/bin/env python3
import csv
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "kcc.env"
DATA_PATH = BASE_DIR / "data" / "kcc_2025.csv"
PAGES_DIR = BASE_DIR / "data" / "pages"
OFFSETS_PATH = BASE_DIR / "data" / "downloaded_offsets.txt"
MERGED_PATH = BASE_DIR / "data" / "kcc_2025_pages_merged.csv"


def load_env(path: Path):
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def build_url(base_url, params):
    return base_url + "?" + urllib.parse.urlencode(params, doseq=True)


def fetch(url, retries=8, backoff=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kcc-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.getcode()
                body = resp.read()
                if status == 200:
                    return body
                last_err = RuntimeError(f"HTTP {status}")
        except Exception as e:
            last_err = e
        time.sleep(backoff * attempt)
    raise last_err


def get_total(base_url, api_key, year):
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[year]": year,
        "limit": 1,
        "offset": 0,
    }
    url = build_url(base_url, params)
    body = fetch(url)
    import json

    j = json.loads(body.decode("utf-8"))
    return int(j.get("total", 0))


def read_existing_offset(csv_path: Path):
    if not csv_path.exists():
        return 0
    # Drop a trailing partial line if present, then count rows to resume.
    if not trim_partial_last_line(csv_path):
        # If trim fails for any reason, fall back to simple counting.
        pass
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = sum(1 for _ in reader)
    if rows <= 1:
        return 0
    return rows - 1


def trim_partial_last_line(csv_path: Path) -> bool:
    """
    Ensure the file ends with a newline. If not, drop the last line
    (likely partial due to interrupted write).
    Returns True if a trim occurred, False otherwise.
    """
    try:
        size = csv_path.stat().st_size
        if size == 0:
            return False
        with csv_path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
            if last == b"\n":
                return False
        # No trailing newline: drop the last line
        with csv_path.open("rb") as f:
            data = f.read()
        # Find last newline and truncate after it
        idx = data.rfind(b"\n")
        if idx == -1:
            # Single partial line; remove entire file
            csv_path.write_bytes(b"")
        else:
            csv_path.write_bytes(data[: idx + 1])
        return True
    except Exception:
        return False


def dedup_csv(csv_path: Path):
    # Deduplicate using sqlite with a UNIQUE constraint across all columns.
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dedup.sqlite"
            conn = sqlite3.connect(db_path.as_posix())
            cur = conn.cursor()
            cols = [c.replace(" ", "_") for c in header]
            col_defs = ", ".join([f'"{c}" TEXT' for c in cols])
            unique_cols = ", ".join([f'"{c}"' for c in cols])
            cur.execute(f'CREATE TABLE data ({col_defs}, UNIQUE({unique_cols}))')
            conn.commit()

            col_names = ", ".join([f'"{c}"' for c in cols])
            placeholders = ", ".join(["?"] * len(cols))
            insert_sql = f"INSERT OR IGNORE INTO data ({col_names}) VALUES ({placeholders})"
            batch = []
            for row in reader:
                if len(row) != len(cols):
                    continue
                batch.append(row)
                if len(batch) >= 5000:
                    cur.executemany(insert_sql, batch)
                    conn.commit()
                    batch = []
            if batch:
                cur.executemany(insert_sql, batch)
                conn.commit()

            # Export back to CSV
            out_path = csv_path.with_suffix(".dedup.csv")
            with out_path.open("w", encoding="utf-8", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(header)
                select_cols = ", ".join([f'"{c}"' for c in cols])
                for row in cur.execute(f"SELECT {select_cols} FROM data"):
                    writer.writerow(row)
            conn.close()

            # Replace original
            out_path.replace(csv_path)


def format_rate(rows_delta, seconds):
    if seconds <= 0:
        return "0 r/s"
    rps = rows_delta / seconds
    return f"{rps:,.1f} r/s"


def format_eta(remaining_rows, rps):
    if rps <= 0:
        return "ETA: --"
    seconds = remaining_rows / rps
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"ETA: {hrs}h {mins}m"
    if mins:
        return f"ETA: {mins}m {sec}s"
    return f"ETA: {sec}s"


def main():
    env = load_env(CONFIG_PATH)
    base_url = env.get("KCC_API_BASE_URL")
    api_key = env.get("KCC_API_KEY")
    year = env.get("KCC_DEFAULT_YEAR", "2024")
    limit = int(env.get("KCC_DEFAULT_LIMIT", "5000"))
    dedup_enabled = env.get("KCC_DEDUP", "0") == "1"
    page_mode = env.get("KCC_PAGE_MODE", "0") == "1"

    if not base_url or not api_key:
        print("Missing KCC_API_BASE_URL or KCC_API_KEY in config/kcc.env", file=sys.stderr)
        sys.exit(1)

    total = get_total(base_url, api_key, year)
    if total <= 0:
        print("No records reported for the requested year.")
        sys.exit(0)

    data_path = BASE_DIR / "data" / f"kcc_{year}.csv"
    pages_dir = BASE_DIR / "data" / f"pages_{year}"
    offsets_path = BASE_DIR / "data" / f"downloaded_offsets_{year}.txt"
    merged_path = BASE_DIR / "data" / f"kcc_{year}_pages_merged.csv"

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    if page_mode:
        run_page_mode(base_url, api_key, year, total, limit, pages_dir, offsets_path, merged_path)
        return

    offset = read_existing_offset(data_path)
    if offset >= total:
        print(f"Already complete: {offset}/{total}")
        return

    # Download in pages and append
    header_written = data_path.exists() and data_path.stat().st_size > 0

    start_time = time.time()
    last_time = start_time
    last_offset = offset

    while offset < total:
        params = {
            "api-key": api_key,
            "format": "csv",
            "filters[year]": year,
            "limit": limit,
            "offset": offset,
        }
        url = build_url(base_url, params)
        body = fetch(url)
        text = body.decode("utf-8")
        if text.lstrip().startswith("{"):
            # Likely an error payload; back off and retry next loop
            print(f"Server returned JSON error at offset {offset}, backing off...", flush=True)
            time.sleep(10)
            continue
        lines = text.splitlines()
        if not lines:
            break

        # First line is header; skip if already written
        if header_written:
            data_lines = lines[1:]
        else:
            data_lines = lines
            header_written = True

        with data_path.open("a", encoding="utf-8", newline="") as f:
            f.write("\n".join(data_lines))
            if data_lines:
                f.write("\n")

        # Update offset by number of data lines written
        offset += max(0, len(data_lines))
        now = time.time()
        delta_rows = offset - last_offset
        delta_time = now - last_time
        total_time = now - start_time
        rps = delta_rows / delta_time if delta_time > 0 else 0
        overall_rps = (offset / total_time) if total_time > 0 else 0
        remaining = max(0, total - offset)
        eta = format_eta(remaining, overall_rps if overall_rps > 0 else rps)

        print(
            f"{datetime.now().strftime('%H:%M:%S')}  "
            f"{offset:,}/{total:,}  "
            f"+{delta_rows:,}  "
            f"{format_rate(delta_rows, delta_time)}  "
            f"avg {overall_rps:,.1f} r/s  "
            f"{eta}",
            flush=True,
        )

        last_time = now
        last_offset = offset
        time.sleep(0.3)

    if dedup_enabled:
        if offset >= total:
            print("Starting de-duplication (KCC_DEDUP=1)...", flush=True)
            dedup_csv(data_path)
            print("De-duplication complete.", flush=True)
        else:
            print(
                "Skipping de-duplication because download is incomplete. "
                "Resume until fully downloaded, then run dedup.",
                flush=True,
            )


def load_downloaded_offsets(offsets_path: Path):
    if not offsets_path.exists():
        return set()
    return set(
        int(x.strip())
        for x in offsets_path.read_text().splitlines()
        if x.strip().isdigit()
    )


def save_downloaded_offset(offsets_path: Path, offset: int):
    with offsets_path.open("a", encoding="utf-8") as f:
        f.write(f"{offset}\n")


def run_page_mode(base_url, api_key, year, total, limit, pages_dir: Path, offsets_path: Path, merged_path: Path):
    print("Page mode enabled. Downloading offset pages safely.", flush=True)
    downloaded = load_downloaded_offsets(offsets_path)

    offsets = list(range(0, total, limit))
    start_time = time.time()
    done = 0

    for offset in offsets:
        if offset in downloaded:
            done += 1
            continue

        params = {
            "api-key": api_key,
            "format": "csv",
            "filters[year]": year,
            "limit": limit,
            "offset": offset,
        }
        url = build_url(base_url, params)
        body = fetch(url)
        text = body.decode("utf-8")
        if text.lstrip().startswith("{"):
            print(f"Server returned JSON error at offset {offset}, retry later.", flush=True)
            time.sleep(5)
            continue

        page_path = pages_dir / f"offset_{offset:07d}.csv"
        page_path.write_text(text, encoding="utf-8")
        save_downloaded_offset(offsets_path, offset)
        downloaded.add(offset)
        done += 1

        elapsed = time.time() - start_time
        rps = done / elapsed if elapsed > 0 else 0
        remaining = len(offsets) - done
        eta = format_eta(remaining, rps) if rps > 0 else "ETA: --"
        print(
            f"{datetime.now().strftime('%H:%M:%S')}  pages {done}/{len(offsets)}  "
            f"{rps:,.2f} pages/s  {eta}",
            flush=True,
        )
        time.sleep(0.1)

    if len(downloaded) == len(offsets):
        print("All pages downloaded. Merging...", flush=True)
        merge_pages(pages_dir, merged_path)
        print(f"Merged CSV written to {merged_path}", flush=True)
    else:
        missing = len(offsets) - len(downloaded)
        print(f"Page download incomplete. Missing {missing} pages.", flush=True)


def merge_pages(pages_dir: Path, merged_path: Path):
    pages = sorted(pages_dir.glob("offset_*.csv"))
    if not pages:
        print("No pages found to merge.")
        return
    header_written = False
    with merged_path.open("w", encoding="utf-8", newline="") as out:
        for page in pages:
            text = page.read_text(encoding="utf-8")
            lines = text.splitlines()
            if not lines:
                continue
            if header_written:
                lines = lines[1:]
            else:
                header_written = True
            if lines:
                out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
