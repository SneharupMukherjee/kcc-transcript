#!/usr/bin/env python3
import csv
import sqlite3
import tempfile
from pathlib import Path

CSV_PATH = Path('/home/sneharup/KCC/apt/data/kcc_2025.csv')


def dedup_csv(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print('No data file found.')
        return
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print('Empty CSV.')
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'dedup.sqlite'
            conn = sqlite3.connect(db_path.as_posix())
            cur = conn.cursor()
            cols = [c.replace(' ', '_') for c in header]
            col_defs = ', '.join([f'"{c}" TEXT' for c in cols])
            unique_cols = ', '.join([f'"{c}"' for c in cols])
            cur.execute(f'CREATE TABLE data ({col_defs}, UNIQUE({unique_cols}))')
            conn.commit()

            col_names = ', '.join([f'"{c}"' for c in cols])
            placeholders = ', '.join(['?'] * len(cols))
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

            out_path = csv_path.with_suffix('.dedup.csv')
            with out_path.open('w', encoding='utf-8', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(header)
                select_cols = ', '.join([f'"{c}"' for c in cols])
                for row in cur.execute(f'SELECT {select_cols} FROM data'):
                    writer.writerow(row)
            conn.close()
            out_path.replace(csv_path)
    print('De-dup complete.')


if __name__ == '__main__':
    dedup_csv(CSV_PATH)
