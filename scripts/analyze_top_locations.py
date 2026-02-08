import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

target_states = ["UTTAR PRADESH", "RAJASTHAN", "MADHYA PRADESH", "HARYANA", "MAHARASHTRA"]
target_districts = [
    ("KURNOOL", "ANDHRA PRADESH"),
    ("NAGAUR", "RAJASTHAN"),
    ("BARMER", "RAJASTHAN"),
    ("CHURU", "RAJASTHAN"),
    ("HISSAR", "HARYANA"),
    ("BARDDHAMAN", "WEST BENGAL"),
    ("JODHPUR", "RAJASTHAN"),
    ("SAGAR", "MADHYA PRADESH"),
    ("BHIWANI", "HARYANA"),
    ("BIKANER", "RAJASTHAN")
]

state_crops = {s: Counter() for s in target_states}
state_qtypes = {s: Counter() for s in target_states}

dist_crops = {f"{dist}, {state}": Counter() for dist, state in target_districts}
dist_qtypes = {f"{dist}, {state}": Counter() for dist, state in target_districts}

csv_path = "data/kcc_merged_2024_2025.csv"

def get_crop_with_fallback(counter):
    items = counter.most_common(5)
    if not items: return "N/A"
    # If top is Others/Other, take the second
    if items[0][0].upper() in ["OTHERS", "OTHER"] and len(items) > 1:
        return items[1][0]
    return items[0][0]

def get_as_is_qtype(counter):
    items = counter.most_common(1)
    return items[0][0] if items else "N/A"

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row: continue
        state = row[0].strip().upper()
        dist = row[1].strip().upper()
        crop = row[6].strip()
        qtype = row[7].strip()

        if state in state_crops:
            if crop: state_crops[state][crop] += 1
            if qtype: state_qtypes[state][qtype] += 1

        d_key = f"{dist}, {state}"
        if d_key in dist_crops:
            if crop: dist_crops[d_key][crop] += 1
            if qtype: dist_qtypes[d_key][qtype] += 1

print("--- UNIFIED STATES ENRICHMENT ---")
for s in target_states:
    top_crop = get_crop_with_fallback(state_crops[s])
    top_qtype = get_as_is_qtype(state_qtypes[s])
    print(f"STATE: {s} | CROP: {top_crop} | QTYPE: {top_qtype}")

print("\n--- UNIFIED DISTRICTS ENRICHMENT ---")
for d_key in dist_crops:
    top_crop = get_crop_with_fallback(dist_crops[d_key])
    top_qtype = get_as_is_qtype(dist_qtypes[d_key])
    print(f"DIST: {d_key} | CROP: {top_crop} | QTYPE: {top_qtype}")
