import json
import os

DATASET_DIR = "../dataset"


high = 0
medium = 0
low = 0

for filename in os.listdir(DATASET_DIR):
    if not filename.endswith(".json"):
        continue

    if filename.startswith("stats") or filename.startswith("dataset"):
        continue

    file_path = os.path.join(DATASET_DIR, filename)

    print(f"Processing {filename}...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])

    for qa in qa_pairs:
        if qa["confidence"] == "high":
            high += 1
        elif qa["confidence"] == "medium":
            medium += 1
        elif qa["confidence"] == "low":
            low += 1

print(f"High: {high}")
print(f"Medium: {medium}")
print(f"Low: {low}")

total = high + medium + low
print(f"Total: {total}")

print(f"High: {high / total * 100}%")
print(f"Medium: {medium / total * 100}%")
print(f"Low: {low / total * 100}%")


print("Done.")


#Currently at 11:07 AM 16/04/2026
# High: 3218
# Medium: 714
# Low: 120
# Total: 4052
# High: 79.41757156959525%
# Medium: 17.620927936821325%
# Low: 2.9615004935834155%