from pathlib import Path
import shutil

results = {}
for file in Path("dataset/labels/train/").rglob("*.txt"):
    with open(file, 'r') as file:
        data = file.read()
        sym = data[0]
        if sym in results:
            results[sym] += 1
        else:
            results[sym] = 1
print(results)