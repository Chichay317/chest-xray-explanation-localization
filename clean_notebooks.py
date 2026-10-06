import glob
import json

MARKER = "This Python 3 environment comes with many helpful analytics libraries"

files = sorted(glob.glob("notebooks/*.ipynb"))
if not files:
    print("No notebooks found. Run this from the project folder, with the notebooks inside 'notebooks'.")

for path in files:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    before = len(nb["cells"])
    nb["cells"] = [c for c in nb["cells"] if MARKER not in "".join(c.get("source", []))]
    removed = before - len(nb["cells"])
    if removed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    print(f"{path}: {'removed the starter cell' if removed else 'nothing to remove'} ({len(nb['cells'])} cells)")
