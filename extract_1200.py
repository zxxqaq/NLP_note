import json


with open("data/dev.json", "r", encoding="utf8") as f:
    data = json.load(f)

subset = data[:1200]


with open("dev_1200.json", "w", encoding="utf8") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print("success dev_1200.json")