import json

import matplotlib.pyplot as plt

results = []

with open("/mnt/workspace/ty/xray/runs/train/compact-multiclass-attempt-1/results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        results.append(json.loads(line))

# Sort theo epoch
results = sorted(results, key=lambda x: x["epoch"])

# =========================================================
# BEST EPOCH THEO mAP50
# =========================================================
best = max(results, key=lambda x: x["map50"])

print(f"Best epoch : {best['epoch']}")
print(f"mAP50      : {best['map50']:.6f}")
print(f"mAP50-95   : {best['map50_95']:.6f}")
print(f"Precision  : {best['precision']:.6f}")
print(f"Recall     : {best['recall']:.6f}")
print(f"Train loss : {best['train_loss']:.6f}")

if "val_loss" in best:
    print(f"Val loss   : {best['val_loss']:.6f}")

epochs = [x["epoch"] for x in results]


# =========================================================
# 1. GLOBAL METRICS
# =========================================================
map50 = [x["map50"] for x in results]
map50_95 = [x["map50_95"] for x in results]
precision = [x["precision"] for x in results]
recall = [x["recall"] for x in results]

plt.figure(figsize=(12, 7))

plt.plot(epochs, map50, marker="o", label="mAP50")
plt.plot(epochs, map50_95, marker="o", label="mAP50-95")
plt.plot(epochs, precision, marker="o", label="Precision")
plt.plot(epochs, recall, marker="o", label="Recall")

plt.axvline(
    x=best["epoch"],
    linestyle="--",
    label=f"Best epoch = {best['epoch']}"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("metrics.png", dpi=200)



# =========================================================
# 2. mAP50 TỪNG CLASS
# =========================================================

# lấy danh sách class từ epoch đầu tiên
class_names = list(results[0]["per_class"].keys())

plt.figure(figsize=(14, 9))

for class_name in class_names:
    class_map50 = []

    for x in results:
        value = x["per_class"].get(class_name, {}).get("map50", None)
        class_map50.append(value)

    plt.plot(
        epochs,
        class_map50,
        marker="o",
        markersize=3,
        label=class_name
    )

plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("mAP50 per Class")
plt.grid(True)
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()

plt.savefig(
    "map50_per_class.png",
    dpi=200,
    bbox_inches="tight"
)



# =========================================================
# 3. mAP50-95 TỪNG CLASS
# =========================================================

plt.figure(figsize=(14, 9))

for class_name in class_names:
    class_map = []

    for x in results:
        value = x["per_class"].get(
            class_name, {}
        ).get("map50_95", None)

        class_map.append(value)

    plt.plot(
        epochs,
        class_map,
        marker="o",
        markersize=3,
        label=class_name
    )

plt.xlabel("Epoch")
plt.ylabel("mAP50-95")
plt.title("mAP50-95 per Class")
plt.grid(True)
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()

plt.savefig(
    "map50_95_per_class.png",
    dpi=200,
    bbox_inches="tight"
)



# =========================================================
# 4. TRAIN LOSS + VAL LOSS CHUNG 1 CHART
# =========================================================

train_loss = [x["train_loss"] for x in results]

has_val_loss = all("val_loss" in x for x in results)

plt.figure(figsize=(12, 7))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train Loss"
)

if has_val_loss:
    val_loss = [x["val_loss"] for x in results]

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )
else:
    print(
        "\nWarning: Không tìm thấy key 'val_loss' "
        "trong JSONL."
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("train_val_loss.png", dpi=200)
