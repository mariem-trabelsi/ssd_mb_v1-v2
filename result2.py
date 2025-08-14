import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor
)

# ----------------- CONFIG -----------------
CHECKPOINT_DIR = "models/model_mb2/"
IMAGES_DIR = "data/last_model/JPEGImages/"
VAL_TXT = "data/last_model/ImageSets/Main/val.txt"  # fichier contenant noms sans extensions
IMAGE_EXTENSION = ".jpg"  # extension des images
CONF_THRESHOLD = 0.4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- CHARGEMENT DU MODELE -----------------
def load_model(model_path, num_classes, device):
    net = create_mobilenetv2_ssd_lite(num_classes=num_classes, is_test=True)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.to(device).eval()
    return net

# ----------------- MEILLEUR CHECKPOINT -----------------
pattern = re.compile(r"mb2-ssd-lite-Epoch-(\d+)-Loss-([0-9]+\.[0-9]+)\.pth")
checkpoints = []

for file in os.listdir(CHECKPOINT_DIR):
    match = pattern.match(file)
    if match:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        checkpoints.append((epoch, loss, file))

if not checkpoints:
    print("Aucun checkpoint trouvé.")
    exit()

checkpoints.sort(key=lambda x: x[1])
best_epoch, best_loss, best_file = checkpoints[0]
model_path = os.path.join(CHECKPOINT_DIR, best_file)

print(f"✅ Meilleur checkpoint : {best_file} (Epoch {best_epoch}, Loss {best_loss:.4f})")
print(f"📂 Chemin complet : {model_path}")

# ----------------- CHARGEMENT DES LABELS -----------------
labels_path = os.path.join(CHECKPOINT_DIR, "labels.txt")
class_names = [l.strip() for l in open(labels_path, "r").readlines()]
num_classes = len(class_names)

# ----------------- INIT PREDICTOR -----------------
net = load_model(model_path, num_classes, DEVICE)
predictor = create_mobilenetv2_ssd_lite_predictor(net, device=DEVICE)

# ----------------- LIRE FICHIER VAL.TXT -----------------
with open(VAL_TXT, "r") as f:
    val_files = [line.strip() for line in f.readlines()]

# ----------------- EVALUATION -----------------
all_scores = []
all_labels = []
person_counts = []

for file_name in tqdm(val_files, desc="Evaluation images"):
    img_path = os.path.join(IMAGES_DIR, file_name + IMAGE_EXTENSION)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Impossible de lire {img_path}, fichier manquant ou corrompu")
        continue

    boxes, labels, probs = predictor.predict(img, top_k=100, prob_threshold=CONF_THRESHOLD)
    person_boxes = []
    for i in range(boxes.size(0)):
        cls_id = int(labels[i])
        cls_name = class_names[cls_id]
        score = float(probs[i])
        all_scores.append(score)
        all_labels.append(cls_name.lower() == "person")
        if cls_name.lower() == "person":
            person_boxes.append(boxes[i].tolist())

    person_counts.append(len(person_boxes))

# ----------------- CALCUL DES METRICS -----------------
# Précision, rappel, mAP simplifiés
from sklearn.metrics import precision_recall_curve, average_precision_score

y_true = np.array(all_labels)
y_scores = np.array(all_scores)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

print(f"\n📊 Evaluation Metrics :")
print(f" - mAP (AP for person): {ap:.4f}")
print(f" - Total images evaluated: {len(person_counts)}")
print(f" - Average persons per image: {np.mean(person_counts):.2f}")

# ----------------- GRAPHIQUES -----------------
os.makedirs("results", exist_ok=True)

# 1. Histogramme des scores
plt.figure(figsize=(8,5))
plt.hist(y_scores, bins=20, color="skyblue", edgecolor="black")
plt.title("Histogramme des scores")
plt.xlabel("Confidence Score")
plt.ylabel("Nombre de détections")
plt.grid(True)
plt.savefig("results/histogram_scores.png")
plt.close()

# 2. Courbe Précision-Rappel
plt.figure(figsize=(8,5))
plt.plot(recall, precision, color="blue", lw=2, label=f"AP={ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig("results/precision_recall_curve.png")
plt.close()

# 3. Compte des personnes par image
plt.figure(figsize=(8,5))
plt.plot(range(len(person_counts)), person_counts, marker="o", color="green")
plt.xlabel("Image index")
plt.ylabel("Person count")
plt.title("Nombre de personnes par image")
plt.grid(True)
plt.savefig("results/person_count.png")
plt.close()

print("\n✅ Graphiques sauvegardés dans le dossier 'results/'")

