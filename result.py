import os
import matplotlib.pyplot as plt

# Saisie du nom du modèle
res = input("\r\nPlease enter model name: ")

# Création du dossier img/ s'il n'existe pas
os.makedirs("img", exist_ok=True)

# Récupération des fichiers du dossier modèle
files = os.listdir(f"models/{res}")
fileLst = [file for file in files if "mb" in file]

# Tri par numéro d'époque (extrait depuis le nom de fichier)
fileLst.sort(key=lambda x: int(x.split("-")[3]))

x = []
y = []

# Extraction des valeurs de perte (loss) et époques
for file in fileLst:
    parts = file.split("-")
    epoch = parts[3]
    loss_str = parts[5].split(".pth")[0]
    y.append(round(float(loss_str), 3))
    x.append(epoch)

# Détermination du meilleur checkpoint (plus petite perte)
bestChkpt = ""
bestLoss = float("inf")

for file in fileLst:
    loss_str = file.split("-")[5].split(".pth")[0]
    loss_val = float(loss_str)
    if loss_val < bestLoss:
        bestLoss = loss_val
        bestChkpt = file

print("Best Checkpoint: {}".format(bestChkpt))

# Tracer et sauvegarder l'image
plt.plot(x, y, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training result")
plt.grid(True)
plt.tight_layout()

# Enregistrement du graphique dans le dossier img/
save_path = os.path.join("img", f"{res}_training_loss.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")

