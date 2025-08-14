# vision/ssd/config/mobilenetv2_ssd_config.py
import numpy as np
from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
import torch
import math

# Configuration par défaut pour SSD MobileNet V2 Lite
image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# Spécifications des feature maps pour 300x300
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

# Générer les priors pour la résolution par défaut (300x300)
priors = generate_ssd_priors(specs, image_size)

def set_image_size(size=300, min_ratio=20, max_ratio=90):
    """
    Ajuste la taille de l'image et recalcule les priors pour MobileNet V2 SSD Lite.
    
    Args:
        size (int): Nouvelle taille d'image (size x size).
        min_ratio (int): Ratio minimum pour les tailles des boîtes (% de image_size).
        max_ratio (int): Ratio maximum pour les tailles des boîtes (% de image_size).
    """
    global image_size, specs, priors
    
    image_size = size
    
    # Créer un modèle temporaire pour obtenir les tailles des feature maps
    ssd = create_mobilenetv2_ssd_lite(num_classes=2, width_mult=1.0)  # 2 classes : person + background
    ssd.eval()  # Set model to evaluation mode to avoid BatchNorm issues
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():  # Disable gradient computation for efficiency
        feature_maps = ssd(x, get_feature_map_size=True)
    
    # Calculer les strides pour chaque feature map
    steps = [math.ceil(image_size * 1.0 / feature_map) for feature_map in feature_maps]
    
    # Calculer les tailles des boîtes (min_sizes et max_sizes)
    step = int(math.floor((max_ratio - min_ratio) / (len(feature_maps) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(image_size * ratio / 100.0)
        max_sizes.append(image_size * (ratio + step) / 100.0)
    min_sizes = [image_size * (min_ratio / 2) / 100.0] + min_sizes
    max_sizes = [image_size * min_ratio / 100.0] + max_sizes
    
    # Conserver la configuration par défaut pour 300x300 pour compatibilité
    if image_size != 300:
        specs = []
        for i in range(len(feature_maps)):
            specs.append(SSDSpec(feature_maps[i], steps[i], SSDBoxSizes(min_sizes[i], max_sizes[i]), [2, 3]))
    
    # Générer les nouveaux priors
    priors = generate_ssd_priors(specs, image_size)
    
    return priors
