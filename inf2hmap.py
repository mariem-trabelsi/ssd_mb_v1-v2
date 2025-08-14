import cv2
import torch
import argparse
import logging
import numpy as np
from collections import deque
import urllib.request
import tempfile
import os
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor
)

def load_model(model_path, num_classes, device):
    net = create_mobilenetv2_ssd_lite(num_classes=num_classes, is_test=True)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.to(device).eval()
    return net

def main():
    parser = argparse.ArgumentParser(description="SSD MobileNetV2 Lite + Heatmap + URL support")
    parser.add_argument('--video', required=True, help='Chemin vidéo, index caméra ou URL HTTP/RTSP')
    parser.add_argument('--model', required=True, help='Chemin du modèle .pth')
    parser.add_argument('--labels', required=True, help='Chemin de labels.txt')
    parser.add_argument('--threshold', type=float, default=0.4, help='Seuil de probabilité')
    parser.add_argument('--use-cuda', type=lambda x: x.lower() in ['1','true','yes'], default='true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    class_names = [l.strip() for l in open(args.labels, 'r').readlines()]
    num_classes = len(class_names)

    net = load_model(args.model, num_classes, device)
    predictor = create_mobilenetv2_ssd_lite_predictor(net, device=device)

    # Gestion URL / local / caméra
    tmp_file = None
    if args.video.startswith("http://") or args.video.startswith("https://"):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        urllib.request.urlretrieve(args.video, tmp_file.name)
        src = tmp_file.name
    else:
        src = 0 if args.video == "0" else args.video

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error(f"Impossible d’ouvrir la source vidéo: {args.video}")
        return

    displayed_count = 0
    smoothing = 0.2
    hmap_decay = 0.95  # persistance de la heatmap

    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, probs = predictor.predict(frame, top_k=100, prob_threshold=args.threshold)
        person_boxes = []
        for i in range(boxes.size(0)):
            cls_name = class_names[int(labels[i])]
            if cls_name.lower() == "person":
                person_boxes.append(boxes[i].tolist())

        # Mise à jour heatmap
        heatmap *= hmap_decay
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            heatmap[cy, cx] += 1.0  # incrémentation pour densité

        # Normalisation pour affichage
        heatmap_display = np.uint8(np.clip(heatmap*10, 0, 255))
        heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # Compteur animé
        current_count = len(person_boxes)
        displayed_count += (current_count - displayed_count) * smoothing
        cv2.putText(overlay, f"Persons: {int(displayed_count)}", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 3, lineType=cv2.LINE_AA)

        # Dessin des boxes vertes
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("SSD-MBv2 Lite + Heatmap", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if tmp_file:
        os.remove(tmp_file.name)

if __name__ == "__main__":
    main()

