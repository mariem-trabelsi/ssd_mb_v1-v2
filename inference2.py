import cv2
import torch
import argparse
import logging
import numpy as np
from collections import deque
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
    parser = argparse.ArgumentParser(description="Inference vidéo SSD MobileNetV2 Lite avec graphique")
    parser.add_argument('--video', required=True, help='Chemin vidéo, index caméra (ex: 0) ou URL RTSP')
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

    src = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error(f"Impossible d’ouvrir la source vidéo: {args.video}")
        return

    displayed_count = 0
    smoothing = 0.2
    history = deque(maxlen=100)  # Historique des N dernières frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, probs = predictor.predict(frame, top_k=100, prob_threshold=args.threshold)
        person_count = sum(1 for i in range(boxes.size(0)) if class_names[int(labels[i])].lower() == "person")
        history.append(person_count)

        # Animation fluide
        displayed_count = displayed_count + (person_count - displayed_count) * smoothing

        # Dessin des boxes vertes
        for i in range(boxes.size(0)):
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            cls_id = int(labels[i])
            score = float(probs[i])
            cls_name = class_names[cls_id]
            if cls_name.lower() == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls_name}: {score:.2f}", (x1, max(0, y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        h, w = frame.shape[:2]

        # Overlay compteur animé stylé (noir, grand)
        cv2.putText(frame, f"Persons: {int(displayed_count)}", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 3, lineType=cv2.LINE_AA)

        # Dessiner l'historique (mini graphique)
        hist_h = 60
        for idx, val in enumerate(history):
            bar_height = int((val / max(history)) * hist_h) if max(history) > 0 else 0
            cv2.line(frame,
                     (w - len(history) + idx, h - 5),
                     (w - len(history) + idx, h - 5 - bar_height),
                     (50, 50, 255), 2)

        cv2.imshow("SSD-MBv2 Lite - Person Count", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

