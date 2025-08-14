import cv2
import torch
import argparse
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer

def resize_with_padding(image, target_size=(300, 300), color=(114, 114, 114)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2
    padded = cv2.copyMakeBorder(resized, dh, target_h - new_h - dh, dw, target_w - new_w - dw,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, scale, dw, dh

# ---- Arguments ----
parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Video path or camera index')
parser.add_argument('--model', required=True, help='Trained .pth model')
parser.add_argument('--label', required=True, help='Label file')
parser.add_argument('--threshold', type=float, default=0.2, help='Detection threshold')
args = parser.parse_args()

# ---- Load model & labels ----
class_names = [line.strip() for line in open(args.label)]
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(args.model)
predictor = create_mobilenetv1_ssd_predictor(net)

# ---- Video setup ----
cap = cv2.VideoCapture(0 if args.video == "0" else args.video)
timer = Timer()

while True:
    ret, orig_frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = orig_frame.shape[:2]
    input_image, scale, dw, dh = resize_with_padding(orig_frame, (300, 300))

    # ---- Detection ----
    timer.start()
    boxes, labels, probs = predictor.predict(input_image, 10, args.threshold)
    timer.end()

    # ---- Count persons ----
    count_person = sum(1 for l in labels if class_names[l] == "person")

    # ---- Draw bounding boxes on original frame ----
    for i in range(boxes.size(0)):
        box = boxes[i]
        x1 = int((box[0] - dw) / scale)
        y1 = int((box[1] - dh) / scale)
        x2 = int((box[2] - dw) / scale)
        y2 = int((box[3] - dh) / scale)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert
        cv2.putText(orig_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- Show person count ----
    cv2.putText(orig_frame, f"Person count: {count_person}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---- Display ----
    cv2.imshow("Detections", orig_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

