# SSD MobileNet Fine-Tuning for "Person" Class

This project demonstrates the workflow to fine-tune **SSD MobileNet** for the **"person"** class.  
It removes the two pre-trained SSD models (MobileNet V1 and V2) and trains a custom model on your dataset.

---

## Python Dependencies

The project requires the following Python packages:

absl-py 2.3.0
boto3 1.38.46
botocore 1.38.46
contourpy 1.3.0
cycler 0.12.1
deep-sort-realtime 1.3.2
filelock 3.18.0
fonttools 4.58.4
fsspec 2025.5.1
grpcio 1.73.1
importlib_metadata 8.7.0
importlib_resources 6.5.2
imutils 0.5.4
Jinja2 3.1.6
jmespath 1.0.1
joblib 1.5.1
kiwisolver 1.4.7
Markdown 3.8.2
MarkupSafe 3.0.2
matplotlib 3.9.4
mpmath 1.3.0
networkx 3.2.1
numpy 2.0.2
nvidia-cublas-cu12 12.6.4.1
nvidia-cuda-cupti-cu12 12.6.80
nvidia-cuda-nvrtc-cu12 12.6.77
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12 9.5.1.17
nvidia-cufft-cu12 11.3.0.4
nvidia-cufile-cu12 1.11.1.6
nvidia-curand-cu12 10.3.7.77
nvidia-cusolver-cu12 11.7.1.2
nvidia-cusparse-cu12 12.5.4.2
nvidia-cusparselt-cu12 0.6.3
nvidia-nccl-cu12 2.26.2
nvidia-nvjitlink-cu12 12.6.85
nvidia-nvtx-cu12 12.6.77
opencv-python 4.11.0.86
packaging 25.0
pandas 2.3.0
pillow 11.2.1
pip 25.1.1
protobuf 6.31.1
pyparsing 3.2.3
python-dateutil 2.9.0.post0
pytz 2025.2
s3transfer 0.13.0
scikit-learn 1.6.1
scipy 1.13.1
setuptools 58.1.0
six 1.17.0
sympy 1.14.0
tensorboard 2.19.0
tensorboard-data-server 0.7.2
threadpoolctl 3.6.0
torch 2.7.1
torchvision 0.22.1
tqdm 4.67.1
triton 3.3.1
typing_extensions 4.14.0
tzdata 2025.2
urllib3 1.26.20
Werkzeug 3.1.3
yt-dlp 2025.8.11
zipp 3.23.0


## Workflow

Follow these steps to train and run the model:

1. **Create a Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows```

2. **Annotate your data**
Use Roboflow to label your images in VOC XML format.

3. **Prepare your dataset**

Unzip your dataset.
Split it using the command:
```bash

```

4. **Start training**

5. evaluation

You can also optionally include an **image preview** of the graphs in your README if you save them as `.png` in the `results/` folder:

```markdown
![Training Loss Graph](results/loss_curve.png)
![Validation mAP](results/map_curve.png)



6. **Run inference**
Test your trained model using the inference script:
```bash
**** inference from local video
python3 inference_mb2_video.py \
  --video data/video.mp4 \
  --model models/model_mb2/best_epoch.pth \
  --labels models/model_mb2/labels.txt \
  --threshold 0.4 \
  --use-cuda true \
  --save output_mb2.mp4
 ```
  
**inference via webcam**
```bash
  python3 inference_mb2_video.py \
  --video 0 \
  --model models/model_mb2/mb2-ssd-lite-Epoch-225-Loss-2.6423.pth \
  --labels models/model_mb2/labels.txt \
  --threshold 0.4
```

**inference via cam**
```bash
python3 inference2.py   --video "rtsp://admin:L2E78815@10.0.22.1:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"   --model models/model_mb2/mb2-ssd-lite-Epoch-530-Loss-2.0126.pth   --labels models/model_mb2/labels.txt   --threshold 0.5
```
