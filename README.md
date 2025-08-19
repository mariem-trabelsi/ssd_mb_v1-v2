# SSD MobileNet Fine-Tuning for "Person" Class

This project demonstrates the workflow to fine-tune **SSD MobileNet** for the **"person"** class.  
You can adjust labels.txt file under models folder to fine tune other objects included in **COCO dataset**
It removes the two pre-trained SSD models (MobileNet V1 and V2) and trains a custom model on your dataset.

---

## Python Dependencies
```bash
pip install \
absl-py==2.3.0 \
boto3==1.38.46 \
botocore==1.38.46 \
contourpy==1.3.0 \
cycler==0.12.1 \
deep-sort-realtime==1.3.2 \
filelock==3.18.0 \
fonttools==4.58.4 \
fsspec==2025.5.1 \
grpcio==1.73.1 \
importlib_metadata==8.7.0 \
importlib_resources==6.5.2 \
imutils==0.5.4 \
Jinja2==3.1.6 \
jmespath==1.0.1 \
joblib==1.5.1 \
kiwisolver==1.4.7 \
Markdown==3.8.2 \
MarkupSafe==3.0.2 \
matplotlib==3.9.4 \
mpmath==1.3.0 \
networkx==3.2.1 \
numpy==2.0.2 \
nvidia-cublas-cu12==12.6.4.1 \
nvidia-cuda-cupti-cu12==12.6.80 \
nvidia-cuda-nvrtc-cu12==12.6.77 \
nvidia-cuda-runtime-cu12==12.6.77 \
nvidia-cudnn-cu12==9.5.1.17 \
nvidia-cufft-cu12==11.3.0.4 \
nvidia-cufile-cu12==1.11.1.6 \
nvidia-curand-cu12==10.3.7.77 \
nvidia-cusolver-cu12==11.7.1.2 \
nvidia-cusparse-cu12==12.5.4.2 \
nvidia-cusparselt-cu12==0.6.3 \
nvidia-nccl-cu12==2.26.2 \
nvidia-nvjitlink-cu12==12.6.85 \
nvidia-nvtx-cu12==12.6.77 \
opencv-python==4.11.0.86 \
packaging==25.0 \
pandas==2.3.0 \
pillow==11.2.1 \
pip==25.1.1 \
protobuf==6.31.1 \
pyparsing==3.2.3 \
python-dateutil==2.9.0.post0 \
pytz==2025.2 \
s3transfer==0.13.0 \
scikit-learn==1.6.1 \
scipy==1.13.1 \
setuptools==58.1.0 \
six==1.17.0 \
sympy==1.14.0 \
tensorboard==2.19.0 \
tensorboard-data-server==0.7.2 \
threadpoolctl==3.6.0 \
torch==2.7.1 \
torchvision==0.22.1 \
tqdm==4.67.1 \
triton==3.3.1 \
typing_extensions==4.14.0 \
tzdata==2025.2 \
urllib3==1.26.20 \
Werkzeug==3.1.3 \
yt-dlp==2025.8.11 \
zipp==3.23.0
```


## 💡Workflow

Follow these steps to train and run the model:

1. **Create a Python virtual environment and install libs**
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

2. **Annotate your data**
Use Roboflow to label your images in VOC XML format.

3. **Prepare your dataset**

Unzip your dataset.
Split it using the command:
```bash
python seperateImages.py
```

4. **Start training**
```bash
python3 train_ssd_mb2.py     --validation-epochs 5     --log-level info     

```
you can resume training by adding this flag with the last epoch
```bash
--resume models/model_mb2/checkpoint-epoch-xx.pth
```
5. **evaluation**
```bash
python result2.py
```
The python file result1.py for ssd mb1 et result2.py for v2.

Output: 

```bash
✅ Meilleur checkpoint : mb2-ssd-lite-Epoch-730-Loss-1.9619.pth (Epoch 730, Loss 1.9619)
📂 Chemin complet : models/model_mb2/mb2-ssd-lite-Epoch-730-Loss-1.9619.pth
Evaluation images: 100%|██████████████████████| 142/142 [00:09<00:00, 15.37it/s]

📊 Evaluation Metrics :
 - mAP (AP for person): 0.8000
 - Total images evaluated: 250
 - Average persons per image: 9.28

✅ Graphiques sauvegardés dans le dossier 'results/'
```

This will generate histogram_scores.png, person_count.png and precision_recall_curve.png in results folder.




6. **Run Inference**
Test your trained model using the inference script:

**Inference from local video**
```bash
python3 inference_mb2_video.py \
  --video data/video.mp4 \
  --model models/model_mb2/best_epoch.pth \
  --labels models/model_mb2/labels.txt \
  --threshold 0.4 \
  --use-cuda true \
  --save output_mb2.mp4
 ```
  
**Inference via webcam**
```bash
  python3 inference_mb2_video.py \
  --video 0 \
  --model models/model_mb2/mb2-ssd-lite-Epoch-xxx-Loss-1.6423.pth \
  --labels models/model_mb2/labels.txt \
  --threshold 0.4
```

**Inference via cam**
```bash
python3 inference2.py   --video "rtsp://admin:xxxxxx@xx.0.xx.1:xxx/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"   --model models/model_mb2/mb2-ssd-lite-Epoch-best-one.pth   --labels models/model_mb2/labels.txt   --threshold 0.5
```
You can update threshold as you want.

## results ( while training)
![result](image1.png)

