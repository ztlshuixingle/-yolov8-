# 深度学习大作业——基于yolov8的施工现场安全帽检测

一个轻量级的**YOLOv8n 模型（采用 MobileNet 主干网络和 CARAFE 上采样机制）**，用于检测施工现场人员是否佩戴安全帽。

<p align="center">
  <img src="docs/train_curves.png" alt="Training curves" width="800">
</p>

---

## 1. Project Structure

```
helmet-detection/
├── datasets/                 # YOLO-format images & labels
│   ├── images/
│   │   ├── train/  (2 607 imgs)
│   │   └── val/    (  682 imgs)
│   └── labels/
├── models/
│   └── yolov8n_mobilenet_carafe.yaml   # custom arch config
├── weights/
│   ├── best.pt
│   └── last.pt
├── train.py                  # Ultralytics CLI wrapper
├── val.py
└── README.md                 # ← you are here
```

---

## 2. Dataset

| Split | Images | Annotation format |
|-------|--------|-------------------|
| Train | 2 607  | YOLO TXT (`class xc yc w h`) |
| Val   |   682  | YOLO TXT |

- **Classes (2)**: `0 = Person-With-Helmet`, `1 = Person-Without-Helmet`
- Image resolution varies; all are resized to **640 × 640** during training.

---

## 3. Model Architecture

| Component   | Details |
|-------------|---------|
| **Base**    | YOLOv8n (Ultralytics) |
| **Backbone**| MobileNet-V3 Small |
| **Neck**    | FPN + CARAFE |
| **Head**    | YOLOv8 decoupled head |
| **Params**  | ≈ 3.2M |
| **Input**   | 640 × 640 RGB |

---

## 4. Environment & Training

```bash
conda create -n yolo-helmet python=3.9
conda activate yolo-helmet
pip install ultralytics==8.1.0 torch>=1.13 torchvision
```

> Tested on **CUDA 11.7 + RTX 3060 (12 GB)**; CPU-only also supported.

###  Training Command

```bash
yolo detect train \
  model=models/yolov8n_mobilenet_carafe.yaml \
  data=datasets/helmet.yaml \
  epochs=250 imgsz=640 batch=16 \
  optimizer=SGD lr0=0.01 weight_decay=5e-4 \
  name=helmet_mnv3_carafe
```

- Early stopping `patience=50`
- Augmentations: Mosaic 0.7, HSV, random flip 0.5, CopyPaste

---

## 5. Evaluation Results (Validation Set)

| Metric          | Value                |
|------------------|----------------------|
| **Precision**     | 0.88                 |
| **Recall**        | 0.79                 |
| **mAP@0.5**       | 0.84                 |
| **mAP@0.5:0.95**  | 0.53                 |
| **Inference FPS** | 68 fps @ 640 × 640  |

>  Values based on best epoch — see training curves above.

---

## 6. Inference

```bash
yolo detect predict \
  model=weights/best.pt \
  source=demo/ \
  imgsz=640 conf=0.25
```

Sample output:

```
demo/img_001.jpg: 640x640 1 Person-Helmet, 0.10s  
demo/img_002.jpg: 640x640 1 Person-NoHelmet, 0.11s  
Results saved to runs/detect/predict
```

---

## 7. Re-training on Your Own Data

1. Label with **LabelImg** or **CVAT** and export to YOLO TXT.
2. Create a config file (e.g., `helmet.yaml`):

```yaml
path:  /absolute/path/to/dataset
train: images/train
val:   images/val
nc: 2
names: [Helmet, NoHelmet]
```

3. Run training using the command in Section 4.

---

## 8. Notes & Tips

- MobileNet + CARAFE achieves **2.5MB** model size with competitive performance.
- Export to ONNX for deployment:
  ```bash
  yolo export model=weights/best.pt format=onnx dynamic
  ```
- Tuning Suggestions:
  - Increase `imgsz=800` if GPU allows
  - Reduce `mosaic` after 80% epochs
  - Try `AdamW` + warm-up for noisy data

---

## 9. Citation

```bibtex
@misc{helmetyolov8n2025,
  title   = {Lightweight YOLOv8n for Safety-Helmet Detection},
  author  = {赵珽李 and 王雨意 and 谢淼},
  year    = {2025}
}
```

---

## 10. License

Released under the MIT License.
