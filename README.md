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

## 2. 数据集

| Split | Images | Annotation format |
|-------|--------|-------------------|
| Train | 2 607  | YOLO TXT (`class xc yc w h`) |
| Val   |   682  | YOLO TXT |

- **一共有两个分类**: `0 = 没带头盔的人`, `1 = 带头盔的人`
- 所有图片的大小在训练中都被充值为 **640 × 640** 

---

## 3. 建模

| 组成   | 细节 |
|-------------|---------|
| **Base**    | YOLOv8n (Ultralytics) |
| **Backbone**| MobileNet-V3 Small |
| **Neck**    | FPN + CARAFE |
| **Head**    | YOLOv8 decoupled head |
| **Params**  | ≈ 3.2M |
| **Input**   | 640 × 640 RGB |

---

## 4. 实验环境和训练配置

```bash
conda create -n yolo-helmet python=3.9
conda activate yolo-helmet
pip install ultralytics==8.1.0 torch>=1.13 torchvision
```

> Tested on **CUDA 11.7 + RTX 3060 (12 GB)**; CPU-only also supported.

###  模型训练

```bash
yolo detect train \
  model=models/yolov8n_mobilenet_carafe.yaml \
  data=datasets/images/train \
  epochs=300 imgsz=640 batch=16 \
  optimizer=SGD lr0=0.01 weight_decay=0.0005 \
  name=yolov8n_mobilenet_carafe
```

- Early stopping `patience=50`
- Augmentations: Mosaic 0.7, HSV, random flip 0.5, CopyPaste

---

## 5. 模型分类评估结果 (验证集)

|分类结果指标          | 值               |
|------------------|----------------------|
| **Precision**     | 0.88                 |
| **Recall**        | 0.79                 |
| **mAP@0.5**       | 0.84                 |
| **mAP@0.5:0.95**  | 0.53                 |
| **Inference FPS** | 68 fps @ 640 × 640  |

>  Values based on best epoch — see training curves above.

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

## 9. 实验说明

```bibtex
@misc{helmetyolov8n2025,
  title   = {Lightweight YOLOv8n for Safety-Helmet Detection},
  author  = {赵珽李 and 王雨意 and 谢淼},
  year    = {2025}
}
```

---

## 10. 训练参数总结

这部分主要是我们本次训练的时候使用到的一些超参数:

| Parameter         | Value     | Description |
|------------------|-----------|-------------|
| `epochs`         | 300       | Total training rounds |
| `batch`          | 16        | Batch size per iteration |
| `imgsz`          | 640       | Input image resolution |
| `optimizer`      | auto      | Auto-selected (default: SGD) |
| `lr0`            | 0.01      | Initial learning rate |
| `momentum`       | 0.937     | Momentum (for SGD) |
| `weight_decay`   | 0.0005    | Weight regularization |
| `warmup_epochs`  | 3.0       | Warm-up phase duration |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 1.5 | Loss component weights |
| `mosaic`         | 1.0       | Mosaic augmentation enabled |
| `fliplr`         | 0.5       | Probability of horizontal flip |
| `device`         | 0         | GPU device used |
| `patience`       | 50        | Early stopping patience |
| `save_dir`       | runs/detect/train3 | Output directory for logs and weights |
