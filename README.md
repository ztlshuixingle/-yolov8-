# 深度学习大作业——基于yolov8的施工现场安全帽检测

一个轻量级的**YOLOv8n 模型（采用 MobileNet 主干网络和 CARAFE 上采样机制）**，用于检测施工现场人员是否佩戴安全帽。

<p align="center">
  <img src="https://raw.githubusercontent.com/ztlshuixingle/-yolov8-/main/images/pic1.png" alt="Training curves" width="800">
</p>

---

## 1.项目架构

```
ultralytics/
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
├── 训练代码.py                  # Ultralytics CLI wrapper
├── 检验代码.py
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

### 5.模型训练

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

## 6.训练参数总结

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

## 7.模型分类评估结果 (验证集)

|分类结果指标          | 值               |
|------------------|----------------------|
| **Precision**     | 0.88                 |
| **Recall**        | 0.79                 |
| **mAP@0.5**       | 0.84                 |
| **mAP@0.5:0.95**  | 0.53                 |
| **Inference FPS** | 68 fps @ 640 × 640  |

> 该结果是基于训练结果最好的epoch计算得来

---


## 8.如何迁移使用我们的检测代码？

- 1.直接运行检测代码
- 2.在UI界面中直接上传本地的待检测的图片
- 3.预训练的模型输出检测结果

---

## 9.模型优势 & 改进建议

- 模型优势：MobileNet + CARAFE 结构使得模型只有**2.5MB,而初始模型有6.23MB**，轻量化改进后模型非常轻便且分类效果仍然较好。
- 模型未来改进建议:
  - 如果显卡性能允许，把图像尺寸 imgsz 提高到 800，可以提升精度。
  - 在训练后期（80% epoch 后）减少 mosaic 数据增强，模型能更稳定。
  - 如果数据噪声较大，可以尝试用 AdamW 优化器 + 学习率预热 策略。

---

## 10.实验说明

```bibtex
@misc{helmetyolov8n2025,
  title   = {基于yolov8的施工现场安全帽检测},
  author  = {赵珽李 and 王雨意 and 谢淼},
  year    = {2025}
}
```

---

