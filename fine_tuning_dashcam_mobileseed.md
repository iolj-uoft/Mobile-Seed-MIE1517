# 🧠 Fine-Tuning Mobile-Seed on Custom Dashcam Data (MMSegmentation)

This guide explains how to fine-tune a pretrained `Mobile-Seed` segmentation model (trained on Cityscapes) using your own labeled dashcam dataset in MMSeg.

---

## 📁 Step 1: Create Your Custom Dataset Config

**Path:** `configs/_base_/datasets/dashcam_1024x1024.py`

```python
dataset_type = 'CustomDataset'  # Or 'CityscapesDataset' if using same format
data_root = 'data/my_dashcam'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=2,  # Adjust based on GPU
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=train_pipeline)
)
```

---

## ⚙️ Step 2: Create Your Fine-Tuning Config

**Path:** `configs/Mobile_Seed/MS_tiny_dashcam.py`

```python
_base_ = [
    '../_base_/models/Mobile_Seed.py',
    '../_base_/datasets/dashcam_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='./ckpt/MS_tiny_cityscapes.pth',  # Load your Cityscapes model
    backbone=dict(type='AFFormer_for_MS_tiny'),
)

optimizer = dict(
    type='AdamW',
    lr=0.00005,  # 🔁 Smaller LR for fine-tuning
    betas=(0.9, 0.999),
    weight_decay=0.01
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)

load_from = './ckpt/MS_tiny_cityscapes.pth'  # Or wherever your model is
```

---

## 🚀 Step 3: Train Your Fine-Tuned Model

```bash
python tools/train.py configs/Mobile_Seed/MS_tiny_dashcam.py --gpu-id 0
```

Output checkpoints will be saved to:

```
work_dirs/MS_tiny_dashcam/latest.pth
```

---

## 🧪 Optional: Evaluate or Visualize

```bash
python tools/test.py configs/Mobile_Seed/MS_tiny_dashcam.py \
    work_dirs/MS_tiny_dashcam/latest.pth \
    --eval mIoU
```

---

## ✅ Notes

- You can label your dataset using tools like [LabelMe](https://github.com/wkentaro/labelme), [CVAT](https://github.com/opencv/cvat), or [Supervisely](https://supervise.ly/).
- If you only label a few classes (e.g., `car`, `road`, `person`), update `num_classes` and `classes` in your dataset accordingly.
- For small datasets, freezing the backbone can help stability:
  ```python
  model = dict(
      backbone=dict(frozen_stages=4)
  )
  ```

---

Need help with dataset formatting, label mapping, or pseudo-label bootstrapping? Just ask!
