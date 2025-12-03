# YOLO VOC Lightweight Detector

This repository implements a lightweight YOLO-style detector tailored for the Pascal VOC 2012 dataset. The code includes training, evaluation, and visualization utilities to monitor classification and bounding-box regression performance over time.

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Make sure the Pascal VOC dataset is available locally at `E:/VOC` (default). Each split should contain `Annotations`, `ImageSets`, `JPEGImages`, `SegmentationClass`, and `SegmentationObject`.

## Training
Run training with default hyper-parameters:
```bash
PYTHONPATH=src python src/train.py --data-root E:/VOC --train-set VOC2012_train_val --val-set VOC2012_test
```

Key outputs are stored under `runs/`:
- `history.json`: epoch-wise loss and mAP@0.5 metrics.
- `training_curves.png`: visualization of loss and mAP trends.
- `visualizations/epoch_*.jpg`: qualitative predictions versus ground truth.
- `tensorboard/`: TensorBoard event files for interactive metric tracking.

Resume training from a checkpoint:
```bash
PYTHONPATH=src python src/train.py --resume runs/best.pt
```

## Inference and Visualization
`training_curves.png` and the qualitative `visualizations` folder summarize how the model improves during training. You can open TensorBoard for interactive metric inspection:
```bash
PYTHONPATH=src tensorboard --logdir runs/tensorboard --port 6006
```

## Notes
- All code comments and docstrings are written in English as required.
- Default anchors and image size target a compact network suited for VOC resolution; adjust `src/yolo/config.py` for experimentation.
