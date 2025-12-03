# YOLO VOC Tiny (PyTorch)

This project implements a compact YOLO-style detector in **PyTorch** for Pascal VOC 2012. The pipeline includes data loading, training, evaluation, and visualization utilities to track both classification and bounding-box regression quality over time.

## Dataset
- Default root: `E:/VOC` containing `VOC2012_train_val` and `VOC2012_test` with the standard `Annotations`, `ImageSets`, `JPEGImages`, `SegmentationClass`, and `SegmentationObject` folders.
- Train split: `VOC2012_train_val`; validation split: `VOC2012_test` (configurable through CLI flags).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
PYTHONPATH=src python src/train.py --data-root E:/VOC --train-set VOC2012_train_val --val-set VOC2012_test \
  --batch-size 8 --epochs 60 --lr 1e-3
```

Key outputs under `runs/`:
- `best.pt`: best-performing checkpoint (highest mAP@0.5).
- `history.json`: epoch-level loss and validation metrics.
- `training_curves.png`: loss and mAP@0.5 trends across epochs.
- `visualizations/epoch_*.jpg`: qualitative predictions vs. ground truth on held-out samples.
- `tensorboard/`: TensorBoard logs for interactive metric tracking.

Resume training with:
```bash
PYTHONPATH=src python src/train.py --resume runs/best.pt
```

Launch TensorBoard for live monitoring:
```bash
PYTHONPATH=src tensorboard --logdir runs/tensorboard --port 6006
```

## Model Highlights
- Tiny CSP-inspired backbone with a single detection scale (13x13 grid by default).
- Anchor-based target assignment and YOLO-style decoding with NMS filtering.
- Metrics: mAP@0.5 plus training loss tracking for both classification and regression quality.

## Notes
- All inline comments and docstrings remain in English.
- Adjust anchors, input resolution, or optimization settings in `src/yolo/config.py` for further experimentation.
