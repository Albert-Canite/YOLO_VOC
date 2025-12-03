# VOC Split Usage Guide

This project trains and validates on the annotated portion of PASCAL VOC 2012. By default all loaders reference the `VOC2012_train_val` folder:

- **Training images/labels** come from `ImageSets/Main/trainval.txt` inside `VOC2012_train_val`, paired with matching JPEGs and XML annotations.
- **Validation images/labels** come from `ImageSets/Main/val.txt` inside the same `VOC2012_train_val` folder. This keeps training and evaluation on the subset that ships with annotations.
- The **test set (`VOC2012_test`) is not used for training or metric reporting** because it does not ship with public annotations; it is only suitable for inference-only visualization if you manually point the paths at that folder.

If you need to change the behavior:

- Edit `config.py` or provide a JSON override to swap `train_split`/`val_split` (for example, to use `train.txt` for training and `val.txt` for validation). Both fields expect the base folder name (e.g., `VOC2012_train_val`) and the code will derive the JPG/XML paths plus the split file names automatically.
- Ensure any custom split file lists IDs that exist in both the `JPEGImages` and `Annotations` subfolders. Missing annotations will trigger a `FileNotFoundError` during loading.

With the defaults, training/validation strictly consume the annotated `train_val` data; the `VOC2012_test` directory is untouched unless you explicitly reconfigure it for inference.
