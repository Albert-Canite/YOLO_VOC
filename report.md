# v1 版本报告（PyTorch YOLO Tiny for VOC）

本版在原仓库基础上完全重写为 PyTorch 实现，包含数据加载、训练、评估与可视化的全流程。以下总结超参数、模型结构及使用要点，便于后续复现与扩展。

## 数据设置
- 数据根目录：`E:/VOC`，默认训练集 `VOC2012_train_val`，验证集 `VOC2012_test`（均可通过 CLI 修改）。
- 输入尺寸：`416 x 416`，采用 letterbox 方式保持长宽比并填充。
- 标签解析：忽略 `difficult=1` 的目标；从 XML 中读取 `xmin,ymin,xmax,ymax` 与类别，并同步缩放到填充后坐标系。

## 模型结构
- Backbone：CSP 风格的轻量级堆叠，通道序列 `3→32→64→128→256→512`，步幅 2 下采样至 `13x13` 网格。
- 主干单元：`Conv-BN-SiLU` 卷积块与 `CSPBlock`（1x1 压缩 + 3x3 恢复并残差相加）。
- 检测头：单尺度 1x1 卷积输出 `A*(5+C)` 通道，其中锚框 `A=3`，类别数 `C=20`。
- 锚框尺寸（像素，基于 416 输入）：`[(12,16), (19,36), (40,28)]`。

## 损失与训练超参数
- 损失：`DetectionLoss`（YOLO 风格）
  - 位置与尺寸：MSE（对 `(tx,ty)` 采用 sigmoid，`(tw,th)` 直接回归），权重 `lambda_box=5.0`。
  - 目标置信度：BCEWithLogits，前景权重 `lambda_obj=1.0`，背景权重 `lambda_noobj=0.5`。
  - 分类：BCEWithLogits 多标签表示（单类别 one-hot）。
- 优化：AdamW，`lr=1e-3`，`weight_decay=5e-4`，余弦退火调度 `T_max=epochs`。
- 训练：批大小 8，epoch=60，`num_workers=4`，随机种子 42，设备优先使用 CUDA。
- 目标分配：在 `13x13` 网格上按锚框与 GT 宽高 IoU 匹配最佳锚；存储 `(tx,ty,tw,th, obj, class_onehot)`。

## 指标与可视化
- 验证指标：mAP@0.5（基于 per-class PR 曲线，VOC 11 点插值）。
- 记录：`runs/history.json` 保存每轮 loss 与 mAP；`runs/training_curves.png` 绘制曲线；TensorBoard 事件存储于 `runs/tensorboard`。
- 质检图：`runs/visualizations/epoch_*.jpg`，每轮将模型预测（红框）与 GT（绿框）并排展示，附带类别与置信度。

## 运行示例
```bash
PYTHONPATH=src python src/train.py --data-root E:/VOC --train-set VOC2012_train_val --val-set VOC2012_test \
  --batch-size 8 --epochs 60 --lr 1e-3
```
- 恢复训练：`--resume runs/best.pt`。
- 监控：`tensorboard --logdir runs/tensorboard --port 6006`。

### 环境提示
- 全部代码基于 **PyTorch**，不依赖 TensorFlow。
- 若本地曾安装过旧版 TensorFlow，并在启动 TensorBoard 时出现 `AttributeError: module 'tensorflow' has no attribute 'io'`，可直接卸载 TensorFlow（`pip uninstall -y tensorflow tensorflow-gpu`），保留 `requirements.txt` 中的轻量级 `tensorboard` 即可。

## 后续改进思路
- 增加数据增强（颜色抖动、随机尺度、MixUp/CutMix）以提升泛化。
- 引入多尺度检测头或更丰富的 anchor 设置，适配小目标。
- 加入 EMA 模型权重、Warmup LR 以稳定早期训练。
- 在报告中记录实际训练曲线与定量结果，便于版本对比。
