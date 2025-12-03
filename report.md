# v1 训练配置与模型说明

本版本实现了一个轻量级 YOLO 风格的 Pascal VOC 目标检测器，核心信息如下：

## 数据与预处理
- 默认数据根目录：`E:/VOC`，分别使用 `VOC2012_train_val` 与 `VOC2012_test` 作为训练与验证分割。
- 输入分辨率：`416 x 416`，采用 letterbox 保持纵横比并填充。
- 标签解析：从 `Annotations` 读取 XML，生成边框与类别标签；同时保存原始尺寸用于可视化对比。

## 模型结构
- Backbone：多层 `ConvBlock(Conv-BN-LeakyReLU)` 叠加与 6 次 `MaxPool` 下采样，输出通道 1024。
- 检测头：`1x1` 卷积生成 `A x (5 + C)` 通道（其中 A=3 个锚框，C=20 个类别）。
- 网格尺寸：`13 x 13`，锚框设置为 `(10,13),(16,30),(33,23)`；输出以 `(B, A, S, S, 5+C)` 的张量给出。

## 损失与训练超参数
- 损失函数：
  - 位置与尺寸：MSE，权重 `lambda_box=5.0`。
  - 置信度：BCEWithLogits，权重 `lambda_obj=1.0`，无目标时 `lambda_noobj=0.5`。
  - 分类：BCEWithLogits。
- 优化器：Adam，`lr=1e-3`，`weight_decay=5e-4`，余弦退火调度器。
- 训练批大小：8；线程数：4；默认训练 50 epoch（可通过命令行参数覆盖）。
- 随机种子固定为 42，支持 GPU/CPU 自动切换。

## 指标与可视化
- 验证阶段计算 `mAP@0.5`，并在 `runs/history.json` 与 TensorBoard 中追踪。
- `runs/training_curves.png` 展示损失与 mAP 随 epoch 变化曲线。
- `runs/visualizations/epoch_*.jpg` 提供模型预测与 GT 的并排对比示例，便于直观检查分类与回归质量。

## 使用提示
- 通过命令行参数可自定义数据路径、分割名称、训练 epoch、学习率等。
- 恢复训练时指定 `--resume runs/best.pt`，以继续优化并保持历史指标记录。
