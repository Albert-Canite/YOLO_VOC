# 项目报告（v1）

## 数据与目录
- 数据路径默认为 `E:/VOC`，训练集与验证集均指向 `VOC2012_train_val`（训练列表使用 `trainval.txt`，验证列表使用 `val.txt`），内部沿用官方 VOC 目录结构（Annotations、ImageSets/Main、JPEGImages 等）。
- 支持通过 `config.py` 或自定义 JSON 覆盖路径与超参数。

## 模型结构
- 采用轻量化 TinyYOLO 变体：
  - Backbone：多层 `ConvBlock(Conv2d+BN+LeakyReLU)` 与 5 次最大池化，下采样到 13x13 特征图。
  - 检测头：`1x1 Conv` 输出 `num_anchors*(5+num_classes)`，默认 9 组 VOC 常用 anchor。
- 解码：使用网格偏移与 anchor 尺寸恢复到原图尺度的中心点与宽高，再转换为 `xyxy` 方便 NMS。

## 训练配置
- 输入尺寸：416x416；批大小：8；优化器：SGD（lr=1e-3，momentum=0.9，weight_decay=5e-4）。
- 训练轮数：50（可调），前 3 个 epoch 预留 warmup（当前实现保持基础学习率，可在配置中扩展调度）。
- 混合精度：默认开启（自动使用 `torch.cuda.amp`）。
- 数据增强：Resize 到目标尺寸，`ColorJitter` 轻量颜色增强。

## 损失与度量
- 损失：YOLO 风格的坐标 MSE（x/y/w/h）+ 目标/分类的 BCE，共同归一化到 batch 平均。
- 评估：
  - 解码后做置信度阈值与 NMS，计算 `mAP@0.5`（简化的每类 AP 积分）。
  - 训练日志每个 epoch 打印 `loss` 与 `mAP@50`，并写入 `outputs/<timestamp>/history.json`。

## 可视化
- `visualize.py`：
  - `plot_history` 生成训练过程曲线（loss 与 mAP）。
  - `visualize_samples` 从验证集抽样绘制预测框与 GT 对比，输出 `pred_*.png` 与 `gt_*.png`。
- 运行示例：
  ```bash
  python visualize.py --history outputs/<run>/history.json \
    --checkpoint outputs/<run>/checkpoint_50.pth --output outputs/vis
  ```

## 使用流程
1. 准备数据集到 `E:/VOC`（或在自定义 JSON 中修改 `root` 等字段）。
2. 运行训练：
   ```bash
   python train.py
   ```
   或使用自定义配置：
   ```bash
   python train.py --config my_config.json
   ```
3. 训练结束后，`history.json` 记录指标，可用 `visualize.py` 绘制曲线并生成样例可视化。

## 后续优化方向
- 添加学习率 warmup/余弦调度。
- 引入更丰富的数据增强（mosaic、随机翻转）。
- 扩展多尺度评估与 `mAP@0.5:0.95` 以对标 COCO 指标。
