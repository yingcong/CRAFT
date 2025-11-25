# 反向规格说明：CRAFT 摄像头相关特征增强

## 目标
- 对 ViPeR 行人重识别数据集的 HIPHOP 特征进行评估，并验证 Camera Correlation Aware Feature Augmentation（CRAFT）的增益。
- 在固定的训练/测试划分上，分别产出未使用 CRAFT 与使用 CRAFT 的 Rank-1/5/10/20 准确率。

## 数据与输入
- 数据集：`viper.mat`，包含特征矩阵 `X`（列为样本）、标签 `Label`、摄像头视角标签 `ViewLabel`。【F:demo.m†L16-L28】
- 划分：`train_test_split.mat` 中的 `trainPerson` 提供 10 组训练/测试划分，用于交叉评估。【F:demo.m†L16-L35】
- 可选参数：
  - `opt.useCRAFT`：是否启用 CRAFT 变换（0 关闭，1 启用）。
  - `opt.beta`：Camera View Discrepancy 正则系数，取值 (0,1]，默认示例为 0.6。【F:demo.m†L38-L53】

## 处理流程
1. **加载数据与划分**：读取 ViPeR 特征与 10 折人员划分，设置是否使用 CRAFT 的选项。【F:demo.m†L16-L41】
2. **基准评估（无 CRAFT）**：在 10 个划分上并行计算 CMC，得到 HIPHOP 原始特征的 Rank 精度均值。【F:demo.m†L18-L34】
3. **CRAFT 增强评估**：
   - 估计摄像头相关性矩阵 `omega`，基于跨摄像头子空间主角度的平均值对称化并归一化。【F:cameraCorrelation.m†L18-L52】
   - 对特征进行视角相关的增广与正则化：复制视角块、按 `omega` 加权，再通过正则矩阵特征分解进行白化获得增强特征 `Y`。【F:CRAFT.m†L21-L51】【F:CRAFT.m†L57-L64】
   - 在 10 个划分上评估增强特征的 CMC 均值。【F:demo.m†L38-L53】
4. **结果汇总**：分别输出未使用与使用 CRAFT 的 Rank-1/5/10/20 百分比准确率。【F:demo.m†L30-L54】

## 接口与约束
- 输入特征需按列组织，`view_label`/`Label` 与列一一对应，并且视角标签为整数编码。【F:CRAFT.m†L4-L9】
- 估计摄像头相关性时，单视角样本少于 2 会退化为相关性 1；为加速计算仅随机采样最多 100 个样本估计主角度。【F:cameraCorrelation.m†L23-L50】
- 变换矩阵构造使用核矩阵 `K`；在线性空间可直接传入单位阵，保证增广块一致性。【F:CRAFT.m†L9-L10】【F:CRAFT.m†L43-L51】

## 运行方式
- 在 MATLAB 中运行 `demo.m` 即可复现流程，支持使用 `parfor` 并行处理 10 次划分评估。【F:demo.m†L18-L53】

## 输出
- 控制台输出两组 CMC 百分比：HIPHOP 原始特征与 CRAFT 增强特征的 Rank1/5/10/20 准确率，便于对比增益。【F:demo.m†L30-L54】
