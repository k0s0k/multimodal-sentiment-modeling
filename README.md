# 多模态情感预测与解释

复杂场景下的多模态情感建模代码，包括时间对齐与特征提取、缺失模态鲁棒预测、Shapley/Owen 贡献解释与证据定位。

## 1. 代码与资源

| 目录 | 用途 |
| --- | --- |
| `q1/` | 五分支特征提取、时序对齐及诊断 |
| `sentiment/` | 缺失模拟、训练、评价与预测 |
| `explanation/` | 可解释模型、归因、忠实性及统计验证 |
| `scripts/` | 数据准备、编码器下载、集成训练与专项预测入口 |
| `configs/` | 定稿配置、数据划分和实验配方 |
| `environment/` | 依赖清单与原实验环境记录 |
| `models/` | 分词器配置和定稿模型参数索引 |
| `results/` | 第三问解释所需的时间定位、词元映射和输出表头 |

本仓库包含算法代码及必要的复现元数据。原始视频、官方特征、模型权重、完整实验结果及论文需另行准备。元数据沿用官方样本与原实验编号。

### 数据和权重准备

1. 准备官方附件 1–4；各入口通过参数接收本地路径。
2. 安装对应环境的依赖，并按 GPU 平台安装 PyTorch；下列版本记录反映原实验环境。
3. 执行 `python scripts/download_encoder.py` 获取固定版本 BERT 权重。
4. 从原支撑材料复制 `models/member_1.pt`、`models/member_2.pt`、`models/member_3.pt`，即可按冻结协议复现专项预测。也可按训练入口训练新的模型。

定稿专项预测入口校验原权重和官方输入身份。重新训练得到的参数应使用相应的新配置与模型入口评价，不能直接作为原冻结权重替换。

仓库未附第三方模型或数据的再分发授权；相关使用遵循其原始许可。本次整理仅做语法、配置与文件依赖检查，未重新训练模型。

## 2. 环境依赖

推理实测环境：Python 3.12.11、PyTorch 2.13.0+cu132、CUDA 13.2、NumPy 2.3.5、transformers 4.57.1、tokenizers 0.22.2。平台兼容的PyTorch需单独安装，其余版本见`environment/requirements.txt`；记录版本用于复核原计算环境。第一问的两类特征环境分别记录于`environment/q1_text_audio.json`和`q1_acoustic_visual.json`。

公共冻结编码器为`google-bert/bert-base-uncased`，固定版本`86b5e0934494bd15c9632b12f734a8a67f723594`，权重SHA256为`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。从本目录运行：

```bash
python scripts/download_encoder.py
```

已有该版本权重时，可用`--local-file /path/to/model.safetensors`导入并校验。包内提供编码器结构与词表文件；公共预训练权重由固定版本获取。本题学习的全部三成员FP32参数、训练集标准化统计、类别先验和回归先验均随包提供。共享模型只存一份，checkpoint保留推理与从头重训配置，优化器历史状态从附件中移除。`models/manifest.json`逐项记录原checkpoint、打包文件及全部349个状态张量的哈希。

## 3. 第一问

```bash
python q1/tools/read_q1_features.py --features q1/features
```

每条样本包含文本1024维、语音1024维、声学25维、面部44维、视频768维五分支特征。特征为FP32，时间为FP64；每个输出保留独立有效掩码、时间区间与原素材身份。文本及语音未知词时间为NaN对；视频分支为`[-1,-1]`并伴随有效性0、覆盖率0。

从原视频重新提取时，先解压附件1到原`E_data`相对结构，再建立新目录：

```bash
python scripts/stage_features.py --official-root /path/to/E_data --output /path/to/q1_reproduce
cd /path/to/q1_reproduce
Q1_PYTHON=/path/to/python bash setup_alignment_dependencies.sh
bash setup_openface.sh --build-dlib
Q1_PYTHON=/path/to/python bash reproduce.sh
```

基础Python和FFmpeg环境见`environment/feature_requirements.txt`；OpenFace编译依赖见安装脚本。独立Qwen与Whisper依赖采用脚本给定的隔离目录。RoBERTa、WavLM、VideoMAE、CTC、Qwen和Whisper版本均固定于`q1/config.json`，实际文件SHA记录于对应样本元数据。该流程依次完成统一时钟、CTC候选、独立边界诊断、保守接受、五分支特征提取、全量核验及导出。异常样本及未知时间依照掩码保留。

第一问表征诊断使用独立Python 3.11环境，依赖见`environment/diagnostic_requirements.txt`。在`q1/`下运行`python -m feature_extraction.experiments --index features/samples_index.csv --labels data/labels.csv --manifest data/manifest.jsonl --output /path/to/new_diagnostics --tag reproduced`。

## 4. 第二问

准备附件3的`aligned_50`目录，从本目录执行：

```bash
python scripts/predict_missing.py --input /path/to/attachment3/aligned_50 \
  --checkpoints models/member_1.pt models/member_2.pt models/member_3.pt \
  --model-path models/bert-base-uncased --freeze configs/missing_protocol.json \
  --output /path/to/new_results/attachment3_predictions.csv \
  --details-prefix /path/to/new_results/attachment3_details --device cuda:0
```

该入口保留附件3原有局部缺失。类别顺序为Negative、Neutral、Positive；强度范围为[-3,3]。计算使用FP32公共BERT、FP16缓存舍入再转FP32、BF16融合及FP32概率等权平均，对应第二问冻结结果。

## 5. 第三问

```bash
python scripts/predict_explain.py --input /path/to/attachment4/aligned_50 \
  --bert models/bert-base-uncased --output /path/to/new_explanations --device cuda:0
```

第三问采用FP32编码与融合、关闭TF32、batch64。计算全部8个模态子集、四个输出目标的局部Owen贡献与20%预算证据。原确定性种子、排列生成和16→64对称采样协议固定保留。`--predictions-only`可单独输出20条数值预测。

## 6. 从头训练模型

```bash
python scripts/train_ensemble.py --source /path/to/attachment2/aligned_50.pkl \
  --bert models/bert-base-uncased --output /path/to/new_training --device cuda:0
```

训练使用3395条训练样本与固定492条开发样本；236条确认数据、727条测试数据分别用于定稿评价。原验证视频分组见`configs/validation_split_v1.json`。训练使用batch64、AdamW、学习率0.0003、权重衰减0.01、50轮调度范围、5%预热、梯度裁剪1.0。三个种子分别为20260924、20260925、20260926。

第二问对照及消融配置位于`configs/training/`，冻结特征模型使用`sentiment.train`入口，在线微调使用`sentiment.finetune`；含知识蒸馏的配置通过`--teacher`提供相应训练集教师。第三问46次成功实验的配置与五折训练/验证索引保存在`configs/explanation_training/`，可用`python -m explanation.train --config CONFIG --source COMPACT_ALIGNED_PKL --bert BERT_DIRECTORY --output NEW_RUN --device cuda:0`复现固定配方。索引列表随配置固化。

`sentiment.metrics`提供Accuracy、Macro-F1、MAE、Pearson及风险汇总；`sentiment.masking`、`diagnostics_masks`生成缺失类型、位置、时长与分布工况。`explanation.faithfulness`提供删除、保留与同预算随机对照评价，`explanation.statistics`提供2000次视频簇重采样及配对区间。