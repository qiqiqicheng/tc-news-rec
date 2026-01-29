本目录包含针对由于天池新闻推荐入门赛的完整训练与预测代码。方案基于 HSTU (Hierarchical Sequential Transduction Unit) 序列推荐模型，采用 Code Submission 形式，支持从原始数据从头训练并生成预测结果。

## 1. 环境依赖

- **操作系统**: Linux
- **Python 版本**: 3.10+
- **主要依赖**:
    - PyTorch 2.0+
    - PyTorch Lightning
    - Hydra-core
    - Pandas, NumPy
- **安装方式**:
    ```bash
    pip install -r requirements.txt
    ```

## 2. 运行说明

本方案提供了一键运行脚本 `run.sh`，整合了数据预处理、模型训练及结果预测的全流程。

### 快速复现

在当前目录下（`code/`）直接运行：

```bash
sh run.sh
```

该脚本将依次执行以下步骤：
1.  **数据预处理**: 读取 `../tcdata` 下的原始 CSV 文件，生成序列特征及 ID 映射表，保存至 `../user_data/processed`。代码会自动识别测试集文件名（如 `testA` 或 `testB`）。
2.  **模型训练**: 使用 `infonce_deeper_model` 配置进行训练，日志与 Checkpoint 自动保存至 `../user_data/logs` 和 `../user_data/model_data`。
3.  **结果预测**: 加载训练得到的最佳 Checkpoint (`best_model.ckpt`)，对测试集用户生成 Top-5 推荐，最终结果保存为 `../prediction_result/result.csv`。

## 3. 算法与模型简介

### 模型结构
- **核心架构**: 采用 **HSTU (Hierarchical Sequential Transduction Unit)** 作为序列编码器，相较于传统的 SASRec/Transformer，在高稀疏、长序列场景下具有更好的收敛速度和效果。
- **输入特征**:
    - 用户点击文章序列 (Sequence of Item IDs)
    - 文章侧特征：Category ID, Created Time bucket, Word Count bucket
    - 上下文特征：Timestamp bucket, Age (click_time - publish_time)
- **损失函数**: InfoNCE Loss，并在训练中开启了 **Hard Negative Mining** 以提升模型对困难样本的区分能力。

### 预处理策略
- **动态分桶**: 对时间戳、文章字数等连续特征采用 Global Quantile Bucketing (100~10000 桶)，保证训练集与测试集特征分布的一致性。
- **鲁棒性设计**: 针对 Code Submission 场景，预处理脚本支持动态加载未知命名的测试集文件，并自动维护 User/Item ID 的双向映射。

## 4. 文件结构说明

```
code/
├── run.sh                  # 全流程主入口脚本
├── requirements.txt        # Python 依赖清单
├── README.md               # 说明文档
├── tc_news_rec/            # 核心代码包
│   ├── data/               # 数据预处理逻辑 (preprocessor.py)
│   ├── models/             # 模型定义 (HSTU, LightningModule)
│   └── scripts/            # 执行脚本 (train.py, predict.py, prepare_data.py)
├── configs/                # Hydra 配置文件
│   ├── experiment/         # 实验级配置 (infonce_deeper_model.yaml 等)
│   └── ...
└── tests/                  # 单元测试 (含预处理鲁棒性测试)
```

## 5. 其他说明
- **训练超参**: 默认配置在 `configs/experiment/infonce_deeper_model.yaml` 中定义。
- **显存要求**: 建议使用 8GB+ 显存的 GPU 进行训练。如遇 OOM，可在 `configs/data/default.yaml` 中调小 `batch_size`。
