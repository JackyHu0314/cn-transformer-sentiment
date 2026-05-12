# AGENTS.md

## 项目概览

这是一个中文外卖评论情感分析学习项目，主要用 PyTorch 和 Hugging Face Transformers 微调中文预训练模型，完成正面 / 负面二分类。

README 面向学习展示和课程说明，文字保持中文、朴素、真实，不要过度包装成商业级项目。

## 环境和依赖

- 推荐使用 Python 3.10 和 conda 环境。
- 安装依赖：`pip install -r requirements.txt`
- 主要依赖：`torch`、`transformers`、`datasets`、`tqdm`

## 常用命令

- 准备公开数据：`python prepare_data.py --dataset waimai_10k`
- 用样例数据跑通训练：`python train.py`
- 测试集评估：`python evaluate.py --model_dir outputs/sentiment-roberta-medium --test_file data/waimai_10k_test.csv`
- 单句预测：`python predict.py --model_dir outputs/sentiment-roberta-medium --text "配送很快，味道也不错"`

## 源码布局

- `prepare_data.py`：下载并整理公开数据集。
- `train.py`：训练 / 微调情感分类模型。
- `predict.py`：加载模型进行单句或多句预测。
- `evaluate.py`：测试集评估并导出错例。
- `data/sentiment_sample.csv`：很小的样例数据，只用于跑通流程。
- `models/`、`outputs/` 和整理后的数据文件是生成内容，通常不提交。

## 编码约定

- 保持脚本结构简单，优先沿用现有 `argparse` 命令行参数风格。
- 文档说明使用中文，语气贴近学习项目，不要写得过于夸张或专业。
- 只做和任务直接相关的改动，避免顺手重构无关代码。
