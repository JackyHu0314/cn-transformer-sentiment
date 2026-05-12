# 中文外卖评论情感分析

这是我做的一个中文情感分析小项目，主要目的是练习 **Transformer 迁移学习**。

一开始我只是想跑通一个简单的情感二分类模型，后来发现真实评论里有很多“味道不错，但是送餐太慢”这种有好有坏的句子，所以这个项目最后不只是训练模型，也包括了测试集评估和错例分析。

目前项目完成了这些内容：

```text
公开数据集准备
  ↓
预训练中文 Transformer 微调
  ↓
单句 / 多句情感预测
  ↓
测试集评估
  ↓
错例导出和分析
```

## 项目目标

这个项目不是为了做一个商业级情感分析系统，而是为了完整走一遍中文 NLP 迁移学习流程。

我主要关注这几个问题：

- 预训练中文模型怎么迁移到情感分类任务上
- `Tokenizer`、`input_ids`、分类头这些东西在训练里怎么串起来
- 全量微调和只训练分类头有什么区别
- 只看 accuracy 是否足够
- 真实评论中混合情绪和标签噪声会怎么影响模型

## 技术路线

项目使用 PyTorch 和 Hugging Face Transformers。

模型不是从零训练的，而是加载中文预训练模型，然后接一个二分类分类头进行微调：

```text
中文评论
  ↓
Tokenizer
  ↓
input_ids / attention_mask
  ↓
中文 RoBERTa / BERT Encoder
  ↓
[CLS] 句向量
  ↓
Linear 分类头
  ↓
负面 / 正面
```

默认小模型是：

```text
uer/chinese_roberta_L-2_H-128
```

我实际训练时使用了更大的模型：

```text
uer/chinese_roberta_L-8_H-512
```

训练方式是 **全量微调**，也就是 Transformer 主体和最后的分类头都会更新。

## 项目结构

仓库中主要保留代码、说明文档和一个小样例数据：

```text
.
├── data/
│   └── sentiment_sample.csv
├── prepare_data.py
├── train.py
├── predict.py
├── evaluate.py
├── .gitignore
├── requirements.txt
└── README.md
```

运行数据准备和训练后，会额外生成这些文件或目录：

```text
data/waimai_10k_train.csv
data/waimai_10k_val.csv
data/waimai_10k_test.csv
models/
outputs/
```

这些生成文件已经写入 `.gitignore`，不建议直接提交到 GitHub。

主要文件说明：

| 文件 | 作用 |
| --- | --- |
| `prepare_data.py` | 下载并整理公开中文情感数据集 |
| `train.py` | 微调预训练 Transformer |
| `predict.py` | 加载模型做情感预测 |
| `evaluate.py` | 在测试集上评估，并导出错例 |
| `data/sentiment_sample.csv` | 一个很小的样例数据，只用于跑通流程 |

## 环境

推荐使用 conda 环境，不建议直接用太新的 Python 版本。

我本机最终使用的是：

```text
Python 3.10
PyTorch CUDA
NVIDIA RTX 5060
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果重新建环境，可以这样：

```powershell
conda create -n sentiment-transformer python=3.10 -y
conda activate sentiment-transformer
pip install -r requirements.txt
```

## 数据

训练数据格式是 CSV：

```csv
text,label
这个电影太好看了,1
这家店服务很差,0
```

标签含义：

```text
0 = 负面
1 = 正面
```

项目里提供了一个小样例数据：

```text
data/sentiment_sample.csv
```

这个样例只适合验证代码能不能跑，不适合训练真正可用的模型。

正式实验使用的是公开外卖评论数据集 `waimai_10k`。整理后数据量是：

```text
训练集：9584 条
验证集：1198 条
测试集：1198 条
总数据：11980 条
```

准备数据：

```powershell
python prepare_data.py --dataset waimai_10k
```

生成：

```text
data/waimai_10k_train.csv
data/waimai_10k_val.csv
data/waimai_10k_test.csv
```

## 训练

用小样例跑通流程：

```powershell
python train.py
```

使用外卖评论数据训练：

```powershell
python -u train.py `
  --model_name uer/chinese_roberta_L-8_H-512 `
  --train_file data/waimai_10k_train.csv `
  --val_file data/waimai_10k_val.csv `
  --output_dir outputs/sentiment-roberta-medium `
  --epochs 5 `
  --batch_size 32 `
  --learning_rate 3e-5 `
  --max_length 128 `
  --fp16
```

如果模型已经下载到本地缓存，且网络不可用，可以加：

```powershell
--local_files_only
```

`train.py` 会保存验证集表现最好的模型。

## 预测

训练完成后可以预测一句或多句：

```powershell
python predict.py --model_dir outputs/sentiment-roberta-medium --text "配送很快，味道也不错"
```

多句预测：

```powershell
python predict.py --model_dir outputs/sentiment-roberta-medium --text "味道很好，下次还会点" "等了两个小时，菜都凉了"
```

输出类似：

```text
正面    0.9821    味道很好，下次还会点
负面    0.9473    等了两个小时，菜都凉了
```

## 测试集结果

我用 `waimai_10k_test.csv` 做了测试集评估：

```powershell
python evaluate.py --model_dir outputs/sentiment-roberta-medium --test_file data/waimai_10k_test.csv
```

结果：

```text
准确率：0.9115
```

简单来说，大概就是测试集里有 91% 左右的评论判断对了。

## 错例分析

`evaluate.py` 会把错例导出到：

```text
outputs/sentiment-roberta-medium/mistakes.csv
```

我看了一些错例后发现，很多“错误”其实不完全是模型的问题，而是数据标签本身有噪声。

例如这些样本被标成正面，但模型判断为负面：

```text
true=正面 pred=负面 | 等了两个小时。
true=正面 pred=负面 | 送餐太慢了！！！！！
true=正面 pred=负面 | 小黄鱼贴饼子根本不是贴饼子，，炸窝头！都黑了！
```

这些样本被标成负面，但模型判断为正面：

```text
true=负面 pred=正面 | 速度快，态度好
true=负面 pred=正面 | 不错，薯条特别好吃
true=负面 pred=正面 | 豆浆超级不错
```

还有一类是真正比较难的混合情绪：

```text
披萨很好，但是希望电话通知
送的挺快，就是加多宝变成可口可乐了
有点腻，味道还可以，不能多吃
```

这类评论很难简单地归成正面或负面。这个现象说明，真实场景里情感分析不一定适合只做二分类，后续可以考虑加入“中性/混合”类别。

## 我学到的点

这次项目里比较有收获的地方：

- 迁移学习不是从零写 Transformer，而是复用预训练模型的语言理解能力
- 情感分类头本质上是接在 Transformer 后面的一个小分类器
- 全量微调会同时更新预训练模型主体和分类头
- 验证集和测试集比训练集准确率更有参考价值
- 数据标签质量会明显影响模型上限
- 只看一个准确率数字不够，还要看看模型具体错在哪些句子上

## 代码里主要做了什么

这个项目虽然规模不大，但我尽量把它做成了一个完整的训练流程，而不是只调用一个现成模型做预测。

主要实现点包括：

- 读取 CSV 格式的评论数据和标签
- 把中文评论转换成模型能处理的输入
- 加载中文预训练模型，并训练成正面 / 负面分类模型
- 写了训练、预测和测试几个脚本
- 训练时保存验证集效果比较好的模型
- 测试时把预测错的评论导出成 CSV，方便后面查看

## 项目目前的不足

这个项目主要还是学习性质的，所以也有一些明显不完善的地方：

- 数据主要是外卖评论，范围还比较窄。
- 现在只判断正面和负面，对“有好有坏”的评论处理得不够好。
- 参数主要是按经验试的，还没有做很多组对比。

比如下面这种评论，单纯分成正面或负面就有点勉强：

```text
味道还可以，就是等得有点久
包装挺好的，但是菜有点凉
价格不贵，不过分量有点少
```

这些句子不是完全好，也不是完全差，所以模型有时候会判断得不太稳定。

## 后续可以做的方向

如果后面还有时间，可以从这些比较实际的方向继续改：

- 手动整理一部分错例，把明显标错的数据改一下，再重新训练看看效果。
- 增加“中性”或“混合情绪”类别，让模型不只是强行判断正面或负面。
- 做一个简单的网页或小接口，输入一句评论后直接显示判断结果。
- 换一些别的数据集试试，比如商品评论、电影评论或酒店评论。

比如以后可以把“味道还行”这类评论单独放到中性，把“味道不错但是配送慢”这类评论放到混合情绪，这样会比现在只分正负更细一点。

## 数据说明

本项目使用公开数据集做学习实验，没有自己去抓平台数据。

如果后面换成自己的评论数据，按前面的 CSV 格式整理就可以。

## 关于许可证

这个仓库目前主要作为我的学习和课程展示项目，所以暂时没有添加开源许可证。

如果后续希望别人可以自由复用这份代码，可以再补充 MIT License。
