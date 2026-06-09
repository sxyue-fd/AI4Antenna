# Forward Proxy CNN for Pixelated Antenna Dataset

## 项目简介

本项目用于构建一个基于 CNN 的前向代理模型，学习如下映射关系：

**像素化天线结构 + 馈电位置 -> S11 频响 + 辐射方向图**

该项目适配一个 HDF5 数据集，数据集由 MATLAB 脚本生成，包含：

- `/X`：输入结构张量
- `/Y`：S11 线性幅值
- `/pattern`：方向图张量

项目采用多任务学习方式：

- 一个共享 CNN backbone 提取结构特征
- 一个 S11 预测头输出频响向量
- 一个方向图预测头输出方向图张量

---

## 数据格式

### 输入

`/X [B, 2, N, N]`

- 第 1 通道：metal mask
- 第 2 通道：feed mask

### 输出

`/Y [B, F]`

- S11 线性幅值

`/pattern [B, Fp, P, T]`

- 方向图张量
- 通常：
  - `Fp = 11`
  - `P = 4`
  - `T = 120`

---

## 项目目录

```text
forward_proxy_cnn/
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── infer.py
├── configs/
│   ├── __init__.py
│   └── default_config.py
├── data/
│   ├── __init__.py
│   ├── h5_dataset.py
│   ├── split_loader.py
│   ├── preprocess.py
│   ├── build_preprocessed_h5.py
│   └── datamodule.py
├── models/
│   ├── __init__.py
│   ├── blocks.py
│   ├── backbone.py
│   ├── heads.py
│   ├── forward_proxy_net.py
│   └── losses.py
├── engine/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── io.py
│   ├── logger.py
│   ├── checkpoint.py
|   |—— plot_loss.py
│   └── visualize.py
├── checkpoints/
├── outputs/
│   ├── logs/
│   ├── figures/
│   └── predictions/
└── datasets/
    ├── dataset.h5
    ├── dataset.preprocessed.h5
    └──dataset.preprocessed.standardizer.pt
``` 

## 项目目录介绍

### data

- h5_dataset.py

数据处理与维度修正：将X的第二通道高斯模糊。修正h5数据集的维度，适配python。对p,y数据进行标准化，同时可选保存p,y原始数据。

- preprocess.py

构建标准化器：标准化与反标准化的具体操作（平均值，标准差等）

- build_preprocessed_h5

生成预处理后的h5文件：利用h5_dataset.py中的函数进行数据处理，生成高斯模糊，维度修正后的h5文件用于后续训练。

- split_loader.py

根据数据集样本数和随机种子自动生成训练集、验证集、测试集索引。

- datamodule.py

得到dataloader：基于训练集数据得到standardizer。训练集返回标准化处理后的数据。验证集和测试集还会额外返回原始数据。


### models

- blocks.py
  
 定义网络的基本模块：convbnact,resblock
- backbone.py
  
 定义特征提取主要流程。
- head.py
  
 定义两个处理头：S11head和patternhead
- loss.py
  
 定义损失函数：损失函数为MSE，为ly,lp的权重相加。
- forward_proxy_net.py
  
 定义整体网络结构

### engine

- metrics.py
  
 计算mae指标。主要用于验证集和测试集的mae计算。
- evaluator.py
  
 评估：得到y,p的预测结果，返回loss，yloss,ploss和原始数据的mae损失。
- trainer.py
  
 包含三个函数。第一个为将训练结果追加写入CSV文件。第二个为one_epoch的训练。第三个为训练模型函数。

### utils

- checkpoint.py

保存和加载pytorch训练检查点

- io.py

文件工具函数

- logger.py
  
创建一个同时输出到日志文件和控制台的logger

- plot_loss.py
  
画总损失和子损失图像
- seed.py

固定随机种子，尽量让结果可复现。

- visualize.py
  
画S11图像和pattern图像。

### import.py

检查h5文件的基本结构。

### configs

配置参数

### train.py

项目入口。用于模型的训练。最后会保存最后的模型和最优模型（val），并画损失函数曲线。optimizer为adam，引入scheduler。

### test.py

用于检验最好模型在测试集上的表现，会输出指定数目样本的预测结果曲线与真实曲线。
