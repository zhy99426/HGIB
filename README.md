# Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation
This is the official code for [**HGIB** (Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation)].

> 📝 RecSys 2025

## 🔬 Overview

In this project, we propose a novel model-agnostic Hierarchical Graph Information Bottleneck (HGIB) framework for multi-behavior recommendation to effectively address these challenges. Following information bottleneck principles, our framework optimizes the learning of compact yet sufficient representations that preserve essential information for target behavior prediction while eliminating task-irrelevant redundancies. To further mitigate interaction noise, we introduce a Graph Refinement Encoder (GRE) that dynamically prunes redundant edges through learnable edge dropout mechanisms. We conduct comprehensive experiments on three real-world public datasets, which demonstrate the superior effectiveness of our framework.

## 🌟 Environment Setup

### Prerequisites

The main prerequisites are listed below:
```
Python 3.9
torch
tensorboard
```

And the entire dependencies can be set up by run:

```
pip install -r requirements.txt
```


### Datasets
We provide the Taobao, Tmall and Jdata datasets in './data' folder.

| Dataset | Users  | Items  | Views       | Collects        | Carts         | Buys   |
|---------|--------:|--------:|-------------:|-----------------:|---------------:|--------:|
| Taobao  | 15,449 | 11,953 | 873,954 | -         | 195,476  | 92,180 |
| Tmall   | 41,738 | 11,953 | 1,813,498 | 221,514 | 1,996    | 255,586|
| Jdata   | 93,334 | 24,624 | 1,681,430| 45,613   | 49,891   | 321,883|

Tmall and Jdata datasets are gathered from [CRGCN](https://github.com/MingshiYan/CRGCN) and Taobao dataset is gathered from [MBCGCN](https://github.com/SS-00-SS/MBCGCN).

You can run the following script for preprocessing:
```
python ./data/preprocess.py
```

## 🚀 Getting Started

#### Train HGIB on the `Taobao` dataset
```python
python ./src/main.py --dataset taobao --lr 5e-4 
```

#### Train HGIB on the `Tmall` dataset
```python
python ./src/main.py --dataset tmall  --lr 5e-4 
```

#### Train HGIB on the `Jdata` dataset
```python
python ./src/main.py --dataset jdata  --lr 5e-4 --alpha 0.5
```

## ❤️ Acknowledgement

Our code is developed based on [`MuLe`](https://github.com/geonwooko/MULE).
