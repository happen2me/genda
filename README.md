# Source-Free Compression & Domain Adaptation

## 用法

### 训练教师网络（LeNet5）
```
python train_teacher.py --dataset=MNIST

选项:
dataset: 数据集。可选的有 'MNIST', 'MNIST-M', 'USPS', 'SVHN', 'MNIST3'
data: 数据集存放地址。默认为 'cache/data/'
output_dir: 模型存放地址。默认为 'cache/models'
batch_size: 默认为512
lr: 学习速率。默认为1e-3
```

### 压缩教师网络

```
python train_dafl.py --dataset=MNIST --target=USPS

选项：
dataset: 源领域数据集。可选的有'MNIST','cifar10','cifar100', 'USPS', 'MNIST3', 'MNIST-M'
target: 目标领域数据集。可选项同dataset
data: 数据集存放地址。默认为 'cache/data/'
teacher_dir：预训练教师模型存放位置
img_opt_step：每个样本的优化次数
```

### 领域适配

```python
python train_adda.py --dataset=MNIST --target='USPS'

选项：
dataset: 源领域数据集。可选的有'MNIST','USPS'
target: 目标领域数据集。可选项同dataset
model_root：预训练模型的存放地址
```

### 代码结构说明：

每一个文件的主要逻辑部分在`run()`函数。

## 结果

## 致谢

本项目引用了如下仓库的源代码：

[Data efficient model compression](https://github.com/huawei-noah/Data-Efficient-Model-Compression)

[pytorch adda](https://github.com/corenel/pytorch-adda)