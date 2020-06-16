# Source-Free Compression & Domain Adaptation
本文提出了对模型进行压缩，并且适配目标领域的方法：首先，原模型的冗余信息和目标领域没有标注的样本被利用于构造出包含源领域数据分布信息的伪源领域样本；其次，使用伪源领域的样本作为输入，源领域模型的识别能力通过知识蒸馏的方式被传递给一个更小的深度网络；最后，通过对抗性的方法使小模型输出与从伪源领域样本提取出的特征分布近似的中间特征，从而消除目标领域与源领域间的分布差异，继而使得直接使用源领域的分类器进行分类成为可能。

本文的对提出的方法在MNIST和USPS之间的领域自适应任务上进行了验证，提出的方法在计算效率优于源领域模型的情况下，在目标领域的识别准确率也高于直接应用原始模型。

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
python train_dafl.py --dataset MNIST --target USPS

选项：
dataset: 源领域数据集。可选的有'MNIST','cifar10','cifar100', 'USPS', 'MNIST3', 'MNIST-M'
target: 目标领域数据集。可选项同dataset
data: 数据集存放地址。默认为 'cache/data/'
teacher_dir：预训练教师模型存放位置
img_opt_step：每个样本的优化次数
```

### 领域适配

```
python train_adda.py --dataset MNIST --target USPS

选项：
dataset: 源领域数据集。可选的有'MNIST','USPS'
target: 目标领域数据集。可选项同dataset
model_root：预训练模型的存放地址
```

### 代码结构说明：

每一个文件的主要逻辑部分在`run()`函数。

## 结果



|            | MNIST→USPS | USPS→MNIST | 备注               |
| ---------| ---------- | ---------- | ------------------ |
| 源领域模型 | 0.79       | 0.529      |                    |
| 压缩后的模型 | 0.74       | 0.623      |                    |
| 目标领域模型 | 0.835      | 0.655      |                    |
| ADDA      | 0.894      | 0.901      | 需要源数据，未压缩 |

## 致谢

本项目引用了如下仓库的源代码：

[Data efficient model compression](https://github.com/huawei-noah/Data-Efficient-Model-Compression)

[pytorch adda](https://github.com/corenel/pytorch-adda)