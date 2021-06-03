# 什么是神经网络

人工神经网络（Artificial Neural Network，ANN）是一种是为模拟人脑神经网络而设计的一种计算模型。ANN在结构上类似生物神经网络。

## 神经网络的基本构成

ANN具有以下三个部分：
1. 结构（Architecture）：结构指定了网络中的变量和它们的拓扑关系
2. 激励函数（Activation Rule）：大部分神经网络模型具有一个短时间尺度的动力学规则，来定义神经元如何根据其他神经元的活动来改变自己的激励值。
3. 学习规则（Learning Rule）：学习规则指定了网络中的权重如何随着时间推进而调整。

一个基本的神经元：

![基本神经元](/img/Ncel.png)

其中：
* a1~an为输入向量的各个分量
* w1~wn为神经元各个突触的权值
* b为偏置
* f为传递函数，通常为非线性函数。
* t为神经元输出

数学表示

$$ t=f(\vec{W^{'}\vec{A}}+b) $$

$\vec{W}$为权向量，$\vec{W^{'}}$为$\vec{W}$的转置
$\vec{A}$为输入向量
$b$为偏置
$f$为传递函数

一个单层神经元网络：
![单层神经元网络](/img/SingleLayerNeuralNetwork.png)

多层神经元网络基本结构：
* 输入层（Input layer），众多神经元（Neuron）接受大量非线形输入消息。
* 输出层（Output layer），消息在神经元链接中传输、分析、权衡，形成输出结果。
* 隐藏层（Hidden layer），简称“隐层”，是输入层和输出层之间众多神经元和链接组成的各个层面。

## ANN分类

1. 依学习策略（Algorithm）分类主要有：
    * 监督式学习网络（Supervised Learning Network）为主
    * 无监督式学习网络（Unsupervised Learning Network）
    * 混合式学习网络（Hybrid Learning Network）
    * 联想式学习网络（Associate Learning Network）
    * 最适化学习网络（Optimization Application Network）
2. 依网络架构（Connectionism）分类主要有：
    * 前馈神经网络（Feed Forward Network）
    * 循环神经网络（Recurrent Network）
    * 强化式架构（Reinforcement Network）

## ANN结构分类

1. 前馈网络
2. 记忆网络
3. 图网络