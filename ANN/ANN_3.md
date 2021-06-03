# 激活函数

激活函数的作用：增加非线性因素，解决线性模型表达能力不足的缺陷。

如果没有激活函数，那么神经网络将会变成一个线性回归的模型，而我们需要的是一个非线性的模型。

常用的激活函数有：
1. Sigmoid
2. ReLU
3. Swish
4. GELU
5. Maxout

## Sigmoid函数

Sigmoid函数主要有Logistics和Tanh函数。

Logistics定义：

$$ \sigma(x) = \frac{1}{1+e^{-x}} $$

Tanh定义：

$$ tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} $$
$$ tanh(x)=2\sigma(2x)-1 $$

Logistics函数值域(0,1)，tanh函数值域(-1,1)。

Logistics函数使得输出具有两点性质：
1. 输出可以视作概率分布
2. 输出看作软性门，控制其他输出的数量

![函数图](/img/LogisticTanh.png)

二者的近似模拟，加快计算速度：

Logistic用hard-logistic(x)近似：

$$
\begin{aligned}
\text{hard-logistic}(x) &= \begin{cases}
        1 & g_{l}(x) \ge 1 \\
        g_{l} & 0<g_{l}(x)<1 \\
        0 & g_{l}(x) \le 0
    \end{cases} \\
    &= \max(\min(g_{l}(x),1),0) \\
    &= \max(\min(0.25x+0.5,1),0)
\end{aligned}
$$

Tanh用hard-tanh(x)近似：

$$
\begin{aligned}
\text{hard-tanh(x)}&=\max(\min(g_{t}(x),1),-1) \\
&=\max(\min(x,1),-1)
\end{aligned}
$$

![函数图](/img/HardLT.png)

## ReLu

ReLU（Rectified Linear Unit，修正线性单元）是目前深度神经网络中经常使用的激活函数。

定义：

$$
\begin{aligned}
ReLu(x)&=\begin{cases}
            x & x\ge 0 \\
            0 & x<0
        \end{cases} \\
        &=\max(0,x)
\end{aligned}
$$

存在死亡ReLu问题。

### 带泄露ReLu

$$
\begin{aligned}
    LeakyReLu(x)&= \begin{cases}
        x & \text{if}\ x>0 \\
        \gamma x & \text{if}\ x\le x
    \end{cases} \\
    &=\max(0,x)+\gamma \min(0,x)
\end{aligned}
$$

一般$\gamma$取很小的常数，若$\gamma < 1$，可以表示为：
$$ LeakyReLU(x)=\max(x, \gamma x) $$

这样当神经元非激活时也能有一个非零的梯度可以更新参数， 避免永远不能被激活

### 带参数的ReLU

$$
\begin{aligned}
    LeakyReLu(x)&= \begin{cases}
        x & \text{if}\ x>0 \\
        \gamma_{i} x & \text{if}\ x\le x
    \end{cases} \\
    &=\max(0,x)+\gamma_{i} \min(0,x)
\end{aligned}
$$

$\gamma_{i}$是一个可以学习的参数。

### ELU函数

$$
\begin{aligned}
    ELU(x)&=\begin{cases}
        x & \text{if}\ x>0 \\
        \gamma (e^(x)-1) & \text{if}\ x \le 0
    \end{cases} \\
    & = \max(0,x) +\min(0,\gamma (e^{x}-1))
\end{aligned}
$$

### Softplus函数

$$Softplus(x) = \log(1 + e^{x})$$

![所有RELU](/img/ReLUEtc.png)

## Swish函数

$$ swish(x)=x\sigma(\beta x) $$

![swish](/img/swish.png)

## GELU函数

$$
\begin{aligned}
    GELU(x) & = xP(X ≤ x) \\
    GELU(x) & \approx 0.5x\left(1+tanh\left(\sqrt{\frac{2}{x}}\left(x+0.044715x^{3}\right)\right)\right) \\
    GELU(x) & \approx x\sigma (1.702x)
\end{aligned}
$$

## Maxout单元

分段线性函数，之前的函数的输入都是z，即所有神经元输入的加权和，此函数则是原始输入。

$$
\begin{aligned}
z_{k}&=w_{k}^{T}x+b_{k}\\
maxout(x)&=\max_{k\in [1,K]}(z_{k})
\end{aligned}
$$