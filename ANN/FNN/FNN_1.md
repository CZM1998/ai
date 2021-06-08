# 前馈神经网络

前馈神经网络（Feedforward Neural Network, FNN）又称多层感知器（Multi-Layer Perceptron，MLP）。

第0层称为输入层，最后一层称为输出层，其他中间层称为隐藏层。整个网络中无反馈，信号从输入层向输出层单向传播，可用一个有向无环图表示．

![fnn](/img/fnn.png)

![fnn](/img/fnn_symbol.png)

其中，L只考虑隐藏层和输出层。

令$a^{0}=x$，前馈神经网络通过不断迭代下面公式进行信息传播：
$$
\begin{aligned}
    z^{(l)}&=W^{(l)}a^{(l-1)}+b^{(l)} \\
    a^{(l)}&=f_{l}(z^{(l)})
\end{aligned}
$$

另一种写法：

$$
\begin{aligned}
    z^{(l)}&=W^{(l)}f_{l-1}(z^{(l-1)})+b^{(l)} \\
    a^{(l)}&=f_{l}(W^{(l)}a^{(l-1)}+b^{(l)})
\end{aligned}
$$

FNN执行流程：

```mermaid
graph TD
start((开始))-->net_activation[计算净活性值]
net_activation-->activation[通过激活函数计算活性值]
activation-->have_next{还有一层神经元?}
have_next--存在-->net_activation
have_next--不存在-->finish[输出]
finish-->check[判断结果准确性]
check-->fix[修正之前所有神经元参数]
fix-->start
```