# 反向传播

反向传播：第l层的第一个神经元的误差项（敏感性）是所有与该神经元相连的第l+1层神经元的误差项的权重和，再乘上该神经元激活函数的梯度。

计算出第l+1层的误差后，就可以用来更新第l层的神经元的参数。先从第一层开始不断计算净输入和激活值直到最后一层，然后再从最后一层不断计算误差修改参数。前馈神经网络就是在这样调整整个网络的参数。

算法过程：

```mermaid
graph TB
start((开始))-->input[输入各项网络参数]
input--->random[随机初始化w,b]
random--->check{FNN错误率不下降}
check--还在下降-->resort[对训练样本重排]
resort--->for{存在样本}
for--存在-->cal1[取出样本]
for--不存在-->check
check--不下降-->output[输出w,b]
output--->finish((结束))
cal1--->cal2[计算净输入和激活值直到最后一层]
cal2--->cal3[计算每一层误差]
cal3--->cal4[计算每一层导数]
cal4--->cal5[根据导数和误差更新参数]
cal5--->for
```

在以上流程中，每一层的误差为$ \delta^{(l)} $，每一层的导数为：
$$
\begin{aligned}
\forall l, &\quad \frac{\partial \mathcal{L}(y^{(n)},\hat{y}^{(n)}) }{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^{T} \\
\forall l, &\quad \frac{\partial \mathcal{L}(y^{(n)},\hat{y}^{(n)})}{\partial b^{(l)}} = \delta ^{(l)}
\end{aligned}
$$

因而，当前层参数可以通过以下方式更新：

$$
\begin{aligned}
W^{(l)} & \quad \leftarrow \quad  W^{(l)} - \alpha (\delta^{(l)}(a^{(l-1)})^{T}+\lambda W^{(l)}) \\
b^{(l)} & \quad \leftarrow \quad b^{(l)} - \alpha \delta^{(l)}
\end{aligned}
$$