# 损失函数

损失函数 （Loss Function） 也可称为代价函数 （Cost Function）或误差函数（Error Function），用于衡量预测值与实际值的偏离程度。损失函数越小，模型的性能就越好。

损失函数分为经验风险损失函数和结构风险损失函数。经验风险损失函数指预测结果和实际结果的差别，结构风险损失函数是指经验风险损失函数加上正则项。

常见的损失函数如下：

1. 0-1损失函数(zero-one loss)

$$
L(Y,f(x))=\begin{cases}
1 & Y\neq f(x)\\
0 & Y=f(x)
\end{cases}
$$

放宽条件：

$$
L(Y,f(x))=\begin{cases}
1 & |Y-f(x)|\geq T\\
0 & |Y=f(x)| < T
\end{cases}
$$

2. 绝对值损失函数
$$ L(Y,f(x))=|Y-f(x)| $$

3. log对数损失函数
$$ L(Y,P(Y|X))=-\log(P(Y|X)) $$

4. 平方损失函数
$$ L(Y|f(x))=\sum_{N}(Y-f(X))^2 $$

5. 指数损失函数（exponential loss）
$$ L(Y|f(X))=e^{-yf(x)} $$

6. Hinge 损失函数

$$ L(y,f(x))=\max(0,1-yf(x)) $$

7. 感知损失(perceptron loss)函数

$$ L(y,f(x))=\max(0, -f(x)) $$

8. 交叉熵损失函数 (Cross-entropy loss function)

$$ C=-\frac{1}{n}\sum_{x}[y\ln(a)+(1-y)\ln(1-a)] $$

此外，还有一个风险函数的概念。损失函数是衡量一次预测的好坏，而风险函数是衡量平均意义下的预测好坏。

风险函数是损失函数的期望值，可表示为：
$$
r(d,\theta)=E[R(d,\theta)]=\sum_{j=1}^{L}R(d,\theta)p(\theta_{j})\\
R(d,\theta)=E[L(d,\theta)]
$$

其中L为损失函数，E为期望，p为概率。