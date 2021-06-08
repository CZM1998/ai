# FNN实践

使用pytorch搭建一个FNN。
以下代码修改自[Towards Data Science](https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c)

以下案例展示了一个识别手写0~9数字的FNN。


```python

# pytorch 库
import torch
# pytorch 神经网络
import torch.nn as nn
# torchvision 图像处理库
# torchvision 常用视觉数据集
import torchvision.datasets as dsets
# torchvision 常用图像操作
import torchvision.transforms as transforms
# 神经网络中用于计算的输入输出
from torch.autograd import Variable

# 图像大小 28 * 28 = 784
input_size = 784
# 隐藏层神经元个数
hidden_size = 500
# 输出结果，分类结果 0 ~ 9
num_classes = 10
# 整个训练集训练的次数
num_epochs = 5
# 按该大小进行一次迭代，用于批训练
batch_size = 100
# 学习速率
learning_rate = 0.001

# 训练集
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 训练数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 测试数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        # 全连接层：784个输入 对应 500个结点
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数
        self.relu = nn.ReLU()
        # 全连接层：500个结点 对应 10个输出
        self.fc2 = nn.Linear(hidden_size, num_classes)

    # 前向传播
    def forward(self, x):
        # 获取输入
        x = self.fc1(x)
        # 激活
        x = self.relu(x)
        # 输出
        x = self.fc2(x)
        return x


# 实例化网络
net = Net(input_size, hidden_size, num_classes)
# 使用GPU
net = net.cuda()
# 使用GPU或者CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = net.to(device)
# 用于比较输出和正确结果的差距，损失函数
criterion = nn.CrossEntropyLoss().cuda()
# 用于更新网络中的参数
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):

    # Load a batch of images with its (index, data, class)
    # 分批加载图像数据
    for i, (images, labels) in enumerate(train_loader):
        # 把tensor转为Variable，方便计算
        images = Variable(images.view(-1, 28*28)).cuda()
        labels = Variable(labels).cuda()
        # 初始化隐藏层权值为0
        optimizer.zero_grad()
        # 前向传播，计算图像
        outputs = net(images)
        # 计算和正确值的差距
        loss = criterion(outputs, labels)
        # 反向传播计算权值
        loss.backward()
        # 更新隐藏层参数
        optimizer.step()

        # 记录
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    # 选择最佳的分类结果
    _, predicted = torch.max(outputs.data, 1)
    # 记录总数
    total += labels.size(0)
    # 记录正确数
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))

# 保存训练模型
torch.save(net.state_dict(), 'fnn_model.pkl')
```