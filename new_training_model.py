import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# 定义一维卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一维卷积层，它有853个输入通道、32个输出通道（=有32个卷积核），卷积核大小为8
        self.conv1 = nn.Conv1d(in_channels=853, out_channels=32, kernel_size=8)
        # 最大池化层,池化窗口大小为2
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层,输入特征数量为32 * 46（这个值可能需要根据实际输入数据调整），输出特征数量为10。
        self.fc1 = nn.Linear(16, 10)  # 更新展平操作后的大小
        # Dropout层,丢弃概率为0.5，用于正则化以减少过拟合
        self.dropout = nn.Dropout(p=0.5)
        # 输出层
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.permute(1,0)
        # 对输入x应用ReLU激活函数和卷积层conv1，然后应用最大池化层pool。
        x = nn.functional.relu(self.conv1(x))
        x = x.permute(1, 0)#矩阵转秩
        x = self.pool(x)
        print('-------pool feature size:{}'.format(x.shape))
        # 将卷积层的输出展平成一维向量，以匹配全连接层fc1的输入要求

        # 对展平后的特征应用通过全连接层fc1。再应用ReLU激活函数
        x = nn.functional.relu(self.fc1(x))
        # 进入Dropout层，减少过拟合
        x = self.dropout(x)
        # 通过输出层fc2，并使用Sigmoid激活函数（通常用于二分类问题），返回预测结果。
        x = torch.sigmoid(self.fc2(x))
        return x

# 创建模型实例
model = Net()
# 打印模型结构
print(model)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_data = pd.read_csv("splited_data/train_data.csv")
# 分离标签和特征
labels = train_data.pop("label")  # 使用pop方法删除并返回'label'列
numpy_features = train_data.values  # 获取剩余的特征列
# 将NumPy数组转换为PyTorch张量
tensor_features = torch.from_numpy(numpy_features).float()  # 假设特征是浮点型
tensor_labels = torch.tensor(labels.values, dtype=torch.int)  # 假设标签是整型（如果是分类问题）

# 创建TensorDataset
train_dataset = TensorDataset(tensor_features, tensor_labels)
print(tensor_features.shape,tensor_labels.shape)
# 定义DataLoader
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 模型训练
num_epochs = 50  # 设置训练的轮数（epoch）为50。
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        print('----input shape:{}---'.format(data.shape))
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target.float())  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        if (epoch + 1) % 10 == 0:  # 每10轮输出一次训练状态。
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # 打印当前轮数和损失值。


# 测试模型
model.eval()  # 将模型设置为评估模式，这会关闭Dropout和Batch Normalization等层。
val_data = pd.read_csv("splited_data/val_data.csv")
# 分离标签和特征
val_labels = val_data.pop("label")  # 使用pop方法删除并返回'label'列
val_input = val_data.values  # 获取剩余的特征列
print(val_input)  # 打印测试输入张量。
val_input = torch.from_numpy(val_input).float()#转为Pytorch张量
with torch.no_grad():  # 禁用梯度计算，这在推理时可以减少内存消耗。
    output = model(val_input)  # 将测试输入通过模型，获取输出。
    for item in output:
        print(item)


