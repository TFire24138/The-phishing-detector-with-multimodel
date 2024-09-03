import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# 定义一维卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一维卷积层，它有853个输入通道、32个输出通道，卷积核大小为7，padding大小为4
        kernel_size = 9
        padding = ((kernel_size - 1) // 2)  # 计算填充量
        self.conv1 = nn.Conv1d(in_channels=686, out_channels=32, kernel_size=kernel_size,padding = padding)
        #self.conv1 = nn.Linear(853, 32)
        # 最大池化层,池化窗口大小为2
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层,输入特征数量为32 * 46（这个值可能需要根据实际输入数据调整），输出特征数量为10。
        self.fc1 = nn.Linear(16, 10)  # 更新展平操作后的大小
        # Dropout层,丢弃概率为0.5，用于正则化以减少过拟合
        self.dropout = nn.Dropout(p=0.01)
        # 输出层
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = x.permute(1,0)
        # 对输入x应用ReLU激活函数和卷积层conv1，然后应用最大池化层pool。
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(1, 0)#矩阵转秩
        x = self.pool(x)
        print('-------pool feature size:{}'.format(x.shape))
        # 将卷积层的输出展平成一维向量，以匹配全连接层fc1的输入要求
        # 对展平后的特征应用通过全连接层fc1。再应用ReLU激活函数
        x = nn.functional.relu(self.fc1(x))
        # 进入Dropout层，减少过拟合
        x = self.dropout(x)
        # 通过输出层fc2，并使用Sigmoid激活函数（通常用于二分类问题），返回预测结果。
        x = self.fc2(x)
        return x


# 创建模型实例
model = Net()
# 打印模型结构
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

#读取数据
train_data = pd.read_csv("splited_data/train_data.csv")
# 分离标签和特征
labels = train_data.pop("label")  # 使用pop方法删除并返回'label'列
numpy_features = train_data.values  # 获取剩余的特征列
# 将NumPy数组转换为PyTorch张量
tensor_features = torch.from_numpy(numpy_features).float()  # 假设特征是浮点型
tensor_labels = torch.tensor(labels.values, dtype=torch.int)  # 假设标签是整型（如果是分类问题）

# 创建TensorDataset
train_dataset = TensorDataset(tensor_features, tensor_labels)
print(tensor_features.shape, tensor_labels.shape)
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
        loss = criterion(output, target.long())  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        if (epoch + 1) % 10 == 0:  # 每10轮输出一次训练状态。
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # 打印当前轮数和损失值。
            torch.save(model.state_dict(), './model_epoch_{}.pth'.format(epoch)) #每训练10轮保存一次模型


# 加载保存的模型权重
model = Net()
model.load_state_dict(torch.load("model_epoch_49.pth"))

#用验证集进行验证
model.eval()  
val_data = pd.read_csv("splited_data/val_data.csv")
val_labels = val_data.pop("label")  # 使用pop方法删除并返回'label'列、
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long) #将标签值转换为张量2
val_input = val_data.values  # 获取剩余的特征列
val_input = torch.from_numpy(val_input).float()  # 转为Pytorch张量
with torch.no_grad():  
    output = model(val_input)  # 将测试输入通过模型，获取输出。
    pred_labels = torch.argmax(output, dim=1)
    correct_predictions = pred_labels == val_labels_tensor #对预测结果和真实结果逐个比较
    correct_predictions_float = correct_predictions.float()#将比较结果转换为float类型（0/1），用于计算准确率
    accuracy = correct_predictions_float.mean().item()
    print('val_accuracy',accuracy)


# 无模态缺失的测试模型
# model.eval()
# test_data = pd.read_csv("splited_data/test_data.csv")
# test_labels = test_data.pop("label")  # 使用pop方法删除并返回'label'列、
# test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
# test_input = test_data.values  # 获取剩余的特征列
# test_input = torch.from_numpy(test_input).float()  # 转为Pytorch张量
# with torch.no_grad():  # 禁用梯度计算，这在推理时可以减少内存消耗。
#     output = model(test_input)  # 将测试输入通过模型，获取输出。
#     pred_labels = torch.argmax(output, dim=1)
#     correct_predictions = pred_labels == test_labels_tensor
#     correct_predictions_float = correct_predictions.float()
#     # Calculate the mean value of correct_predictions to get the accuracy
#     accuracy = correct_predictions_float.mean().item()
#     print('test_accuracy',accuracy)

# # 测试模态缺失情况
# model.eval()
# test_data = pd.read_csv("splited_data/test_data.csv")
# test_labels = test_data.pop("label")
# test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
# test_data.iloc[:,85:] = 0    #将URL特征设为0，作为模态缺失
# test_input = test_data.values
# test_input = torch.from_numpy(test_input).float()
# with torch.no_grad():
#     output = model(test_input)
#     pred_labels = torch.argmax(output, dim=1)
#     correct_predictions = pred_labels == test_labels_tensor
#     correct_predictions_float = correct_predictions.float()
#     # Calculate the mean value of correct_predictions to get the accuracy
#     accuracy = correct_predictions_float.mean().item()
#     print('test_accuracy',accuracy)
