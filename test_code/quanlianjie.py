import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 20, hidden_size: int = 128, num_classes: int = 5):
        super(NeuralNetwork, self).__init__()
        # 输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # 定义激活函数，例如ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    pr_data = torch.load('pred_a')
    label = torch.load('la0')
    label = label.to(device)
    num_epochs = 50
    model = NeuralNetwork().to(device)
    dataset = TensorDataset(pr_data, label)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        accu_num = torch.zeros(1).to(device)
        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()  # 清零梯度
            labels = labels
            # 前向传播
            pred = model(inputs.detach())
            pred_classes = torch.max(pred, dim=1)[1]
            # 计算损失
            loss = criterion(pred, labels.long())
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            sds = accu_num.item() / 2939
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 打印每个epoch的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}', sds)

