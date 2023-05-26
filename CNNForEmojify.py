import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_dir = './train'
test_dir = './test'

TrainTransforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomRotation(10),  # 旋转角度为（-10，10）
                                      transforms.Grayscale(num_output_channels=1),  # 转化为灰度图像
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

TestTransforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(0.5, 0.5)])

TrainData = ImageFolder(train_dir, transform=TrainTransforms)
TestData = ImageFolder(test_dir, transform=TestTransforms)

TrainLoader = DataLoader(TrainData, batch_size=64, shuffle=True, pin_memory=True)
TestLoader = DataLoader(TestData, batch_size=64, shuffle=True, pin_memory=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Con1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1)  # Conv2d是指定大小的二维卷积核，Conv1d是一维
        self.Con2 = torch.nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.Con3 = torch.nn.Conv2d(20, 10, kernel_size=5, stride=1)
        self.Pooling = torch.nn.MaxPool2d(2)  # Maxpool2d(二维池化)，Maxpool1d(一维池化)
        self.Activate = torch.nn.ReLU()
        self.fc = torch.nn.Linear(250, 7)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.Activate(self.Pooling(self.Con1(x)))
        x = self.Activate(self.Pooling(self.Con2(x)))
        x = self.Activate(self.Con3(x))  # 注意这里的维度变成了64*250
        x = x.view(batch_size, -1)  # 注意全连接层的维度
        return self.fc(x)


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)  # 把模型迁移到GPU
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def Train():
    run_loss = 0
    count_labels = 0
    for images, labels in TrainLoader:
        images = images.to(device)
        labels = labels.to(device)
        predict = model(images)
        loss = criterion(predict, labels)
        count_labels += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    print(run_loss/count_labels)


def Test():
    total = 0
    correct = 0
    with torch.no_grad():  # 不要漏掉这里的括号
        for images, labels in TestLoader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predict = torch.max(pred.data, dim=1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)  # size(0)是什么意思
    print('accuracy on test is %d %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(20):
        Train()
        Test()
