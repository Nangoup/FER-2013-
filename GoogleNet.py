import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F

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


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Con1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.Con2 = torch.nn.Conv2d(88, 20, kernel_size=5, stride=1)
        self.Con3 = torch.nn.Conv2d(88, 10, kernel_size=5, stride=1)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.Pooling = torch.nn.MaxPool2d(2)
        self.Activate = torch.nn.ReLU()
        self.fc = torch.nn.Linear(250, 7)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.Activate(self.Pooling(self.Con1(x)))
        x = self.incep1(x)
        x = self.Activate(self.Pooling(self.Con2(x)))
        x = self.incep2(x)
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
