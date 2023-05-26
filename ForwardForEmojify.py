import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder

train_dir = './train'  # 当前目录下的训练集目录
test_dir = './test'

TrainTransforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomRotation((-10, 10)),
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

TestTransforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 参数为1表示转换为灰度图像
                                     transforms.ToTensor(),
                                     transforms.Normalize(0.5, 0.5)])  # 均值和标准差要根据所有的图像求出来，这里先随便取一个值

TrainData = ImageFolder(train_dir, transform=TrainTransforms)  # ImageFolder把图片按照文件夹分类，可以自动打标签
TestData = ImageFolder(test_dir, transform=TestTransforms)

TrainLoad = DataLoader(TrainData, batch_size=64, shuffle=True, pin_memory=True)  # pin_memory：如果设置为True，
                                                                                 # 那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
TestLoad = DataLoader(TestData, batch_size=64, shuffle=True, pin_memory=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 所有继承自torch.nn.Module的类都需要写这个初始化方法
        self.Linear1 = torch.nn.Linear(48 * 48, 512)  # 注意这个L是大写
        # self.Linear2 = torch.nn.Linear(512, 256)
        # self.Linear3 = torch.nn.Linear(256, 128)
        self.Linear4 = torch.nn.Linear(512, 64)
        self.Linear5 = torch.nn.Linear(64, 7)
        self.Activate = torch.nn.ReLU()  # ReLu在神经网络中相当于一个通用的激活函数
        # self.softmax = torch.nn.Softmax(dim=1)  # dim=0表示按列计算，dim=1表示按行计算

    def forward(self, x):  # 前馈神经网络中方法名必须为forward，不能更改
        x = x.view(x.size(0), -1)  # 把输入数据转化为64行，自动获取列数，因为x.size的第0维是64
        x = self.Activate(self.Linear1(x))
        # x = self.Activate(self.Linear2(x))
        # x = self.Activate(self.Linear3(x))
        x = self.Activate(self.Linear4(x))
        x = self.Linear5(x)
        return x  # 用交叉熵损失函数就不需要softmax了，因为交叉熵会自动做softmax处理


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = torch.nn.CrossEntropyLoss()  # 多分类用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters())  # 也可以用别的优化器对比效果


def Train():
    run_loss = 0
    total = 0
    for image, label in TrainLoad:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # image = image.view(image.shape[0], -1)  # 注意这里和模型里面转换的区别
        pred = model(image)
        loss = criterion(pred, label)
        run_loss += loss.item()  # loss是一个张量。取值要用item
        total += label.size(0)
        loss.backward()
        optimizer.step()
    print(run_loss/total)


def Test():
    total = 0
    correct = 0
    with torch.no_grad():
        for image, label in TestLoad:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            _, predict = torch.max(pred.data, dim=1)  # torch.max返回张量中的最大值和索引，dim=1按照行计算，不需要使用的值用下划线储存
            total += label.size(0)  # size返回label的行数和列数，0表示行数
            number = (predict == label).sum().item()  # sum对64维的布尔型张量求值得到一个只有一个值的张量，item把这个张量转换为int
    print("accuracy on test is %d %%" % (100*correct/total))


if __name__ == '__main__':
    for i in range(10):
        Train()
        Test()


# # 测试代码
# img = Image.open('./test/angry/PrivateTest_88305.jpg')
# # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# example = transforms.ToTensor()(img)  # 把img转换成tensor
# # img.show()
# print(img)
# print(example.shape)
# print(example)
