import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Define the paths to the train and test directories
train_dir = './train'
test_dir = './test'

# Define the transforms to be applied to the images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=1),  # 参数为1时，将图像转为灰度图；参数为3时，为RGB三通道图像
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the train and test datasets
train_data = ImageFolder(train_dir, transform=train_transform)
test_data = ImageFolder(test_dir, transform=test_transform)


# Define the batch size for loading the data
batch_size = 64

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Create data loaders for train and test datasets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)


class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(48 * 48, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 7)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input properly
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.output(x)
        output = self.softmax(x)
        return output


model = FullNet().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())
epoch = 10
i = 0

if __name__ == '__main__':
    for _ in range(epoch):
        running_loss = 0
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            image = image.view(image.shape[0], -1)
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f'Training loss: {running_loss / len(train_loader):.4f}')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                x_input, labels = data
                x_input = x_input.to(device)
                labels = labels.to(device)
                y_pred = model(x_input)
                _, predicted = torch.max(y_pred.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy on test set: %d %%' % (100 * correct / total))