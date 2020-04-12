import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.fc1 = nn.Linear(64 * 4 * 4, 180)
        self.fc2 = nn.Linear(180, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

trainloss = []
trainaccuracy = []
testloss = []
testaccuracy = []

for epoch in range(50):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[%d, %5d] trainloss: %.3f' %
          (epoch + 1, i + 1, running_loss / 12500))
    trainloss.append(running_loss / 12500)
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    print('[%d, %5d] testloss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2500))
    testloss.append(running_loss / 2500)
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on 10000 test images: %d %%' % (
        100 * correct / total))
        testaccuracy.append(correct / total)
        correct = 0
        total = 0
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on 10000 train images: %d %%' % (
        100 * correct / total))
        trainaccuracy.append(correct / total)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

with open('trainloss.txt', 'w') as file1:
    file1.write(str(trainloss))
with open('trainaccuracy.txt', 'w') as file2:
    file2.write(str(trainaccuracy))
with open('testloss.txt', 'w') as file3:
    file3.write(str(testloss))
with open('testaccuracy.txt', 'w') as file4:
    file4.write(str(testaccuracy))

x = np.arange(1, 51)
plt.figure(num=1,figsize=(8,7))
plt.axis([0,51,0,3])
loss1, = plt.plot(x, testloss, label='loss on testset')
plt.grid()
plt.legend(handles=[loss1])
plt.show()
plt.figure(num=1,figsize=(8,7))
plt.axis([0,51,0,3])
loss2, = plt.plot(x, trainloss, label='loss on trainset')
plt.grid()
plt.legend(handles=[loss2])
plt.show()
plt.figure(num=1,figsize=(8,7))
plt.axis([0,51,0,1.02])
accuracy1, = plt.plot(x, testaccuracy, label='accuracy on testset')
plt.grid()
plt.legend(handles=[accuracy1])
plt.show()
plt.figure(num=1,figsize=(8,7))
plt.axis([0,51,0,1.02])
accuracy2, = plt.plot(x, trainaccuracy, label='accuracy on trainset')
plt.grid()
plt.legend(handles=[accuracy2])
plt.show()