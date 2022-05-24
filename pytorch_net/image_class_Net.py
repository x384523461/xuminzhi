import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
# import image_class
class load_data:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def return_trainloader(self):
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True,num_workers=2)  # 将数据集分成多类，用于多线程训练
        return self.trainloader
    def return_testloader(self):
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)
        return self.testloader
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))

        x=self.fc3(x)
        return x
if __name__ == '__main__':

    net=Net()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    ld=load_data()
    trainloader=ld.return_trainloader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)


    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            # labels=F.one_hot(labels,num_classes=10)
            # labels=torch.tensor(labels,dtype=torch.float32)
            # loss=torch.zeros((4,1),dtype=float)
            # for i in range(4):
            loss=criterion(outputs,labels)

            loss.backward()

            optimizer.step()
            running_loss+=loss.item()
            if(i%2000==1999):
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    testloader = ld.return_testloader()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # net.to(device)
            images,labels=images.to(device),labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


