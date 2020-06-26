import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
# Importing Modules 
import data as d 
import show_images as s 
import model as m 
import train_test as t 

classes, trainloader, testloader = d.load()
s.show_random_images(trainloader, classes)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = m.ResNet18().to(device)
summary(model, input_size=(3, 32, 32))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=0.0005)
testLoss = []
testAcc = []
EPOCHS = 1
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    print("Device:", device)
    t.train(model, device, trainloader, optimizer, criterion, epoch)
    test_loss , test_acc = t.test(model, device, criterion, testloader)
    testLoss.append(test_loss)
    testAcc.append(test_acc)
def classwise_accuracy(model, device, classes, test_loader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
            	label = labels[i]
            	class_correct[label] += c[i].item()
            	class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
classwise_accuracy(model, device, classes, testloader)
