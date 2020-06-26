import torch 
import torchvision
import torchvision.transforms as transforms


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np


def load():
  transform = transforms.Compose([
              #transforms.Resize((28, 28)),
              #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
              #transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
              transforms.RandomCrop(32, padding=4),
		          transforms.RandomHorizontalFlip(),
              #transforms.RandomVerticalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
          ])
  test_transform = transforms.Compose([
            
              transforms.ToTensor(),
              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
          ])
          # Training set and train loader
  train_set = torchvision.datasets.CIFAR10(root='./data', download = True, train = True, transform= transform)
  trainloader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, num_workers=2)

          # Test set and test loader 
  test_set = torchvision.datasets.CIFAR10(root='./data', download = True, train = False, transform = transform)
  testloader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, num_workers = 2)

  print("Finished loading data")

  classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  print(classes)
  return classes, trainloader, testloader
