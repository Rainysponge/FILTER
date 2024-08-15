import torch


import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, CIFAR100

from AuxiliaryDataset import gen_poison_data
transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)
  

if __name__ == "__main__":
    train_dataset = CIFAR10("../dataRepo/cifar10", train=True, transform=transform_train)
    loader = DataLoader(dataset=train_dataset, batch_size=1)
    for epoch in range(3):
        for ids, (inputs, targets) in enumerate(loader):

            print(inputs.shape)
            inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
            print(inp1.shape)
            inp1, targets = gen_poison_data("trigger", inp1, targets)
            torchvision.utils.save_image(inputs.squeeze(), f'/home/users/HuZhanyi/VFL_Defense/VFL_defense/Data/figs/transtest{epoch}.png')
            # torchvision.utils.save_image(inp2.squeeze(), f'/home/users/HuZhanyi/VFL_Defense/VFL_defense/Data/figs/inp2_{ids}.png')
            # torchvision.utils.save_image(inputs.squeeze(), f'/home/users/HuZhanyi/VFL_Defense/VFL_defense/Data/figs/inputs_{ids}.png')
            break
    

