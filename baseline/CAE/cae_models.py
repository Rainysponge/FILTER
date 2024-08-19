import torch
import torch.nn as nn



class EnDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super(EnDecoder, self).__init__()
        self.fc1 = nn.Linear(num_classes, (6*num_classes + 2)**2)
        self.fc2 = nn.Linear((6*num_classes + 2)**2, num_classes)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x



