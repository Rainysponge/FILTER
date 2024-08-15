import torch
import torch.nn as nn
import torch.nn.functional as F

class BankModelBottom(nn.Module):
    def __init__(self, input_size=10):
        super(BankModelBottom, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(30, 60)  # Second hidden layer: 30 neurons
        self.fc3 = nn.Linear(60, 30)  # Third hidden layer: 10 neurons
        self.fc4 = nn.Linear(30, 10)  # Third hidden layer: 10 neurons
        # self.fc4 = nn.Linear(10, 1)   # Output layer: 2 classes

    def forward(self, x):
        # x = self.fc1(x)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = self.fc4(x)
        # x = torch.sigmoid(x)       
        return x



class BankModelFuse(nn.Module):
    def __init__(self, inputs_size=20):
        super(BankModelFuse, self).__init__()
        # self.fc1 = nn.Linear(12, 60)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(inputs_size, 60)  # Second hidden layer: 30 neurons
        self.bn1 = nn.BatchNorm1d(60)
        self.fc3 = nn.Linear(60, 10)  # Third hidden layer: 10 neurons
        self.fc4 = nn.Linear(10, 2)   # Output layer: 2 classes

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = torch.relu(self.bn1(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(x)       
        return x

class CreditModelFuse(nn.Module):
    def __init__(self, inputs_size=60):
        super(CreditModelFuse, self).__init__()
        # self.fc1 = nn.Linear(12, 60)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(inputs_size, 60)  # Second hidden layer: 30 neurons
        self.bn1 = nn.BatchNorm1d(60)
        self.fc3 = nn.Linear(60, 2)  # Third hidden layer: 10 neurons
        self.bn2 = nn.BatchNorm1d(10)

        self.fc4 = nn.Linear(10, 2)   # Output layer: 2 classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = self.fc2(x)
        # x = self.fc3(x)
        
        # x = self.fc4(x)
        # x = torch.sigmoid(x)       
        return x

# class CreditModelBottom(nn.Module):
#     def __init__(self, input_size=10):
#         super(CreditModelBottom, self).__init__()
#         self.fc1 = nn.Linear(input_size, 30)  # Input layer: 20 features, first hidden layer: 60 neurons
#         self.bn1 = nn.BatchNorm1d(30)

#         self.fc2 = nn.Linear(30, 60)  # Second hidden layer: 30 neurons
#         self.fc3 = nn.Linear(60, 10)  # Third hidden layer: 10 neurons
#         # self.fc4 = nn.Linear(60, 60)  # Third hidden layer: 10 neurons
#         # self.fc4 = nn.Linear(10, 1)   # Output layer: 2 classes

#     def forward(self, x):
#         # x = self.fc1(x)
#         x = torch.relu(self.bn1(self.fc1(x)))
#         # x = torch.relu(self.fc2(x))
#         # x = torch.relu(self.fc3(x))
#         # x = torch.sigmoid(self.fc4(x))
#         # x = self.fc1(x)
#         # x = self.fc2(x)
#         # x = self.fc3(x)
#         # x = self.fc4(x)
#         # x = torch.sigmoid(self.fc2(x))
#         # x = torch.sigmoid(x)       
#         return x


class CreditModelBottom(nn.Module):
    def __init__(self, input_size=12):
        super(CreditModelBottom, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(60, 30)  # Second hidden layer: 30 neurons
        # self.fc3 = nn.Linear(30, 10)  # Third hidden layer: 10 neurons
        # self.fc4 = nn.Linear(10, 1)   # Output layer: 2 classes

    def forward(self, x):
        x = self.fc1(x)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = self.fc4(x)
        # x = torch.sigmoid(x)       
        return x



class CreditModel(nn.Module):
    def __init__(self, input_size=10):
        super(CreditModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(30, 60)  # Second hidden layer: 30 neurons
        # self.fc3 = nn.Linear(60, 30)  # Third hidden layer: 10 neurons
        self.fc4 = nn.Linear(60, 2)  # Third hidden layer: 10 neurons
        # self.fc5 = nn.Linear(10, 2)   # Output layer: 2 classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc4(x)
        # x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        # x = self.fc5(x)
        x = torch.sigmoid(x)       
        return x