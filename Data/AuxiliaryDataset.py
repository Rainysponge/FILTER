import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def gen_poison_data(poison_method, inputs, targets, noise=0, p=1.0, clean_label=False):
    tmp_inputs = inputs.clone()
    tmp_targets = targets.clone()
    length = round(p * len(tmp_inputs))
    tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
    tmp_inputs[:, :, :, :] += noise * torch.rand(tmp_inputs.shape)

    # poison data
    if not clean_label:
        if poison_method == "mid_random_noise":
            tmp_inputs[:length, 12:20, 12:20, :] = torch.from_numpy(
                np.random.rand(length, 8, 8, tmp_inputs.shape[-1])
            )
            tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))
        elif poison_method == "trigger":
            trigger = np.zeros([length, 3, 3, tmp_inputs.shape[-1]])
            trigger[:length, 0, 0, 0] = 1
            trigger[:length, 0, 2, 0] = 1
            trigger[:length, 1, 1:3, 0] = 1
            trigger[:length, 2, 0:2, 0] = 1

            tmp_inputs[:length, -5:-2, -5:-2, :] = torch.from_numpy(trigger)
            tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))

        elif poison_method == "dismissing":
            for i in range(len(tmp_targets)):
                if tmp_targets[i] == torch.Tensor(np.array([0])):
                    tmp_targets[i] = torch.Tensor(np.array([9]))

        tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
        return tmp_inputs, tmp_targets.long()
    else:

        target_class = 0
        trigger = torch.zeros((tmp_targets.shape[0], 3, 3, tmp_inputs.shape[-1]))
        trigger[:, 0, 0, 0] = 1
        trigger[:, 0, 2, 0] = 1
        trigger[:, 1, 1:3, 0] = 1
        trigger[:, 2, 0:2, 0] = 1

        # Add the trigger to tmp_inputs only for samples with the specified target class
        tmp_inputs[tmp_targets == target_class, 1:4, 1:4, :] = trigger[
            tmp_targets == target_class
        ]
        tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
        return tmp_inputs, tmp_targets.long()


class DetectorDataset(Dataset):
    # 2 client
    def __init__(self, smdata_a, smdata_b, labels):
        self.smdata_a = smdata_a
        self.smdata_b = smdata_b
        self.labels = labels

    def __len__(self):
        return len(self.smdata_a)

    def __getitem__(self, index):

        index = index % len(self.smdata_a)  # 取余数来处理超出索引范围的情况
        return (self.smdata_a[index], self.smdata_b[index]), self.labels[index]


class DetectorUnionDataset(Dataset):
    # 2 client
    def __init__(self, smdata, labels):
        self.smdata = smdata

        self.labels = labels

    def __len__(self):
        return len(self.smdata)

    def __getitem__(self, index):

        index = index % len(self.smdata)  # 取余数来处理超出索引范围的情况
        return self.smdata[index], self.labels[index]


def gen_poison_data_replace_attack(inputs):
    tmp_inputs = inputs.clone()
    # tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
    height, weight = tmp_inputs.shape[-2], tmp_inputs.shape[-1]
    # print(height, weight)
    trigger = np.zeros([tmp_inputs.shape[0], tmp_inputs.shape[1], 3, 3])
    # print(tmp_inputs)
    trigger[:, 0, 2, 2] = 1
    trigger[:, 0, 0, 2] = 1
    trigger[:, 1, 2, 0] = 1
    trigger[:, 1, 1, 1] = 1
    # poison data
    tmp_inputs[:, :, height - 3 : height, weight - 3 : weight]
    # trigger = np.zeros([length, 3, 3, tmp_inputs.shape[-1]])
    # trigger[:length, 0, 0, 0] = 1
    # trigger[:length, 0, 2, 0] = 1
    # trigger[:length, 1, 1:3, 0] = 1
    # trigger[:length, 2, 0:2, 0] = 1

    tmp_inputs[:, :, height - 3 : height, weight - 3 : weight] = torch.from_numpy(
        trigger
    )

    # tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
    return tmp_inputs


def gen_poison_dataset_replace_attack(inputs):
    # tmp_inputs = inputs.clone()
    # tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
    print(inputs.shape)
    print(type(inputs))
    inputs[:, 2, 0, :] = np.array([0, 255, 0])
    inputs[:, 1, 1, :] = np.array([0, 255, 0])
    inputs[:, 0, 0, :] = np.array([255, 0, 255])
    inputs[:, 0, 2, :] = np.array([255, 0, 255])

    return inputs


def gen_poison_dataset_replace_attack_cinic(inputs):
    # tmp_inputs = inputs.clone()
    # tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
    # print(inputs.shape)
    # print(type(inputs))
    inputs[:, :, 2, 0] = torch.tensor([0, 1, 0])
    inputs[:, :, 1, 1] = torch.tensor([0, 1, 0])
    inputs[:, :, 0, 0] = torch.tensor([1, 0, 1])
    inputs[:, :, 0, 2] = torch.tensor([1, 0, 1])

    return inputs
