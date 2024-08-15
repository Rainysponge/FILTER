from typing import Callable, Optional
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from PIL import Image
import numpy as np


class IndexedCIFAR10(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class IndexedCIFAR100(CIFAR100):
    def _filter_data(self, data, targets):
        # 过滤数据和目标
        filtered_data = []
        filtered_targets = []
        for i, target in enumerate(targets):
            if target < 30:
                filtered_data.append(data[i])
                filtered_targets.append(target)
        return filtered_data, filtered_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class split_dataset(Dataset):
    def __init__(self, data):
        super(split_dataset, self).__init__()
        self.Xa_data = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.Xa_data[index]
        Y = self.labels[index]
        return X, Y


class cluster_dataset(CIFAR10):
    def __init__(self, *args, new_targets=None, **kwargs):
        super().__init__(*args, **kwargs)

        # 如果提供了新的标签，那么就替换原始标签
        if new_targets is not None:
            assert len(new_targets) == len(self.targets)
            self.targets = new_targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


from torch.utils.data import TensorDataset


class IndexTensorDataset(TensorDataset):
    def __getitem__(self, index):
        return tuple(list(tensor[index] for tensor in self.tensors) + [index])


class IndexedCIFAR10Split(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split_stragey: list = [],
    ) -> None:
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            download,
        )
        self.split_stragey = split_stragey
        self.toTensorTrans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.to_pil_image = transforms.ToPILImage()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # Split
        # image_list = [] =
        data_tensor = self.toTensorTrans(img)
        image_list = torch.split(data_tensor, self.split_stragey, dim=2)
        image_list = [self.to_pil_image(t) for t in image_list]
        if self.transform is not None:

            # img = self.transform(img)
            img = [self.transform(image) for image in image_list]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class IndexedMNISTSplit(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split_stragey: list = [14, 14],
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.split_stragey = split_stragey
        self.toTensorTrans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.to_pil_image = transforms.ToPILImage()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_list = torch.split(img, self.split_stragey, dim=1)
        img_list = [Image.fromarray(img.numpy(), mode="L") for img in img_list]

        if self.transform is not None:
            img_list = [self.transform(img) for img in img_list]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_list, target, index


class IndexedMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split_stragey: list = [14, 14],
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.split_stragey = split_stragey
        self.toTensorTrans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.to_pil_image = transforms.ToPILImage()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class IndexedCINIC10(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
