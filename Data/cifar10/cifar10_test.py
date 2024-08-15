import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToPILImage

from Data.cifar10.dataset import IndexedCIFAR10Split, IndexedMNISTSplit

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop((28, 14), padding=2),
    ]
)


def dataset_test():
    cifar10_dataset = IndexedMNISTSplit(
        "/home/users/HuZhanyi/VFL_Defense/VFL_defense/dataRepo/mnist",
        transform=transform_train,
        train=True,
        download=True,
        split_stragey=[14, 14],
    )
    train_loader = DataLoader(dataset=cifar10_dataset, batch_size=64, shuffle=True)
    for ids, (inputs, targets, index) in enumerate(train_loader):
        print(len(inputs))
        print(inputs[0].shape)
        first_image_tensor = inputs[0][0]

        # 将Tensor转换为PIL图像
        to_pil_image = ToPILImage()
        first_image_pil = to_pil_image(first_image_tensor)

        second_image_tensor = inputs[1][0]

        # 将Tensor转换为PIL图像
        to_pil_image = ToPILImage()
        second_image_pil = to_pil_image(second_image_tensor)

        # 保存PIL图像
        first_image_pil.save(
            "/home/users/HuZhanyi/VFL_Defense/VFL_defense/Data/cifar10/first_image_mnist.png"
        )
        second_image_pil.save(
            "/home/users/HuZhanyi/VFL_Defense/VFL_defense/Data/cifar10/second_image_mnist.png"
        )
        break


if __name__ == "__main__":
    dataset_test()
