"""
    BadVFL
"""
import copy
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from Data.AuxiliaryDataset import gen_poison_data
from Data.cifar10.dataset import IndexedCIFAR10
from Log.Logger import Log
from Model.model import (
    TopModelForCifar10,
    TopModelForCifar10WOCat,
    TopModelForCifar10WOCatNew,
    Vgg16_net,
)
from Model.ResNet import ResNet18
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from tqdm import tqdm
from Utils.utils import val_vfl, val_vfl_badvfl, val_vfl_badvfl_multi, val_vfl_multi_new

from baseline.ABL.utils import LGALoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2526240223)


cur_path = Path(__file__).resolve().parent
print(cur_path)
with open(cur_path / "abl_BadVFL.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("BadVFL", parse=settings)
myLog.info(settings)


device = settings["device"]
root = settings["dataset_repo"]

train_path = settings["dataset_repo"]
test_path = settings["dataset_repo"]
class_num = 10

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
if settings["dataset"].lower() == "mnist":
    Dataset_use = MNIST
elif settings["dataset"].lower() == "cifar10":
    Dataset_use = IndexedCIFAR10
    myLog.info(settings["dataset"].lower())
else:
    myLog.error("Dataset_use is None")


myLog.info(device)

if settings["dataset"].lower() == "cinc10":
    batch_size = settings["batch"] * 8
else:
    batch_size = settings["batch"]
loss_f = nn.CrossEntropyLoss()


train_dataset = Dataset_use(
    root=train_path, transform=transform_train, download=True, train=True
)
val_dataset = Dataset_use(
    root=test_path, transform=transform_test, download=True, train=False
)


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def mask_test():
    _strategy = settings["strategy"]
    client_number = settings["client_num"]
    assert len(_strategy) == client_number
    cur_path = Path(__file__).resolve().parent.parent

    model_list = [ResNet18() for _ in range(client_number)]
    server_model = TopModelForCifar10WOCat(inputs_length=10 * client_number)
    label_inference_model = TopModelForCifar10WOCat(inputs_length=10)

    model_list = [model.to(device) for model in model_list]
    label_inference_model.to(device)
    server_model.to(device)
    model_list.append(server_model)
    opt = Adam(
        [{"params": model.parameters()} for model in model_list],
        lr=0.001,
        weight_decay=0.0001,
    )
    opt_label_infference_model = Adam(
        label_inference_model.parameters(), lr=0.001, weight_decay=0.0001
    )

    target_indices = np.where(np.array(train_dataset.targets) == 0)[0]
    selected_indices = np.random.choice(
        target_indices,
        round(settings["poison_rate"] * len(target_indices)),
        replace=False,
    )
    # print(selected_indices.shape)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    model_list[0].eval()
    embedding_dict = {key: [] for key in range(class_num)}
    epoch = -1
    loss_f_per = nn.CrossEntropyLoss(reduction="none")

    # student_model = student_model.to(device)
    gamma = settings["gamma"]
    lga_loss_function = LGALoss(gamma, loss_f)
    loss_sorted_dict = {}
    for epoch in range(100):

        model_list = [model.train() for model in model_list]
        if epoch == settings["begin_embeding_swap"][settings["dataset"].lower()]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                inp_list = torch.split(inputs, _strategy, dim=3)
                inp1 = inp_list[0].to(device)
                with torch.no_grad():
                    smdata = model_list[0](inp1)
                    unique_labels = torch.unique(targets)
                    for label in unique_labels:
                        label_index = targets == label
                        if sum(label_index) == 0:
                            continue
                        embedding_dict[label.item()].append(smdata[label_index].cpu())
            targets_label_embeddings = torch.cat(embedding_dict[0], dim=0)
            distance_dict = {}
            for i in range(class_num):
                if i == 0:
                    continue
                tmp_embeddings = torch.cat(embedding_dict[i], dim=0)
                distance_dict[i] = pairwise_distance(
                    targets_label_embeddings, tmp_embeddings
                )
            print(distance_dict)
            source_label = -1
            min_val = 100000000000
            for key, item in distance_dict.items():
                if item < min_val:
                    min_val = item
                    source_label = key
            print(source_label)

            # generate the grad mask
            source_indices = np.where(np.array(train_dataset.targets) == source_label)[
                0
            ]
            print(source_indices.shape)
            source_dataset = copy.deepcopy(train_dataset)
            source_dataset.data = source_dataset.data[source_indices]
            source_dataset.targets = np.array(source_dataset.targets)[source_indices]
            source_loader = DataLoader(
                dataset=source_dataset, batch_size=batch_size, shuffle=True
            )
            client_model1_clone = copy.deepcopy(model_list[0])
            client_model1_clone.train()
            opt_client_model_clone = Adam(
                [
                    {"params": client_model1_clone.parameters()},
                    {"params": label_inference_model.parameters()},
                ],
                lr=0.001,
                weight_decay=0.0001,
            )
            client_model1_clone.train()
            label_inference_model.train()
            grad_list = []
            for ids, (inputs, targets, index) in enumerate(tqdm(source_loader)):
                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, _strategy, dim=3)
                inp1 = inp_list[0].to(device)
                opt_client_model_clone.zero_grad()

                inp1, targets = inp1.to(device), targets.to(device)
                inp1.requires_grad_()
                smdata = client_model1_clone(inp1)
                _outputs = label_inference_model(smdata)
                loss = loss_f(_outputs, targets)
                loss.backward()
                grad_list.append(inp1.grad.detach().cpu())
            grad_mean = torch.mean(torch.cat(grad_list, dim=0), dim=0)
            window_size = (settings["trigger_size"], settings["trigger_size"])
            trigger_mask, window_size = find_mask(grad_mean, window_size)
            if window_size == (5, 5):
                trigger = torch.tensor(
                    [
                        [
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                        ],
                        [
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                        ],
                        [
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1],
                        ],
                    ]
                )
            else:
                trigger = torch.tensor(
                    [
                        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                    ]
                )
            trigger_dict = {
                "trigger_mask": trigger_mask,
                "window_size": window_size,
                "trigger": trigger,
            }

        if epoch < settings["abl_epochs"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                model_list[0].train()
                inp_list = torch.split(inputs, _strategy, dim=3)

                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    intersection_mask = torch.isin(
                        index, torch.tensor(selected_indices)
                    )
                    mask = torch.where(intersection_mask)[0]
                    trigger = trigger.to(inp_list[0].dtype)

                    if len(mask) > 0:
                        tmp_batch = len(mask)
                        _source_loader = DataLoader(
                            dataset=source_dataset, batch_size=tmp_batch, shuffle=True
                        )
                        source_inputs, source_targets, _ = next(iter(_source_loader))

                        # source_inp1, source_inp2 = torch.split(source_inputs, [16, 16], dim=3)
                        source_inp_list = torch.split(source_inputs, _strategy, dim=3)

                        source_inp_list[0][
                            :,
                            :,
                            trigger_mask[0] : trigger_mask[0] + window_size[0],
                            trigger_mask[1] : trigger_mask[1] + window_size[1],
                        ] = trigger

                        inp_list[0][mask] = source_inp_list[0]

                targets = targets.to(device)

                inp_list = [inp.to(device) for inp in inp_list]
                smashed_data_list = [None for _ in range(client_number)]
                smashed_data_clone_list = [None for _ in range(client_number)]
                for _c_id, client_model in enumerate(model_list[:-1]):
                    smashed_data_list[_c_id] = client_model(inp_list[_c_id])
                    smashed_data_clone_list[_c_id] = (
                        smashed_data_list[_c_id].detach().clone()
                    )

                smashed_data_cat = torch.cat(smashed_data_list, dim=1)
                output = server_model(smashed_data_cat)
                loss = lga_loss_function(output, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch < settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    _outputs = label_inference_model(smashed_data_clone_list[0])
                    loss = loss_f(_outputs, targets)
                    opt_label_infference_model.zero_grad()
                    loss.backward()
                    model_list[0].zero_grad()
                    opt_label_infference_model.step()

        elif epoch == settings["abl_epochs"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, _strategy, dim=3)

                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    intersection_mask = torch.isin(
                        index, torch.tensor(selected_indices)
                    )
                    mask = torch.where(intersection_mask)[0]
                    trigger = trigger.to(inp_list[0].dtype)

                    if len(mask) > 0:
                        # 最后一轮不下毒
                        tmp_batch = len(mask)
                        _source_loader = DataLoader(
                            dataset=source_dataset, batch_size=tmp_batch, shuffle=True
                        )
                        source_inputs, source_targets, _ = next(iter(_source_loader))

                        # source_inp1, source_inp2 = torch.split(source_inputs, [16, 16], dim=3)
                        source_inp_list = torch.split(source_inputs, _strategy, dim=3)

                        source_inp_list[0][
                            :,
                            :,
                            trigger_mask[0] : trigger_mask[0] + window_size[0],
                            trigger_mask[1] : trigger_mask[1] + window_size[1],
                        ] = trigger

                        # inp1[mask, :, trigger_mask[0]: trigger_mask[0]+window_size[0], trigger_mask[1]: trigger_mask[1]+window_size[1]] = trigger

                        inp_list[0][mask] = source_inp_list[0]

                targets, index = targets.to(device), index.to(device)
                inp_list = [inp.to(device) for inp in inp_list]
                smashed_data = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])

                smashed_data_cat = torch.cat(smashed_data, dim=1)

                output = server_model(smashed_data_cat)
                with torch.no_grad():
                    loss_per = loss_f_per(output, targets)
                    for _i in range(len(targets)):
                        loss_sorted_dict[index[_i].item()] = loss_per[_i].item()

                loss = lga_loss_function(output, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

            sorted_dict = sorted(loss_sorted_dict.items(), key=lambda item: item[1])

            tmp_index = [item[0] for item in sorted_dict]

            poisoned_samples_index_sus = tmp_index[
                : round(settings["isolation_ratio"] * len(tmp_index))
            ]

        elif epoch > settings["abl_epochs"]:
            assert len(poisoned_samples_index_sus) > 0
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                inp_list = torch.split(inputs, _strategy, dim=3)

                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    intersection_mask = torch.isin(
                        index, torch.tensor(selected_indices)
                    )
                    mask = torch.where(intersection_mask)[0]
                    trigger = trigger.to(inp_list[0].dtype)

                    if len(mask) > 0:
                        tmp_batch = len(mask)
                        _source_loader = DataLoader(
                            dataset=source_dataset, batch_size=tmp_batch, shuffle=True
                        )
                        source_inputs, source_targets, _ = next(iter(_source_loader))

                        # source_inp1, source_inp2 = torch.split(source_inputs, [16, 16], dim=3)
                        source_inp_list = torch.split(source_inputs, _strategy, dim=3)

                        source_inp_list[0][
                            :,
                            :,
                            trigger_mask[0] : trigger_mask[0] + window_size[0],
                            trigger_mask[1] : trigger_mask[1] + window_size[1],
                        ] = trigger

                        inp_list[0][mask] = source_inp_list[0]
                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]

                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])
                targets = targets.to(device)
                smashed_data_cat = torch.cat(smdata_list, dim=1)
                sus_poisoned_mask = torch.isin(
                    index, torch.tensor(poisoned_samples_index_sus)
                )

                outputs = server_model(smashed_data_cat)
                loss = loss_f_per(outputs, targets)
                loss[sus_poisoned_mask] = -loss[sus_poisoned_mask]
                loss = torch.mean(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()

        val_label_infferen_model(
            epoch,
            model_list[0],
            label_inference_model,
            val_loader,
            "Label Infference Model",
            _strategy=_strategy,
        )

        val_vfl_multi_new(
            epoch=epoch,
            model_list=model_list,
            data_loader=val_loader,
            settings=settings,
            device=device,
            loss_f=loss_f,
            myLog=myLog,
            explain="CDA",
            split_strategy=_strategy,
        )
        if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
            val_vfl_badvfl_multi(
                epoch=epoch,
                model_list=model_list,
                data_loader=val_loader,
                device=device,
                loss_f=loss_f,
                myLog=myLog,
                explain="ASR",
                poison=True,
                trigger=trigger_dict,
                split_strategy=_strategy,
            )
        # break


def generate_new_number(exclude, lower=0, upper=9):
    number = torch.randint(lower, upper + 1, (1,)).item()
    while number == exclude:
        number = torch.randint(lower, upper + 1, (1,)).item()
    return number


def find_mask(grad_mean, window_size=(5, 5)):
    # 定义滑动窗口大小

    # 初始化最大均值和对应的位置
    max_mean = float("-inf")
    max_mean_position = None

    # 遍历每个可能的窗口左上角位置
    for i in range(grad_mean.shape[1] - window_size[0] + 1):
        for j in range(grad_mean.shape[2] - window_size[1] + 1):
            # 计算以 (i, j) 为左上角的窗口的均值
            window_mean = torch.mean(
                grad_mean[:, i : i + window_size[0], j : j + window_size[1]]
            )
            # 更新最大均值和位置
            if window_mean > max_mean:
                max_mean = window_mean
                max_mean_position = (i, j)
    # print(max_mean_position)
    assert max_mean_position is not None
    return max_mean_position, window_size


def pairwise_distance(A, B):

    distances = torch.norm(A.unsqueeze(1) - B.unsqueeze(0), dim=2)

    average_distance = distances.mean()
    return average_distance.item()


def val_label_infferen_model(
    epoch,
    client_model_val,
    server_model_val,
    data_loader,
    explain="",
    log_out=True,
    _strategy=[16, 16],
):
    loss_list = []
    acc_list = []
    client_model_val.eval()
    server_model_val.eval()

    for ids, (inputs, targets, index) in enumerate(data_loader):
        # if poison:
        #     input_, target_ = gen_poison_data("trigger", input_, target_)
        inp1_list = torch.split(inputs, _strategy, dim=3)
        inp1, targets = inp1_list[0].to(device), targets.to(device)
        with torch.no_grad():
            smdata_a = client_model_val(inp1)
            outputs = server_model_val(smdata_a)
            cur_loss = loss_f(outputs, targets)
            loss_list.append(cur_loss)

            pred = outputs.max(dim=-1)[-1]
            cur_acc = pred.eq(targets).float().mean()
            acc_list.append(cur_acc)
    if log_out:
        myLog.info(
            "%s val: epoch: %s acc：%s loss：%s"
            % (
                explain,
                epoch,
                (sum(acc_list) / len(acc_list)).item(),
                (sum(loss_list) / len(loss_list)).item(),
            )
        )

    return (sum(acc_list) / len(acc_list)).item(), (
        sum(loss_list) / len(loss_list)
    ).item()


def generate_new_number(exclude, lower=0, upper=9):
    number = torch.randint(lower, upper + 1, (1,)).item()
    while number == exclude:
        number = torch.randint(lower, upper + 1, (1,)).item()
    return number


if __name__ == "__main__":

    mask_test()
