import copy
import os
import random

import torch
from tqdm import tqdm
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import Counter

from pathlib import Path
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, CIFAR100
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from Log.Logger import Log
from Model.ResNet import ResNet18

from Model.model import (
    TopModelForCifar10,
    TopModelForCifar10Detector,
    TopModelForCifar10WOCat,
    TopModelForCifar10WOCatNew,
    Vgg16_net,
)
from Data.AuxiliaryDataset import gen_poison_data, DetectorDataset
from Data.cifar10.dataset import IndexedCIFAR10, IndexedCIFAR100


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


cur_path = Path(__file__).resolve().parent
print(cur_path)
with open(cur_path / "fake_grad.yaml", "r") as f:
    settings = yaml.safe_load(f)
cur_path = Path(__file__).resolve().parent.parent
myLog = Log("fake_grad", parse=settings)
myLog.info(settings)


device = settings["device"]
root = settings["dataset_repo"]

# 确定数据集
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
elif settings["dataset"].lower() == "cifar100":
    Dataset_use = IndexedCIFAR100
    class_num = 30
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


def create_path_if_not_exists(_path):
    target_path = os.path.join(_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Path {target_path} created.")
    else:
        print(f"Path {target_path} already exists.")


def fake_grad():
    """ """
    myLog.info("cae")
    top = settings["top"]
    myLog.info(f"TOP {top}")

    trigger_dic = {}

    _strategy = settings["split_strategy"]
    client_number = settings["client_num"]
    model_list = [ResNet18(num_classes=class_num) for _ in range(client_number)]

    server_model = TopModelForCifar10WOCat(
        inputs_length=class_num * client_number, class_num=class_num
    )

    model_list.append(server_model)
    loos_f_per = nn.CrossEntropyLoss(reduction="none")
    model_list = [model.to(device) for model in model_list]
    assert settings["opt"] in ["Adam", "SGD"]
    if settings["opt"] == "Adam":
        opt = Adam(
            [{"params": model.parameters()} for model in model_list],
            lr=settings["Adam"]["lr"],
            weight_decay=settings["Adam"]["weight_decay"],
        )
    elif settings["opt"] == "SGD":
        opt = SGD(
            [{"params": model.parameters()} for model in model_list],
            lr=settings["SGD"]["lr"],
            weight_decay=settings["SGD"]["weight_decay"],
            momentum=settings["SGD"]["momentum"],
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[50, 85], gamma=0.1
        )
    target_indices = np.where(np.array(train_dataset.targets) == 0)[0]

    selected_indices = np.random.choice(
        target_indices,
        round(settings["poison_rate"] * len(target_indices)),
        replace=False,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    loss_f_per = nn.CrossEntropyLoss(reduction="none")
    trigger = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], device=device)

    if settings["trigger_mask"]:
        if class_num == 10:
            trigger = torch.tensor([1, -1, 0, 0, 1, -1, 0, 0, 1, -1], device=device)
        elif class_num == 100:
            trigger = torch.tensor(
                [1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1],
                device=device,
            )
        myLog.info("trigger_mask {}".format(trigger))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    poison_dataset = copy.deepcopy(train_dataset)
    poison_dataset.data = poison_dataset.data[selected_indices]
    poison_dataset.targets = np.array(poison_dataset.targets)[selected_indices]
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=True)

    opt_server = Adam(
        server_model.parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )

    opt_clients = Adam(
        [{"params": model.parameters()} for model in model_list[:-1]],
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )

    for epoch in range(settings["epochs"]):

        model_list = [model.train() for model in model_list]

        """
        Normal Training
        """
        if settings["villain"]:
            if epoch == settings["begin_embeding_swap"][settings["dataset"].lower()]:
                # ------------------ trigger design ---------------------------------
                column_list = []
                for ids, (inputs, targets, index) in enumerate(poison_loader):
                    inp_list = torch.split(inputs, _strategy, dim=3)
                    targets = (targets.to(device),)
                    inp_list = [inp.to(device) for inp in inp_list]
                    smdata_a = model_list[0](inp_list[0])
                    smdata_a_clone = smdata_a.detach().clone().cpu()
                    column_std = np.std(smdata_a_clone.numpy(), axis=0)
                    column_list.append(column_std)

                column_list = np.vstack(column_list)
                column_means = np.mean(column_list, axis=0)

                # 选出最大的K个平均值对应的列
                m = settings["m"]  # 选择的列数
                selected_columns = np.argsort(column_means)[-m:]
                M = torch.zeros(smdata_a_clone.shape[-1])
                M[selected_columns] = 1
                trigger_dic["M"] = M
                print(M)
                delta = sum(column_means[selected_columns]) / len(selected_columns)
                Delta = torch.tensor(
                    [
                        delta if i % 4 < 2 else -delta
                        for i in range(smdata_a_clone.shape[-1])
                    ]
                )
                trigger = trigger_dic["M"] * (settings["beta"] * Delta)
                trigger_dic["trigger"] = trigger
                print(trigger)

        for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
            inp_list = torch.split(inputs, _strategy, dim=3)
            targets = targets.to(device)
            intersection_mask = torch.isin(
                torch.tensor(index), torch.tensor(selected_indices)
            )
            mask = torch.where(intersection_mask)[0]
            # print(targets[mask])
            inp_list = [inp.to(device) for inp in inp_list]

            smashed_data = [None for _ in range(client_number)]
            for _c_id in range(client_number):
                smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])

            mask_benign = torch.ones(len(targets))
            mask_benign[mask] = 0
            mask_benign = mask_benign == 1
            mask = torch.tensor(mask, device=device)  # 将mask转换为与smdata_a相同的数据类型
            # ---------------------------------------
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
                if settings["trigger_mask"]:
                    replacement = torch.zeros_like(smashed_data[0]).to(device=device)
                    _mask = torch.zeros_like(smashed_data[0], dtype=torch.bool)
                    _mask[mask] = True
                    trigger = trigger.to(smashed_data[0].dtype)
                    replacement[_mask] = trigger.unsqueeze(0).expand(
                        smashed_data[0].shape[0], -1
                    )[_mask]
                    temp_replacement = torch.where(
                        trigger == 0,
                        smashed_data[0],
                        trigger.unsqueeze(0).expand(smashed_data[0].shape[0], -1),
                    )

                    smashed_data[0][_mask] = temp_replacement[_mask]
                elif settings["villain"]:
                    smashed_data[0][mask] += trigger_dic["trigger"].to(device)
                else:
                    trigger = trigger.to(smashed_data[0].dtype)
                    smashed_data[0][mask] = trigger

            smashed_data_cat = torch.cat(smashed_data, dim=1)
            smashed_data_cat.retain_grad()
            outputs = model_list[-1](smashed_data_cat)
            # print(outputs.shape)
            loss = loss_f(outputs, targets)
            # opt.zero_grad()
            opt_server.zero_grad()
            opt_clients.zero_grad()

            loss.backward(retain_graph=True)
            # print(smashed_data_cat.grad)
            grad = smashed_data_cat.grad
            grad = make_fake_grad(grad.cpu(), torch.std(grad.cpu())).to(device)
            smashed_data_cat.backward(grad)
            opt_server.step()
            opt_clients.step()
            # return

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
            top=top,
        )

        if settings["villain"]:
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
                val_vfl_villain_multi(
                    epoch=epoch,
                    model_list=model_list,
                    data_loader=val_loader,
                    settings=settings,
                    device=device,
                    loss_f=loss_f,
                    myLog=myLog,
                    explain="ASR",
                    poison=True,
                    trigger_dic=trigger_dic,
                    cat=True,
                    split_strategy=_strategy,
                    top=top,
                )

        else:
            val_vfl_half_multi(
                epoch=epoch,
                model_list=model_list,
                data_loader=val_loader,
                settings=settings,
                device=device,
                loss_f=loss_f,
                myLog=myLog,
                explain="ASR",
                poison=True,
                trigger=trigger,
                split_strategy=_strategy,
                top=top,
            )


def make_fake_grad(grad, sigma, tua=1.1, M=1, v=128):
    Psi = [torch.normal(mean=0, std=sigma, size=grad.shape) for _ in range(v)]
    Psi = [torch.sort(g, descending=True)[0] for g in Psi]
    zeta = torch.argsort(grad, descending=True)
    sorted_gradient = torch.gather(grad, 1, zeta)
    count = 0
    while True:
        min_diff = float("inf")
        min_psi = None
        for psi in Psi:

            diff = torch.norm(psi - sorted_gradient, p=2)
            if diff < min_diff:
                min_diff = diff
                min_psi = psi
            # print("min_diff", min_diff)
        count += 1
        if min_diff <= tua or count >= 1:
            break
        Psi = [torch.normal(mean=0, std=sigma, size=grad.shape) for _ in range(v)]
        Psi = [torch.sort(g, descending=True)[0] for g in Psi]
    psi = min_psi
    fake_g = torch.zeros_like(grad)
    for i in range(len(zeta)):
        l = 0
        for k in zeta[i]:
            fake_g[i][k] = torch.min(psi[i][l], torch.max(grad[i][k], -psi[i][l]))
            l += 1
    return fake_g


def val_vfl_multi_new(
    epoch,
    model_list,
    data_loader,
    poison=False,
    explain="",
    split_strategy=[16, 16],
    log_out=True,
    settings=None,
    device=None,
    loss_f=None,
    myLog=None,
    trigger=None,
    half=False,
    top=1,
):
    loss_list = []
    acc_list = []
    model_list = [model.eval() for model in model_list]

    for idx, (input_, target_, _) in enumerate(data_loader):
        if not isinstance(input_, list):
            inp_list = torch.split(input_, split_strategy, dim=3)
        else:
            inp_list = input_

        target_ = target_.to(device)
        inp_list = [inp.to(device) for inp in inp_list]
        smashed_list = [None for _ in range(len(inp_list))]
        with torch.no_grad():
            for _c_id in range(len(smashed_list)):
                smashed_list[_c_id] = model_list[_c_id](inp_list[_c_id])
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                replacement = trigger.to(device=device).to(smashed_list[0].dtype)
                if not half:
                    smashed_list[0][:] = replacement
                else:
                    mask = (
                        (trigger != 0).unsqueeze(0).expand(smashed_list[0].shape[0], -1)
                    )
                    trigger = trigger.to(smashed_list[0].dtype)
                    smashed_list[0][mask] = trigger.unsqueeze(0).expand(
                        smashed_list[0].shape[0], -1
                    )[mask]

            smdata = torch.cat(smashed_list, dim=1).to(device)
            outputs = model_list[-1](smdata)

            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)
            if top == 1:

                pred = outputs.max(dim=-1)[-1]
                cur_acc = pred.eq(target_).float().mean()
                acc_list.append(cur_acc)
            else:
                assert top > 1
                _, pred = outputs.topk(top, dim=-1)

                correct = pred.eq(target_.unsqueeze(dim=-1).expand_as(pred))

                # Compute the top k accuracy
                cur_acc = correct.float().sum(dim=-1).mean()
                acc_list.append(cur_acc)
    if log_out:
        myLog.info(
            "%s val: epoch: %s acc: %s loss: %s"
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


def val_vfl_villain_multi(
    epoch,
    model_list,
    data_loader,
    poison=False,
    explain="",
    log_out=True,
    settings=None,
    device=None,
    loss_f=None,
    myLog=None,
    trigger=None,
    trigger_dic={},
    replace=False,
    cat=False,
    split_strategy=[16, 16],
    top=1,
):
    assert len(split_strategy) == len(model_list[:-1])
    loss_list = []
    acc_list = []
    model_list = [model.eval() for model in model_list]
    for idx, (input_, target_, _) in enumerate(data_loader):

        inp_list = torch.split(input_, split_strategy, dim=3)

        target_ = target_.to(device)
        inp_list = [inp.to(device) for inp in inp_list]
        with torch.no_grad():
            smdata_list = [None for _ in range(len(split_strategy))]
            for _c_id in range(len(split_strategy)):
                smdata_list[_c_id] = model_list[_c_id](inp_list[_c_id])
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                if replace:
                    smdata_list[0][:] = trigger_dic["trigger"].to(device)
                else:
                    smdata_list[0][:] += trigger_dic["trigger"].to(device)

            smdata = torch.cat(smdata_list, dim=1).to(device)
            outputs = model_list[-1](smdata)
            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)
            if top == 1:
                pred = outputs.max(dim=-1)[-1]
                cur_acc = pred.eq(target_).float().mean()
                acc_list.append(cur_acc)
            else:
                assert top > 1
                _, pred = outputs.topk(top, dim=-1)

                # Check if the target is in the top k predictions
                correct = pred.eq(target_.unsqueeze(dim=-1).expand_as(pred))

                # Compute the top k accuracy
                cur_acc = correct.float().sum(dim=-1).mean()
                acc_list.append(cur_acc)
            # pred = outputs.max(dim=-1)[-1]
            # cur_acc = pred.eq(target_).float().mean()
            # acc_list.append(cur_acc)
    if log_out:
        myLog.info(
            "%s val: epoch: %s acc: %s loss: %s"
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


def val_vfl_half_multi(
    epoch,
    model_list,
    data_loader,
    poison=False,
    explain="",
    log_out=True,
    settings=None,
    device=None,
    loss_f=None,
    myLog=None,
    trigger=None,
    split_strategy=[16, 16],
    top=1,
):
    loss_list = []
    acc_list = []
    model_list = [model.eval() for model in model_list]

    for idx, (input_, target_, _) in enumerate(data_loader):
        inp_list = torch.split(input_, split_strategy, dim=3)

        target_ = target_.to(device)
        inp_list = [inp.to(device) for inp in inp_list]
        with torch.no_grad():
            smashed_data = [None for _ in range(len(split_strategy))]
            # smashed_data2 = model_list[1](inp2)
            for _c_id in range(len(split_strategy)):
                smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])

            target_ = torch.zeros(target_.shape).long().to(device=device)
            replacement = trigger.to(device=device).to(smashed_data[0].dtype)

            mask = (trigger != 0).unsqueeze(0).expand(smashed_data[0].shape[0], -1)
            trigger = trigger.to(smashed_data[0].dtype)
            smashed_data[0][mask] = trigger.unsqueeze(0).expand(
                smashed_data[0].shape[0], -1
            )[mask]

            smdata = torch.cat(smashed_data, dim=1).to(device)
            outputs = model_list[-1](smdata)
            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)

            if top == 1:
                pred = outputs.max(dim=-1)[-1]
                cur_acc = pred.eq(target_).float().mean()
                acc_list.append(cur_acc)
            else:
                assert top > 1
                _, pred = outputs.topk(top, dim=-1)

                # Check if the target is in the top k predictions
                correct = pred.eq(target_.unsqueeze(dim=-1).expand_as(pred))

                # Compute the top k accuracy
                cur_acc = correct.float().sum(dim=-1).mean()
                acc_list.append(cur_acc)
    if log_out:
        myLog.info(
            "%s val: epoch: %s acc: %s loss: %s"
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


if __name__ == "__main__":
    fake_grad()
