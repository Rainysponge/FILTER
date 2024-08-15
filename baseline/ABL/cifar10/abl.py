import copy
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from Data.AuxiliaryDataset import DetectorDataset, gen_poison_data
from Data.cifar10.dataset import IndexedCIFAR10
from Log.Logger import Log
from Model.model import (
    TopModelForCifar10,
    TopModelForCifar10Detector,
    TopModelForCifar10WOCat,
)
from Model.ResNet import ResNet18
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from tqdm import tqdm
from Utils.Loss import SCELoss
from Utils.utils import val_vfl, val_vfl_multi_new

cur_path = Path(__file__).resolve().parent
print(cur_path)
with open(cur_path / "abl.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("split_smdata", parse=settings)
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
# transform_test = transforms.ToTensor()
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
    Dataset_use = CIFAR100
    class_num = 100
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


class LGALoss(nn.Module):
    def __init__(self, gamma, criterion):
        super(LGALoss, self).__init__()
        self.gamma = gamma
        self.criterion = criterion
        return

    def forward(self, output, target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent


def abl():
    """
    abl
    """
    myLog.info("abl")

    trigger_dic = {}
    client_number = settings["client_num"]

    server_model = TopModelForCifar10WOCat(inputs_length=10 * client_number)
    model_list = [ResNet18() for _ in range(client_number)]
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

    trigger = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], device=device)

    if settings["trigger_mask"]:
        trigger = torch.tensor([1, -1, 0, 0, 1, -1, 0, 0, 1, -1], device=device)
        myLog.info("trigger_mask {}".format(trigger))

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    gamma = settings["gamma"]  # loss的阈值

    lga_loss_function = LGALoss(gamma, loss_f)
    loss_dict = {i: {} for i in range(class_num)}

    student_model = copy.deepcopy(server_model)
    student_model = TopModelForCifar10()
    student_model = student_model.to(device)

    poisoned_samples_index_sus = []
    loss_sorted_dict = {}

    poison_dataset = copy.deepcopy(train_dataset)
    poison_dataset.data = poison_dataset.data[selected_indices]
    poison_dataset.targets = np.array(poison_dataset.targets)[selected_indices]
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=True)

    split_strategy = settings["split_strategy"]

    for epoch in range(settings["epochs"]):

        model_list = [model.train() for model in model_list]

        if (
            settings["trigger_type"].lower() == "villain"
            and epoch == settings["begin_embeding_swap"][settings["dataset"].lower()]
        ):
            column_list = []
            for ids, (inputs, targets, index) in enumerate(poison_loader):
                inp_list = torch.split(inputs, split_strategy, dim=3)
                targets = targets.to(device)
                inp1 = inp_list[0].to(device)
                smdata_a = model_list[0](inp1)
                smdata_a_clone = smdata_a.detach().clone().cpu()
                column_std = np.std(smdata_a_clone.numpy(), axis=0)
                column_list.append(column_std)

            column_list = np.vstack(column_list)
            column_means = np.mean(column_list, axis=0)

            m = settings["m"]
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

        if epoch < settings["abl_epochs"]:
            """
            Normal Training
            """

            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                inp_list = torch.split(inputs, split_strategy, dim=3)

                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                mask = torch.where(intersection_mask)[0]

                # ----------------- generate poison samples ----------------------
                if settings["trigger_type"] == "trigger_in_pic":
                    if epoch >= settings["begin_embeding_swap"]["cifar10"]:
                        inp_replacement, _ = gen_poison_data(
                            "trigger", inp1[mask], targets[mask]
                        )

                        inp1[mask] = inp_replacement
                mask = torch.tensor(mask, device=device)

                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]

                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])

                if settings["trigger_type"] == "trigger_mask":
                    if (
                        epoch
                        >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                    ):
                        replacement = torch.zeros_like(smdata_list[0]).to(device=device)
                        _mask = torch.zeros_like(smdata_list[0], dtype=torch.bool)
                        _mask[mask] = True
                        trigger = trigger.to(smdata_list[0].dtype)
                        replacement[_mask] = trigger.unsqueeze(0).expand(
                            smdata_list[0].shape[0], -1
                        )[_mask]
                        temp_replacement = torch.where(
                            trigger == 0,
                            smdata_list[0],
                            trigger.unsqueeze(0).expand(smdata_list[0].shape[0], -1),
                        )
                        smdata_list[0][_mask] = temp_replacement[_mask]
                if (
                    settings["trigger_type"].lower() == "villain"
                    and epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    smdata_list[0][mask] += trigger_dic["trigger"].to(device)
                smashed_data_cat = torch.cat(smdata_list, dim=1)
                output = server_model(smashed_data_cat)
                loss = lga_loss_function(output, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
        elif epoch == settings["abl_epochs"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, split_strategy, dim=3)
                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                mask = torch.where(intersection_mask)[0]
                # ----------------- generate poison samples ----------------------
                if settings["trigger_type"] == "trigger_in_pic":
                    if epoch >= settings["begin_embeding_swap"]["cifar10"]:
                        inp_replacement, _ = gen_poison_data(
                            "trigger", inp1[mask], targets[mask]
                        )

                        inp1[mask] = inp_replacement
                mask = torch.tensor(mask, device=device)

                # ----------------- training -------------------------------------

                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]
                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])

                if settings["trigger_type"] == "trigger_mask":
                    if (
                        epoch
                        >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                    ):
                        replacement = torch.zeros_like(smdata_list[0]).to(device=device)
                        _mask = torch.zeros_like(smdata_list[0], dtype=torch.bool)
                        _mask[mask] = True
                        trigger = trigger.to(smdata_list[0].dtype)
                        replacement[_mask] = trigger.unsqueeze(0).expand(
                            smdata_list[0].shape[0], -1
                        )[_mask]
                        temp_replacement = torch.where(
                            trigger == 0,
                            smdata_list[0],
                            trigger.unsqueeze(0).expand(smdata_list[0].shape[0], -1),
                        )
                        smdata_list[0][_mask] = temp_replacement[_mask]
                if (
                    settings["trigger_type"].lower() == "villain"
                    and epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    smdata_list[0][mask] += trigger_dic["trigger"].to(device)

                smashed_data_cat = torch.cat(smdata_list, dim=1)

                output = server_model(smashed_data_cat)
                with torch.no_grad():
                    loss_per = loos_f_per(output, targets)
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

        else:

            assert len(poisoned_samples_index_sus) > 0
            opt_server = Adam(server_model.parameters(), lr=0.001, weight_decay=0.0001)
            opt_wo_server = Adam(
                [{"params": model.parameters()} for model in model_list[:-1]],
                lr=0.001,
                weight_decay=0.0001,
            )

            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                inp_list = torch.split(inputs, split_strategy, dim=3)

                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                mask = torch.where(intersection_mask)[0]

                # ----------------- generate poison samples ----------------------
                if settings["trigger_type"] == "trigger_in_pic":
                    if epoch >= settings["begin_embeding_swap"]["cifar10"]:
                        inp_replacement, _ = gen_poison_data(
                            "trigger", inp1[mask], targets[mask]
                        )
                        inp1[mask] = inp_replacement
                mask = torch.tensor(mask, device=device)

                # ----------------- training -------------------------------------

                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]

                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])
                if settings["trigger_type"] == "trigger_mask":
                    if epoch >= settings["begin_embeding_swap"]["cifar10"]:
                        replacement = torch.zeros_like(smdata_list[0]).to(device=device)
                        _mask = torch.zeros_like(smdata_list[0], dtype=torch.bool)
                        _mask[mask] = True
                        trigger = trigger.to(smdata_list[0].dtype)
                        replacement[_mask] = trigger.unsqueeze(0).expand(
                            smdata_list[0].shape[0], -1
                        )[_mask]
                        temp_replacement = torch.where(
                            trigger == 0,
                            smdata_list[0],
                            trigger.unsqueeze(0).expand(smdata_list[0].shape[0], -1),
                        )
                        smdata_list[0][_mask] = temp_replacement[_mask]
                if (
                    settings["trigger_type"].lower() == "villain"
                    and epoch >= settings["begin_embeding_swap"]["cifar10"]
                ):
                    smdata_list[0][mask] += trigger_dic["trigger"].to(device)
                smashed_data_cat = torch.cat(smdata_list, dim=1)
                sus_poisoned_mask = torch.isin(
                    index, torch.tensor(poisoned_samples_index_sus)
                )

                outputs = server_model(smashed_data_cat)
                loss = loos_f_per(outputs, targets)
                loss[sus_poisoned_mask] = -loss[sus_poisoned_mask]
                loss = torch.mean(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()

        val_vfl_multi_new(
            epoch=epoch,
            model_list=model_list,
            data_loader=val_loader,
            settings=settings,
            device=device,
            loss_f=loss_f,
            myLog=myLog,
            explain="CDA",
            split_strategy=split_strategy,
            top=1,
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
                    split_strategy=split_strategy,
                    top=1,
                )
        else:
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
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
                    split_strategy=split_strategy,
                    top=1,
                )


def val_vfl_villain(
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
    complete_model=False,
    cat=False,
):
    loss_list = []
    acc_list = []
    model_list = [model.eval() for model in model_list]
    for idx, (input_, target_, _) in enumerate(data_loader):
        if not complete_model:
            inp1, inp2 = torch.split(input_, [16, 16], dim=3)
        else:
            inp1, inp2 = input_, input_
        inp1, inp2, target_ = inp1.to(device), inp2.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data1 = model_list[0](inp1)
            smashed_data2 = model_list[1](inp2)
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                if replace:
                    smashed_data1[:] = trigger_dic["trigger"].to(device)
                else:
                    smashed_data1[:] += trigger_dic["trigger"].to(device)
            if cat:
                outputs = model_list[2](smashed_data1, smashed_data2)
            else:
                smdata = torch.cat([smashed_data1, smashed_data2], dim=1).to(device)
                outputs = model_list[2](smdata)
            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)

            pred = outputs.max(dim=-1)[-1]
            cur_acc = pred.eq(target_).float().mean()
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


def val_vfl_trigger_in_pic(
    epoch,
    model_list,
    data_loader,
    poison=True,
    explain="",
    log_out=True,
    settings=None,
    cat=True,
):
    loss_list = []
    acc_list = []
    model_list = [model.eval() for model in model_list]

    for idx, (input_, target_, _) in enumerate(data_loader):
        inp1, inp2 = torch.split(input_, [16, 16], dim=3)

        if poison:
            inp1, _ = gen_poison_data("trigger", inp1, target_)
            target_ = torch.zeros(target_.shape).long().to(device=device)

        inp1, inp2, target_ = inp1.to(device), inp2.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data1 = model_list[0](inp1)
            smashed_data2 = model_list[1](inp2)
            if not cat:
                smdata = torch.cat([smashed_data1, smashed_data2], dim=1).to(device)
                outputs = model_list[2](smdata)
            else:
                outputs = model_list[2](smashed_data1, smashed_data2)
            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)

            pred = outputs.max(dim=-1)[-1]
            cur_acc = pred.eq(target_).float().mean()
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
    abl()
