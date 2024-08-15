import copy
import os
import random
from collections import Counter
from pathlib import Path

import attack.optimal_trigger.cal_centers as cc
import attack.optimal_trigger.search_vec as sv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from Data.AuxiliaryDataset import gen_poison_dataset_replace_attack
from Data.cifar10.dataset import IndexedCIFAR10, IndexedCIFAR100
from Log.Logger import Log
from Model.model import (
    TopModelForCifar10,
    TopModelForCifar10Detector,
    TopModelForCifar10WOCat,
    TopModelForCifar10WOCatNew,
    Vgg16_net,
)
from Model.ResNet import ResNet18
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from tqdm import tqdm
from Utils.utils import val_vfl, val_vfl_multi_new


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


cur_path = Path(__file__).resolve().parent
print("cur_path", cur_path)
with open(cur_path / "optimal_attack_abl.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("optimal_attack_abl", parse=settings)
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
print("class_num", class_num)
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
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent


def loss_priority_class_loss_score():
    myLog.info("loss_priority_class_loss_score")
    top = settings["top"]
    myLog.info(f"TOP {top}")

    _strategy = settings["split_strategy"]
    client_number = settings["client_num"]

    model_list = [ResNet18(num_classes=10) for _ in range(client_number)]
    server_model = TopModelForCifar10WOCat(inputs_length=10 * client_number)

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

    target_indices = np.where(np.array(train_dataset.targets) == 0)[0]  # for D_{target}

    selected_indices = np.random.choice(
        target_indices,
        round(settings["poison_rate"] * len(target_indices)),
        replace=False,
    )  # for D_{target}
    target_num = 100
    gamma = settings["gamma"]

    lga_loss_function = LGALoss(gamma, loss_f)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    epoch = -1
    steal_set = copy.deepcopy(train_dataset)
    steal_set.data = steal_set.data[selected_indices]
    steal_set.targets = np.array(steal_set.targets)[selected_indices]

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
    vec_arr = design_vec(
        class_num, model_list[0], 0, steal_set, target_num, split_strategy=_strategy
    )
    print("vec_arr", vec_arr)
    trigger = torch.tensor(vec_arr).to(device)
    # return

    opt_server = Adam(
        model_list[-1].parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )
    shadow_server = None
    # student_model = copy.deepcopy(server_model)
    if settings["unlearn_type"] == "add_label":
        student_model = TopModelForCifar10(class_num=11)
    else:
        student_model = TopModelForCifar10WOCat(
            class_num=class_num, inputs_length=class_num * client_number
        )
    # student_model = TopModelForCifar10()
    student_model = student_model.to(device)

    loss_sorted_dict = {}

    for epoch in range(settings["epochs"]):
        if epoch == settings["begin_embeding_swap"][settings["dataset"].lower()]:
            vec_arr = design_vec(
                class_num,
                model_list[0],
                0,
                steal_set,
                target_num,
                split_strategy=_strategy,
            )
            print("vec_arr", vec_arr)
            trigger = torch.tensor(vec_arr).to(device)
        model_list = [model.train() for model in model_list]
        if epoch < settings["abl_epochs"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                # inp_list = torch.split(inputs, _strategy, dim=3)

                targets = targets.to(device)
                inp_list = list(torch.split(inputs, _strategy, dim=3))
                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                inp_list = [inp.to(device) for inp in inp_list]

                smashed_data = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])
                if epoch > settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    if settings["add_noise"]:
                        smashed_data[0][intersection_mask] = trigger + add_noise(
                            trigger, smashed_data[_c_id][0][:20]
                        )
                    else:
                        smashed_data[0][intersection_mask] = trigger

                smashed_data_cat = torch.cat(smashed_data, dim=1)
                output = server_model(smashed_data_cat)
                loss = lga_loss_function(output, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

        elif epoch == settings["abl_epochs"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):

                inp_list = list(torch.split(inputs, _strategy, dim=3))
                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                mask = torch.where(intersection_mask)[0]
                mask = torch.tensor(mask, device=device)

                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]
                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])

                if epoch > settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    if settings["add_noise"]:
                        smdata_list[0][intersection_mask] = trigger + add_noise(
                            trigger, smdata_list[_c_id][0][:20]
                        )
                    else:
                        smdata_list[0][intersection_mask] = trigger
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
                inp_list = torch.split(inputs, _strategy, dim=3)

                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                mask = torch.where(intersection_mask)[0]

                mask = torch.tensor(mask, device=device)

                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]

                smdata_list = [None for _ in range(client_number)]
                for c_id in range(client_number):
                    smdata_list[c_id] = model_list[c_id](inp_list[c_id])
                if epoch > settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    if settings["add_noise"]:
                        smdata_list[0][intersection_mask] = trigger + add_noise(
                            trigger, smdata_list[_c_id][0][:20]
                        )
                    else:
                        smdata_list[0][intersection_mask] = trigger
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

        val_vfl_multi_opt_trigger(
            epoch,
            model_list,
            val_loader,
            poison=False,
            explain="CDA",
            split_strategy=_strategy,
            log_out=True,
            top=1,
            trigger=trigger,
        )
        val_vfl_multi_opt_trigger(
            epoch,
            model_list,
            val_loader,
            poison=True,
            explain="ASR",
            split_strategy=_strategy,
            log_out=True,
            top=1,
            trigger=trigger,
        )


def generate_new_number(exclude, lower=0, upper=9):
    number = torch.randint(lower, upper + 1, (1,)).item()
    while number == exclude:
        number = torch.randint(lower, upper + 1, (1,)).item()
    return number


def generate_vec(model, dataloader, unit, bottom_series, split_strategy=[8, 8, 8, 8]):
    vecs = []
    for i, (x_in, y_in, _) in enumerate(dataloader):
        x_in = x_in.to(device)
        # print("x_in.shape", x_in.shape)
        if unit != 1:
            x_part = x_in.split(split_strategy, dim=3)
            pred = model(x_part[bottom_series]).detach().cpu().numpy()
        else:
            pred = model(x_in).detach().cpu().numpy()
        # print("pred.shape", pred.shape)
        vecs.append(pred)
        print(pred.shape)
    vecs = np.concatenate(vecs, axis=0)
    shape = vecs.shape
    print("vecs.shape", vecs.shape)
    return vecs


def generate_target_clean_vecs(
    model, testset, unit, bottom_series=0, split_strategy=[8, 8, 8, 8]
):
    target_set = testset
    targetloader = torch.utils.data.DataLoader(
        target_set, batch_size=1000, shuffle=True
    )
    target_clean_vecs = generate_vec(
        model, targetloader, unit, bottom_series, split_strategy
    )
    return target_clean_vecs


def design_vec(
    class_num, model, label, steal_set, target_num, split_strategy=[8, 8, 8, 8]
):
    target_clean_vecs = generate_target_clean_vecs(
        model, steal_set, 0.5, bottom_series=0, split_strategy=split_strategy
    )

    dim = filter_dim(target_clean_vecs, target_num)

    center = cc.cal_target_center(target_clean_vecs[dim].copy(), kernel_bandwidth=1000)
    print("center.shape", center.shape)
    target_vec = sv.search_vec(center, target_clean_vecs, 0.5)

    target_vec = target_vec.view()

    return target_vec


def add_noise(vec, normal_vecs):
    avg_value = torch.mean(normal_vecs, dim=0).reshape((-1))
    con = torch.where(avg_value < 0.001)[0]

    size = vec.size()
    vec = vec.reshape((-1))

    vec = vec.clamp_(0, 2.5)
    vec *= 1.15

    gauss_noise_big = torch.normal(mean=0, std=0.2, size=vec.size()).to(device=device)
    gauss_noise_small = torch.normal(mean=0, std=0.05, size=vec.size()).to(
        device=device
    )

    condition = torch.randn(vec.size()).to(device=device)
    zeros = torch.zeros_like(vec).to(device=device)
    replace = torch.where(condition < 0.8, zeros, vec + gauss_noise_small)
    vec = torch.where(vec < 0.4, replace, vec + gauss_noise_big)
    vec = vec.clamp_(0).reshape((size[0], -1))
    vec[:, con] = 0

    return vec


def filter_dim(vecs, target_num):
    coef = np.corrcoef(vecs)
    rows = np.sum(coef, axis=1)
    selected = np.argpartition(rows, -target_num)[-target_num:]
    print("np.mean(np.corrcoef(vecs[selected]))", np.mean(np.corrcoef(vecs[selected])))
    return selected


def val_vfl_multi_opt_trigger(
    epoch,
    model_list,
    data_loader,
    poison=False,
    explain="",
    split_strategy=[16, 16],
    log_out=True,
    top=1,
    trigger=None,
):
    loss_list = []
    acc_list = []
    # print("model_list", len(model_list))

    model_list = [model.eval() for model in model_list]
    # print("model_list", len(model_list))

    for idx, (input_, target_, _) in enumerate(data_loader):
        if not isinstance(input_, list):
            inp_list = list(torch.split(input_, split_strategy, dim=3))
        else:
            inp_list = input_

        target_ = target_.to(device)

        smashed_list = [None for _ in range(len(inp_list))]
        with torch.no_grad():
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                # inp_list[0] = gen_poison_data_replace_attack(inp_list[0])

            inp_list = [inp.to(device) for inp in inp_list]
            # print("len(inp_list)",len(inp_list))
            for _c_id in range(len(smashed_list)):
                # print(inp_list[_c_id].shape)
                smashed_list[_c_id] = model_list[_c_id](inp_list[_c_id])
            if poison:
                smashed_list[0][:] = trigger
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

    loss_priority_class_loss_score()
