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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from tqdm import tqdm

from Data.AuxiliaryDataset import DetectorDataset, gen_poison_data
from Data.cifar10.dataset import IndexedCIFAR10, IndexedCIFAR100
from Log.Logger import Log
from Model.model import (
    TopModelForCifar10,
    TopModelForCifar10Detector,
    TopModelForCifar10WOCat,
    Vgg16_net,
)
from Model.ResNet import ResNet18
from Utils.utils import val_vfl, val_vfl_multi_new


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(252624022)


cur_path = Path(__file__).resolve().parent
print(cur_path)
with open(cur_path / "VILLAIN_Mask_CIFAR10.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("split_smdata_label_shift", parse=settings)
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
elif settings["dataset"].lower() == "cifar100":
    Dataset_use = IndexedCIFAR100
    class_num = 20
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


if settings["dataset"].lower() == "cifar100":
    target_labels = list(range(class_num))
    indices = [
        i for i, (_, label, _) in enumerate(train_dataset) if label in target_labels
    ]
    train_dataset.data = train_dataset.data[indices]
    train_dataset.targets = np.array(train_dataset.targets)[indices]
    # -------------------------- val ------------------------------------
    val_indices = [
        i for i, (_, label, _) in enumerate(val_dataset) if label in target_labels
    ]
    val_dataset.data = val_dataset.data[val_indices]
    val_dataset.targets = np.array(val_dataset.targets)[val_indices]
    print("train_dataset.targets", len(train_dataset.targets))


def loss_priority_class_loss_score():
    myLog.info("loss_priority_class_loss_score")
    top = settings["top"]
    myLog.info(f"TOP {top}")

    trigger_dic = {}

    _strategy = settings["split_strategy"]
    client_number = settings["client_num"]
    # server_model = TopModelForCifar10()
    model_list = [ResNet18(num_classes=class_num) for _ in range(client_number)]
    server_model = TopModelForCifar10WOCat(
        inputs_length=class_num * client_number, class_num=class_num
    )
    # if settings["dataset"].lower() == "cifar100":
    #     server_model = TopModelForCifar100WOCat(inputs_length=class_num*client_number, class_num=class_num)
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
        trigger = torch.tensor([1, -1, 0, 0, 1, -1, 0, 0, 1, -1], device=device)
        myLog.info("trigger_mask {}".format(trigger))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    gamma = settings["gamma"]

    poison_dataset = copy.deepcopy(train_dataset)
    poison_dataset.data = poison_dataset.data[selected_indices]
    poison_dataset.targets = np.array(poison_dataset.targets)[selected_indices]
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=True)

    loss_dict = {i: {} for i in range(class_num)}
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
    opt_student_model = Adam(
        student_model.parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )
    for epoch in range(settings["epochs"]):
        max_label_indict = {}
        model_list = [model.train() for model in model_list]

        fishes = 0
        smdata_score_dict = {"benign": [], "malicious": []}
        if epoch < settings["shadow_server_epoch"]:
            """
            Normal Training
            """
            c_dict_epoch = {}
            if settings["villain"]:
                if (
                    epoch
                    == settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
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
                mask = torch.tensor(mask, device=device)
                # ---------------------------------------
                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    if settings["trigger_mask"]:
                        replacement = torch.zeros_like(smashed_data[0]).to(
                            device=device
                        )
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
                smashed_data_clone_list = [
                    smdata.detach().clone() for smdata in smashed_data
                ]

                # smashed_data_cat = torch.cat(smashed_data, dim=1)
                smashed_data_cat = torch.cat(smashed_data, dim=1)
                # print(smashed_data_cat.shape)
                outputs = model_list[-1](smashed_data_cat)
                # print(outputs.shape)
                loss_per = loos_f_per(outputs, targets)

                loss = torch.mean(loss_per)
                opt.zero_grad()
                loss.backward()
                opt.step()
                for i in range(len(index)):
                    loss_dict[targets[i].item()][index[i].item()] = loss_per[i].item()

        else:
            """
            without low loss
            """
            if settings["villain"]:
                if (
                    epoch
                    == settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    # ------------------ trigger design ---------------------------------
                    column_list = []
                    for ids, (inputs, targets, index) in enumerate(poison_loader):

                        inp_list = torch.split(inputs, _strategy, dim=3)
                        targets = targets.to(device)
                        intersection_mask = torch.isin(
                            torch.tensor(index), torch.tensor(selected_indices)
                        )
                        mask = torch.where(intersection_mask)[0]

                        inp_list = [inp.to(device) for inp in inp_list]
                        # smashed_data = [None for _ in range(client_number)]
                        # for _c_id in range(client_number):
                        #     smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])

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

            student_model.train()
            class_length = len(train_dataset) // class_num
            # class_top = round(class_length * settings["distill_p"])
            total_remain_epoch = settings["epochs"] - epoch
            distill_epoch = settings["epochs"] - settings["distill_epochs"]
            # if distill_epoch > 0:
            #     class_top = round(class_length * (total_remain_epoch / distill_epoch) * settings["distill_p"])
            # else:
            #     class_top = round(class_length * settings["distill_p"])

            if shadow_server is None:
                shadow_server = copy.deepcopy(model_list[-1])
                shadow_server.to(device)
                # shadow_server.load_state_dict(server_model.state_dict())
                for layer in model_list[-1].children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            assert shadow_server is not None
            shadow_server_list = [model for model in model_list[:-1]]
            shadow_server_list.append(shadow_server)
            shadow_server_list = [model.train() for model in shadow_server_list]
            opt_shadow_server = Adam(
                [{"params": model.parameters()} for model in shadow_server_list],
                lr=settings["Adam"]["lr"],
                weight_decay=settings["Adam"]["weight_decay"],
            )
            opt_server = Adam(
                server_model.parameters(),
                lr=settings["Adam"]["lr"],
                weight_decay=settings["Adam"]["weight_decay"],
            )
            p_flag = False
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):

                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, _strategy, dim=3)
                targets = targets.to(device)
                inp_list = [inp.to(device) for inp in inp_list]
                intersection_mask = torch.isin(index, torch.tensor(selected_indices))
                mask = torch.where(intersection_mask)[0]
                smashed_data = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])
                    smashed_data_clone_list[_c_id] = (
                        smashed_data[_c_id].detach().clone()
                    )

                mask_benign = torch.ones(len(targets))
                mask_benign[mask] = 0
                mask_benign = mask_benign == 1
                mask = mask.to(device=device)
                # ---------------------------------------
                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    if settings["trigger_mask"]:
                        replacement = torch.zeros_like(smashed_data[0]).to(
                            device=device
                        )
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
                # ---------------------------------------
                # smdata_a_clone, smdata_b_clone = smdata_a.detach().clone(), smdata_b.detach().clone()
                # smdata_clone_list = [
                #     smdata.detach().clone() for smdata in smashed_data
                # ]

                if ids % settings["detect"] == 0 and settings["thred"] >= 0:
                    unique_targets = torch.unique(targets)
                    vec_dict = {
                        c_id: {t.item(): {} for t in unique_targets}
                        for c_id in range(client_number)
                    }

                    key_dict = {
                        c_id: [
                            torch.empty((0, *smashed_data[c_id].shape[1:]))
                            .to(smashed_data[c_id].dtype)
                            .to(device)
                            for _ in range(class_num)
                        ]
                        for c_id in range(client_number)
                    }

                    for _c_id in range(client_number):
                        smdata_t = smashed_data[_c_id].detach().clone()
                        for row in range(len(smdata_t)):

                            t = targets[row].item()
                            class_centers = key_dict[_c_id][t].to(device)
                            if len(class_centers) > 1:
                                dis = torch.norm(class_centers - smdata_t[row], dim=1)
                                key_index = torch.argmin(dis)
                                if dis[key_index] < settings["thred"]:
                                    key = tuple(
                                        class_centers[key_index]
                                        .clone()
                                        .detach()
                                        .tolist()
                                    )
                                    vec_dict[_c_id][t][key][0] += 1
                                    vec_dict[_c_id][t][key][1].append(row)
                                else:
                                    vec_dict[_c_id][t][
                                        tuple(smdata_t[row].clone().detach().tolist())
                                    ] = [1, [row]]
                                    key_dict[_c_id][t] = torch.cat(
                                        [
                                            key_dict[_c_id][t],
                                            smdata_t[row].clone().detach().unsqueeze(0),
                                        ],
                                        dim=0,
                                    )
                            elif len(class_centers) == 1:
                                dis = torch.norm(class_centers - smdata_t[row])
                                key_index = torch.argmin(dis)
                                if dis < settings["thred"]:
                                    key = tuple(
                                        class_centers[key_index]
                                        .clone()
                                        .detach()
                                        .tolist()
                                    )
                                    vec_dict[_c_id][t][key][0] += 1
                                    vec_dict[_c_id][t][key][1].append(row)
                                else:
                                    vec_dict[_c_id][t][
                                        tuple(smdata_t[row].clone().detach().tolist())
                                    ] = [1, [row]]
                                    key_dict[_c_id][t] = torch.cat(
                                        [
                                            key_dict[_c_id][t],
                                            smdata_t[row].clone().detach().unsqueeze(0),
                                        ],
                                        dim=0,
                                    )
                            else:
                                vec_dict[_c_id][t][
                                    tuple(smdata_t[row].clone().detach().tolist())
                                ] = [1, [row]]
                                key_dict[_c_id][t] = (
                                    smdata_t[row].clone().detach().unsqueeze(0)
                                )

                    c_dict = {c_id: None for c_id in range(client_number)}
                    for _c_id in range(client_number):

                        _max = -1
                        tmp_key = None
                        tmp_index = None
                        tmp_targets = None
                        for t in unique_targets:
                            for key in vec_dict[_c_id][t.item()].keys():
                                if vec_dict[_c_id][t.item()][key][0] > _max:
                                    _max = vec_dict[_c_id][t.item()][key][0]
                                    tmp_key = key
                                    tmp_index = vec_dict[_c_id][t.item()][key][1]
                                    tmp_targets = t.item()

                        c_dict[_c_id] = (tmp_key, _max, tmp_index, tmp_targets)

                    # return
                    for _c_id in range(client_number):
                        if len(c_dict[_c_id][2]) > 1:
                            targets[c_dict[_c_id][2]] = torch.tensor(
                                [
                                    generate_new_number(
                                        num.item(), lower=0, upper=class_num - 1
                                    )
                                    for num in targets[c_dict[_c_id][2]]
                                ],
                                dtype=targets.dtype,
                            ).to(device)
                            if c_dict[_c_id][3] not in c_dict_epoch:
                                c_dict_epoch[c_dict[_c_id][3]] = 0
                            c_dict_epoch[c_dict[_c_id][3]] += 1

                smashed_data_cat = torch.cat(smashed_data, dim=1)
                smashed_data_cat_clone = smashed_data_cat.detach().clone()
                output = shadow_server(smashed_data_cat)
                loss_per = loos_f_per(output, targets)
                loss_per_clone = loss_per.detach().clone()
                loss_shadowserver = torch.mean(loss_per)
                opt_shadow_server.zero_grad()
                loss_shadowserver.backward()
                opt_shadow_server.step()

                for i in range(len(index)):
                    loss_dict[targets[i].item()][index[i].item()] = loss_per[i].item()

                if epoch == settings["shadow_server_epoch"]:
                    _, max_indices = torch.topk(
                        loss_per, k=int(settings["score_k_p_min"] * len(targets))
                    )

                    opt_server.zero_grad()
                    smashed_data_clone_cat = smashed_data_cat_clone[max_indices]
                    outputs_server = server_model(smashed_data_clone_cat)
                    loss_server = loss_f(outputs_server, targets[max_indices])

                    loss_server.backward()
                    opt_server.step()

                    index_for_distill = []

                if epoch == settings["distill_epochs"]:
                    index = index.to(device)
                    p = settings["score_k_p_min"]

                    with torch.no_grad():
                        smashed_data_clone_cat = torch.cat(
                            smashed_data_clone_list, dim=1
                        )
                        # outputs_shadow = shadow_server(smashed_data_clone_cat)
                        loss_shadow = loss_per_clone
                        outputs_server = server_model(smashed_data_clone_cat)
                        loss_server = loss_f_per(outputs_server, targets)
                        # ---------------------------------- Calculate Score ------------------------------------------

                        Score = loss_shadow / (torch.abs(loss_server - loss_shadow))
                        _, score_large_indices = torch.topk(
                            Score, k=round(p * len(targets))
                        )
                        score_large_indices = score_large_indices.to(device)
                        # smdata_score_dict

                        for _i in range(len(index)):
                            _index = index[_i].item()
                            if _index in selected_indices:
                                smdata_score_dict["malicious"].append(Score[_i].item())
                            else:
                                smdata_score_dict["benign"].append(Score[_i].item())

                        for _index in index[score_large_indices]:
                            _index = _index.item()
                            if _index in selected_indices:
                                fishes += 1
                    _, max_indices = torch.topk(
                        loss_per, k=int(settings["score_k_p_min"] * len(targets))
                    )
                    # _, max_indices = torch.topk(loss_per, k=int(settings["score_k_p_min"] * len(targets) / 2))
                    score_large_indices = max_indices.to(device)
                    smashed_data_clone_score_large_indices = [
                        smashed_data_clone[score_large_indices]
                        for smashed_data_clone in smashed_data_clone_list
                    ]
                    smashed_data_clone_score = torch.cat(
                        smashed_data_clone_score_large_indices, dim=1
                    )
                    # smdata_a_score, smdata_b_score = smdata_a_clone[score_large_indices], smdata_b_clone[score_large_indices]
                    targets_score = targets[score_large_indices]
                    opt_student_model.zero_grad()
                    outputs_student = student_model(smashed_data_clone_score)
                    # 计算KL散度
                    student_loss = loss_f(outputs_student, targets_score)

                    # 使用KL Loss来更新student model
                    student_loss.backward()
                    opt_student_model.step()

                if epoch > settings["distill_epochs"]:

                    if not p_flag:
                        p = min(
                            settings["score_k_p_max"],
                            p + (1 - p) * settings["score_k_p_min"],
                        )
                        p_flag = True
                    # smashed_data_clone
                    index = index.to(device)
                    with torch.no_grad():
                        smashed_data_clone_cat = torch.cat(
                            smashed_data_clone_list, dim=1
                        )
                        # outputs_shadow = shadow_server(smashed_data_clone_cat)
                        # loss_shadow = loss_f_per(outputs_shadow, targets)
                        loss_shadow = loss_per_clone
                        # student_model.train()
                        outputs_students = student_model(smashed_data_clone_cat)
                        loss_student = loss_f_per(outputs_students, targets)
                        # student_model.zero_grad()
                        opt_student_model.zero_grad()
                        # ---------------------------------- Calculate Score ------------------------------------------
                        assert settings["Score_type"].lower() in [
                            "plus",
                            "div",
                            "sub2",
                            "sub",
                            "original",
                        ]
                        if settings["Score_type"].lower() == "plus":
                            Score = loss_student + (
                                torch.abs(loss_student - loss_shadow)
                            )
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets)), largest=False
                            )
                        elif settings["Score_type"].lower() == "div":
                            Score = loss_shadow / (
                                torch.abs(loss_student - loss_shadow)
                            )
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets))
                            )
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
                                largest=False,
                            )
                        elif settings["Score_type"].lower() == "sub2":
                            Score = loss_shadow - torch.abs(loss_student - loss_shadow)
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets))
                            )
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
                                largest=False,
                            )
                        elif settings["Score_type"].lower() == "sub":
                            # Score = torch.abs(loss_student - loss_shadow)
                            Score = torch.abs(loss_student - loss_shadow)
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets))
                            )
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
                                largest=False,
                            )

                        elif settings["Score_type"].lower() == "original":
                            Score = loss_shadow
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets))
                            )
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
                                largest=False,
                            )
                        # Score = loss_shadow / (torch.abs(loss_student - loss_shadow))

                        score_large_indices = score_large_indices.to(device)

                        for _index in index[score_large_indices]:
                            _index = _index.item()
                            if _index in selected_indices:
                                fishes += 1

                        for _i in range(len(index)):
                            _index = index[_i].item()
                            if _index in selected_indices:
                                smdata_score_dict["malicious"].append(Score[_i].item())
                            else:
                                smdata_score_dict["benign"].append(Score[_i].item())

                    choose_tensor = torch.zeros(len(targets), dtype=bool)
                    choose_tensor[score_large_indices] = True

                    smashed_data_clone_score_large_indices = [
                        smashed_data_clone[score_large_indices]
                        for smashed_data_clone in smashed_data_clone_list
                    ]
                    smashed_data_clone_score = torch.cat(
                        smashed_data_clone_score_large_indices, dim=1
                    )
                    smashed_data_clone_score_small_indices = [
                        smashed_data_clone[score_small_indices]
                        for smashed_data_clone in smashed_data_clone_list
                    ]

                    smashed_data_clone_shift = torch.cat(
                        smashed_data_clone_score_small_indices, dim=1
                    )

                    if settings["unlearn_type"] == "add_label":
                        targets_shift = torch.tensor(
                            [class_num for _ in range(len(score_small_indices))],
                            dtype=targets.dtype,
                        ).to(device)

                    elif settings["unlearn_type"] == "label_shift":
                        targets_shift = torch.tensor(
                            [
                                generate_new_number(
                                    num.item(), lower=0, upper=class_num - 1
                                )
                                for num in targets[score_small_indices]
                            ],
                            dtype=targets.dtype,
                        ).to(device)
                        # targets_shift = (targets[score_small_indices] + 1) % class_num
                    else:
                        myLog.warning("WARNING UNLEARN TYPE ERROR!")
                    targets_score = targets[score_large_indices]

                    if settings["epochs"] - epoch >= settings["label_shift_epoch"]:
                        # Normal PCAT
                        opt_student_model.zero_grad()
                        outputs_student = student_model(smashed_data_clone_score)
                        # 计算KL散度
                        student_loss = loss_f(outputs_student, targets_score)

                        # 使用KL Loss来更新student model
                        student_loss.backward()
                        opt_student_model.step()

                    else:
                        # Label Shift
                        opt_student_model.zero_grad()
                        smdata_score = torch.cat(
                            [smashed_data_clone_score, smashed_data_clone_shift], dim=0
                        ).to(device)
                        # smdata_b_score = torch.cat([smdata_b_score, smdata_b_shift], dim=0).to(device)
                        targets_score = torch.cat(
                            [targets_score, targets_shift], dim=0
                        ).to(device)
                        outputs_student = student_model(smdata_score)
                        student_loss = loss_f(outputs_student, targets_score)

                        student_loss.backward()
                        opt_student_model.step()

        if epoch >= settings["shadow_server_epoch"] and p_flag:
            myLog.info("p {}".format(p))

        # val_vfl(epoch=epoch, model_list=model_list, data_loader=val_loader, settings=settings, device=device, loss_f=loss_f, myLog=myLog, explain="CDA")
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
        # val_vfl_badvfl_multi(epoch=epoch, model_list=model_list, data_loader=val_loader, device=device, loss_f=loss_f, myLog=myLog, explain="ASR", poison=True, trigger=trigger_dict, split_strategy=_strategy)
        if shadow_server is not None:
            val_vfl_multi_new(
                epoch=epoch,
                model_list=model_list[:-1] + [shadow_server],
                data_loader=val_loader,
                settings=settings,
                device=device,
                loss_f=loss_f,
                myLog=myLog,
                explain="SHADOW CDA",
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
                if shadow_server is not None:
                    val_vfl_villain_multi(
                        epoch=epoch,
                        model_list=model_list[:-1] + [shadow_server],
                        data_loader=val_loader,
                        settings=settings,
                        device=device,
                        loss_f=loss_f,
                        myLog=myLog,
                        explain="SHADOW ASR",
                        poison=True,
                        trigger_dic=trigger_dic,
                        cat=True,
                        split_strategy=_strategy,
                        top=top,
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
                    split_strategy=_strategy,
                    top=top,
                )
                if shadow_server is not None:
                    val_vfl_half_multi(
                        epoch=epoch,
                        model_list=model_list[:-1] + [shadow_server],
                        data_loader=val_loader,
                        settings=settings,
                        device=device,
                        loss_f=loss_f,
                        myLog=myLog,
                        explain="SHADOW ASR",
                        poison=True,
                        trigger=trigger,
                        split_strategy=_strategy,
                        top=top,
                    )

        if epoch >= settings["distill_epochs"]:
            # myLog.info(f"max_label_indict: {max_label_indict}")
            val_vfl_multi_new(
                epoch=epoch,
                model_list=model_list[:-1] + [student_model],
                data_loader=val_loader,
                settings=settings,
                device=device,
                loss_f=loss_f,
                myLog=myLog,
                explain="student CDA",
                split_strategy=_strategy,
                top=top,
            )
            # val_vfl(epoch=epoch, model_list=[client_model1, client_model2, student_model], data_loader=val_loader, settings=settings, device=device, loss_f=loss_f, myLog=myLog, explain="student ASR", poison=True, trigger=trigger, half=True)
            if (
                len(smdata_score_dict["benign"]) > 0
                and len(smdata_score_dict["malicious"]) > 0
            ):
                myLog.info(
                    "smdata score dict - Benign: {}, Malicious: {}, Score_type: {}".format(
                        sum(smdata_score_dict["benign"])
                        / len(smdata_score_dict["benign"]),
                        sum(smdata_score_dict["malicious"])
                        / len(smdata_score_dict["malicious"]),
                        settings["Score_type"],
                    )
                )
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
                if settings["villain"]:
                    val_vfl_villain_multi(
                        epoch=epoch,
                        model_list=model_list[:-1] + [student_model],
                        data_loader=val_loader,
                        settings=settings,
                        device=device,
                        loss_f=loss_f,
                        myLog=myLog,
                        explain="student ASR",
                        poison=True,
                        trigger_dic=trigger_dic,
                        cat=True,
                        split_strategy=_strategy,
                        top=top,
                    )
                else:
                    val_vfl_half_multi(
                        epoch=epoch,
                        model_list=model_list[:-1] + [student_model],
                        data_loader=val_loader,
                        settings=settings,
                        device=device,
                        loss_f=loss_f,
                        myLog=myLog,
                        explain="student ASR",
                        poison=True,
                        trigger=trigger,
                        split_strategy=_strategy,
                        top=top,
                    )


def generate_new_number(exclude, lower=0, upper=9, other=-1):
    number = torch.randint(lower, upper + 1, (1,)).item()
    while number == exclude or number == other:
        number = torch.randint(lower, upper + 1, (1,)).item()
    return number


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
    loss_priority_class_loss_score()
