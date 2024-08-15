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
from Model.model import (TopModelForCifar10, TopModelForCifar10Detector,
                         TopModelForCifar10WOCat, TopModelForCifar10WOCatNew,
                         Vgg16_net)
from Model.ResNet import ResNet18
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
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
with open(cur_path / "optimal_trigger.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("optimal_trigger", parse=settings)
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


def loss_priority_class_loss_score():
    myLog.info("loss_priority_class_loss_score")
    top = settings["top"]
    myLog.info(f"TOP {top}")

    _strategy = settings["split_strategy"]
    client_number = settings["client_num"]
    # server_model = TopModelForCifar10()

    model_list = [ResNet18(num_classes=10) for _ in range(client_number)]
    server_model = TopModelForCifar10WOCat(inputs_length=10 * client_number)
    model_path = (
        Path(__file__).resolve().parent.parent.parent
        / "model_save"
        / "badvfl"
        / f"client_num{client_number}"
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

    target_indices = np.where(np.array(train_dataset.targets) == 0)[0]  # for D_{target}

    selected_indices = np.random.choice(
        target_indices,
        round(settings["poison_rate"] * len(target_indices)),
        replace=False,
    )  # for D_{target}
    target_num = 100

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    loss_f_per = nn.CrossEntropyLoss(reduction="none")

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
    opt_student_model = Adam(
        student_model.parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )
    p = 0
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
        fishes = 0
        smdata_score_dict = {"benign": [], "malicious": []}
        if epoch < settings["shadow_server_epoch"]:
            """
            Normal Training
            """

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
                        # smashed_data[0][intersection_mask] = trigger
                        smashed_data[0][intersection_mask] = trigger + add_noise(
                            trigger, smashed_data[_c_id][0][:20]
                        )
                    else:
                        smashed_data[0][intersection_mask] = trigger

                smashed_data_cat = torch.cat(smashed_data, dim=1)
                output = model_list[-1](smashed_data_cat)
                loss = loss_f(output, targets)
                opt.zero_grad()

                loss.backward()
                opt.step()

        else:
            """
            without low loss
            """

            student_model.train()

            if shadow_server is None:
                shadow_server = copy.deepcopy(model_list[-1])
                shadow_server.to(device)
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
                vec_dict = {c_id: {} for c_id in range(client_number)}
                targets = targets.to(device)
                inp_list = list(torch.split(inputs, _strategy, dim=3))
                intersection_mask = torch.isin(
                    torch.tensor(index), torch.tensor(selected_indices)
                )
                inp_list = [inp.to(device) for inp in inp_list]

                smashed_data = [None for _ in range(client_number)]
                smashed_data_clone_list = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])

                if settings["add_noise"]:
                    smashed_data[0][intersection_mask] = trigger + add_noise(
                        trigger, smashed_data[_c_id][0][:20]
                    )
                else:
                    smashed_data[0][intersection_mask] = trigger

                smashed_data_clone_list = [
                    _smdata.detach().clone() for _smdata in smashed_data
                ]
                if ids % settings["detect"] == 0:
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
                    # detector
                    for _c_id in range(client_number):
                        smdata_t = smashed_data[_c_id].detach().clone()
                        for row in range(len(smdata_t)):

                            t = targets[row].item()
                            class_centers = key_dict[_c_id][t].to(device)
                            if len(class_centers) > 1:
                                dis = torch.norm(class_centers - smdata_t[row], dim=1)
                                key_index = torch.argmin(dis)
                                if dis[key_index] < 0.1:
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
                                if dis < 0.1:
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
                        c_dict[_c_id] = (tmp_key, _max, tmp_index, tmp_targets)

                    # return
                    for _c_id in range(client_number):
                        #     smashed_data[_c_id][c_dict[_c_id][2]] = torch.rand(smashed_data[_c_id][c_dict[_c_id][2]].shape).to(device)
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

                smashed_data_clone_list = [
                    _smdata.detach().clone() for _smdata in smashed_data
                ]
                smashed_data_cat = torch.cat(smashed_data, dim=1)
                smashed_data_cat_clone = smashed_data_cat.detach().clone()
                output = shadow_server(smashed_data_cat)
                loss_per = loos_f_per(output, targets)
                loss_per_clone = loss_per.detach().clone()
                loss_shadowserver = torch.mean(loss_per)
                opt_shadow_server.zero_grad()
                loss_shadowserver.backward()
                opt_shadow_server.step()
                # return

                if epoch == settings["shadow_server_epoch"]:
                    _, max_indices = torch.topk(
                        loss_per, k=int(settings["score_k_p_min"] * len(targets) / 2)
                    )

                    opt_server.zero_grad()

                    smashed_data_clone_cat = smashed_data_cat_clone[max_indices]
                    outputs_server = server_model(smashed_data_clone_cat)
                    loss_server = loss_f(outputs_server, targets[max_indices])

                    loss_server.backward()
                    opt_server.step()

                if epoch == settings["distill_epochs"]:
                    # print(index)
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
                    student_loss = loss_f(outputs_student, targets_score)

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
                    # smdata_a_clone, smdata_b_clone = smdata_a.detach().clone(), smdata_b.detach().clone()
                    with torch.no_grad():
                        smashed_data_clone_cat = torch.cat(
                            smashed_data_clone_list, dim=1
                        )
                        loss_shadow = loss_per_clone
                        outputs_students = student_model(smashed_data_clone_cat)
                        loss_student = loss_f_per(outputs_students, targets)
                        # ---------------------------------- Calculate Score ------------------------------------------
                        assert settings["Score_type"].lower() in [
                            "plus",
                            "div",
                            "multi",
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

                        elif settings["Score_type"].lower() == "multi":
                            Score = loss_student * (
                                torch.abs(loss_student - loss_shadow)
                            )
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets)), largest=False
                            )
                        elif settings["Score_type"].lower() == "sub":
                            # Score = torch.abs(loss_student - loss_shadow)
                            Score = torch.abs(loss_student - loss_shadow)
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
                                largest=False,
                            )

                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets))
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
                    else:
                        myLog.warning("WARNING UNLEARN TYPE ERROR!")
                    # smdata_a_unlabel, smdata_b_unlabel = smdata_a_clone[~choose_tensor], smdata_b_clone[~choose_tensor]
                    targets_score = targets[score_large_indices]

                    if (
                        p < settings["score_k_p_max"]
                        and settings["epochs"] - epoch >= settings["label_shift_epoch"]
                    ):
                        # if settings["epochs"] - epoch >= settings["label_shift_epoch"]:
                        # Normal PCAT
                        opt_student_model.zero_grad()
                        outputs_student = student_model(smashed_data_clone_score)
                        student_loss = loss_f(outputs_student, targets_score)

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

        myLog.info(f"fishes: {fishes}")
        if shadow_server is None:
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
        if shadow_server is not None:
            val_vfl_multi_opt_trigger(
                epoch,
                model_list[:-1] + [shadow_server],
                val_loader,
                poison=False,
                explain="SHADOW CDA",
                split_strategy=_strategy,
                log_out=True,
                top=1,
                trigger=trigger,
            )
            val_vfl_multi_opt_trigger(
                epoch,
                model_list[:-1] + [shadow_server],
                val_loader,
                poison=True,
                explain="SHADOW ASR",
                split_strategy=_strategy,
                log_out=True,
                top=1,
                trigger=trigger,
            )

        if epoch >= settings["distill_epochs"]:
            myLog.info(f"p: {p}")
            val_vfl_multi_opt_trigger(
                epoch,
                model_list[:-1] + [student_model],
                val_loader,
                poison=False,
                explain="student CDA",
                split_strategy=_strategy,
                log_out=True,
                top=1,
                trigger=trigger,
            )
            val_vfl_multi_opt_trigger(
                epoch,
                model_list[:-1] + [student_model],
                val_loader,
                poison=True,
                explain="student ASR",
                split_strategy=_strategy,
                log_out=True,
                top=1,
                trigger=trigger,
            )
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
