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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from tqdm import tqdm

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
from Utils.utils import val_vfl, val_vfl_badvfl, val_vfl_badvfl_multi, val_vfl_multi_new


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(25262422)


cur_path = Path(__file__).resolve().parent
print(cur_path)
with open(cur_path / "CIFAR10_BadVFL.yaml", "r") as f:
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

aux_dataset = copy.deepcopy(train_dataset)
point = len(train_dataset) // 10
idx = np.random.permutation(len(train_dataset))
pub_idx = idx[point:]
aux_idx = idx[:point]
aux_dataset.data = aux_dataset.data[aux_idx]

aux_dataset.targets = np.array(aux_dataset.targets)[aux_idx]

num_per_class = int(settings["sample_per_class"])
tmp_target_ = [num_per_class for _ in range(class_num)]
idx_ = []
label_dict = [[] for _ in range(class_num)]

for i in range(len(aux_dataset.targets)):
    if tmp_target_[aux_dataset.targets[i]] > 0:
        idx_.append(i)
        tmp_target_[aux_dataset.targets[i]] -= 1

    if sum(tmp_target_) == 0:
        break
if settings["dataset"].lower() == "cinc10":
    aux_dataset.samples = [aux_dataset.samples[i] for i in idx_]

else:
    aux_dataset.data = aux_dataset.data[idx_]
aux_dataset.targets = np.array(aux_dataset.targets)[idx_]
for i in range(len(aux_dataset.targets)):
    label_dict[aux_dataset.targets[i]].append(i)


def create_path_if_not_exists(_path):
    target_path = os.path.join(_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Path {target_path} created.")
    else:
        print(f"Path {target_path} already exists.")


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def mask_test():

    _strategy = settings["strategy"]
    client_number = settings["client_num"]
    assert len(_strategy) == client_number
    cur_path = Path(__file__).resolve().parent.parent
    model_list = [ResNet18(num_classes=10) for _ in range(client_number)]
    print("model_list", len(model_list))
    server_model = TopModelForCifar10WOCat(inputs_length=10 * client_number)
    label_inference_model = TopModelForCifar10WOCat(inputs_length=10)
    model_list.append(server_model)

    model_list = [model.to(device) for model in model_list]

    label_inference_model.to(device)
    target_indices = np.where(np.array(train_dataset.targets) == 0)[0]

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    label_inference_model.to(device)
    label_inference_model.eval()
    server_model.to(device)
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
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    model_list[0].eval()
    embedding_dict = {key: [] for key in range(class_num)}
    epoch = -1
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

    opt_server = Adam(
        model_list[-1].parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )
    shadow_server = None
    loss_f_per = nn.CrossEntropyLoss(reduction="none")
    if settings["unlearn_type"] == "add_label":
        student_model = TopModelForCifar10(class_num=11)
    else:
        student_model = TopModelForCifar10WOCat(
            inputs_length=10 * client_number, class_num=10
        )

    student_model = student_model.to(device)
    opt_student_model = Adam(
        student_model.parameters(),
        lr=settings["Adam"]["lr"],
        weight_decay=settings["Adam"]["weight_decay"],
    )
    p = 0
    trigger = torch.tensor(
        [
            [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
        ]
    )
    for epoch in range(100):
        smdata_score_dict = {"malicious": [], "benign": []}
        p_flag = False
        continue_label_list = []
        model_list = [model.train() for model in model_list]
        fishes = 0
        c_dict_epoch = {}

        if epoch == settings["begin_embeding_swap"][settings["dataset"].lower()]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
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
            gamma = settings["gamma"]
            trigger_dict = {
                "trigger_mask": trigger_mask,
                "window_size": window_size,
                "trigger": trigger,
            }
            print(trigger_dict)

        if epoch < settings["shadow_server_epoch"]:
            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):
                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, _strategy, dim=3)

                # print(inputs.shape)
                intersection_mask = torch.isin(index, torch.tensor(selected_indices))
                mask = torch.where(intersection_mask)[0]
                # mask.to(device)
                # trigger = trigger.to(inp_list[0].dtype)
                # break
                if (
                    len(mask) > 0
                    and epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    # 最后一轮不下毒
                    tmp_batch = len(mask)
                    _source_loader = DataLoader(
                        dataset=source_dataset, batch_size=tmp_batch, shuffle=True
                    )
                    source_inputs, source_targets, _ = next(iter(_source_loader))

                    source_inp_list = torch.split(source_inputs, _strategy, dim=3)

                    source_inp_list[0][
                        :,
                        :,
                        trigger_mask[0] : trigger_mask[0] + window_size[0],
                        trigger_mask[1] : trigger_mask[1] + window_size[1],
                    ] = trigger

                    inp_list[0][mask] = source_inp_list[0]
                smdata_list = [None for _ in range(client_number)]
                targets, index = targets.to(device), index.to(device)
                inp_list = [inp.to(device) for inp in inp_list]
                smashed_data = [None for _ in range(client_number)]
                smashed_data_clone_list = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data[_c_id] = model_list[_c_id](inp_list[_c_id])
                    smashed_data_clone_list[_c_id] = (
                        smashed_data[_c_id].detach().clone()
                    )

                smashed_data_cat = torch.cat(smashed_data, dim=1)
                outputs = model_list[-1](smashed_data_cat)
                opt.zero_grad()
                loss = loss_f(outputs, targets)
                loss.backward()
                opt.step()
                if epoch < settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    _outputs = label_inference_model(smashed_data_clone_list[0])
                    loss = loss_f(_outputs, targets)
                    opt_label_infference_model.zero_grad()
                    loss.backward()
                    model_list[0].zero_grad()
                    opt_label_infference_model.step()
                    # val_label_infferen_model(epoch, model_list[0], label_inference_model, val_loader, "Label Infference Model", _strategy=_strategy)
        else:
            if epoch == settings["shadow_server_epoch"]:
                # shadow_server = TopModelForCifar10(class_num=10)
                shadow_server = copy.deepcopy(model_list[-1])
                shadow_server.to(device)
                # reset_parameters(model_list[2])
                for layer in model_list[-1].children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                val_vfl_multi_new(
                    epoch=epoch,
                    model_list=model_list,
                    data_loader=val_loader,
                    settings=settings,
                    device=device,
                    loss_f=loss_f,
                    myLog=myLog,
                    explain="Reset CDA",
                    split_strategy=_strategy,
                )
                if (
                    epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    val_vfl_badvfl_multi(
                        epoch=epoch,
                        model_list=model_list,
                        data_loader=val_loader,
                        device=device,
                        loss_f=loss_f,
                        myLog=myLog,
                        explain="Reset ASR",
                        poison=True,
                        trigger=trigger_dict,
                        split_strategy=_strategy,
                    )

            assert shadow_server is not None
            shadow_model_list = [model for model in model_list[:-1]]
            shadow_model_list.append(shadow_server)
            opt_shadow_server = Adam(
                [{"params": model.parameters()} for model in shadow_model_list],
                lr=0.001,
                weight_decay=0.0001,
            )
            shadow_server.train()

            for ids, (inputs, targets, index) in enumerate(tqdm(train_loader)):

                # inp1, inp2 = torch.split(inputs, [16, 16], dim=3)
                inp_list = torch.split(inputs, _strategy, dim=3)

                # print(inputs.shape)
                intersection_mask = torch.isin(index, torch.tensor(selected_indices))
                mask = torch.where(intersection_mask)[0]
                # mask.to(device)
                trigger = trigger.to(inp_list[0].dtype)
                # print(targets[mask])
                # break
                if (
                    len(mask) > 0
                    and epoch
                    >= settings["begin_embeding_swap"][settings["dataset"].lower()]
                ):
                    tmp_batch = len(mask)
                    _source_loader = DataLoader(
                        dataset=source_dataset, batch_size=tmp_batch, shuffle=True
                    )
                    source_inputs, source_targets, _ = next(iter(_source_loader))

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
                smashed_data_list = [None for _ in range(client_number)]
                smashed_data_clone_list = [None for _ in range(client_number)]
                smashed_data_clone_list2 = [None for _ in range(client_number)]
                for _c_id in range(client_number):
                    smashed_data_list[_c_id] = model_list[_c_id](inp_list[_c_id])
                    smashed_data_clone_list[_c_id] = (
                        smashed_data_list[_c_id].detach().clone()
                    )
                    smashed_data_clone_list2[_c_id] = (
                        smashed_data_list[_c_id].detach().clone()
                    )

                if ids % settings["detect"] == 0 and settings["thred"] >= 0:
                    unique_targets = torch.unique(targets)
                    vec_dict = {
                        c_id: {t.item(): {} for t in unique_targets}
                        for c_id in range(client_number)
                    }

                    key_dict = {
                        c_id: [
                            torch.empty((0, *smashed_data_list[c_id].shape[1:]))
                            .to(smashed_data_list[c_id].dtype)
                            .to(device)
                            for _ in range(class_num)
                        ]
                        for c_id in range(client_number)
                    }

                    for _c_id in range(client_number):
                        smdata_t = smashed_data_list[_c_id].detach().clone()
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

                smashed_data_cat = torch.cat(smashed_data_list, dim=1)
                smashed_data_cat_clone = smashed_data_cat.detach().clone()
                outputs = shadow_server(smashed_data_cat)
                opt_shadow_server.zero_grad()
                loss_per = loss_f_per(outputs, targets)
                loss_per_clone = loss_per.detach().clone()
                loss = torch.mean(loss_per)
                loss.backward()
                opt_shadow_server.step()

                if epoch < settings["begin_embeding_swap"][settings["dataset"].lower()]:
                    _outputs = label_inference_model(smashed_data_clone_list2[0])
                    loss = loss_f(_outputs, targets)
                    opt_label_infference_model.zero_grad()
                    loss.backward()
                    model_list[0].zero_grad()
                    opt_label_infference_model.step()

                if epoch == settings["shadow_server_epoch"]:
                    _, max_indices = torch.topk(
                        loss_per, k=int(settings["score_k_p_min"] * len(targets) / 2)
                    )

                    try:
                        if epoch < settings["distill_epochs"]:

                            for _index in index[max_indices]:
                                _index = _index.item()
                                if _index in selected_indices:
                                    fishes += 1
                    except Exception as e:
                        print(e)
                        print(max_indices)
                        print(index)
                        return
                    opt_server.zero_grad()
                    # smashed_data_clone_max_indices = [
                    #     smashed_data_clone[max_indices] for smashed_data_clone in smashed_data_clone_list
                    # ]
                    smashed_data_clone_cat = smashed_data_cat_clone[max_indices]
                    outputs_server = server_model(smashed_data_clone_cat)
                    loss_server = loss_f(outputs_server, targets[max_indices])

                    loss_server.backward()
                    opt_server.step()

                    index_for_distill = []

                if epoch == settings["distill_epochs"]:
                    # print(index)
                    p = settings["score_k_p_min"]
                    # smashed_data_clone_list
                    # smdata_a_clone, smdata_b_clone = smdata_a.detach().clone(), smdata_b.detach().clone()
                    with torch.no_grad():
                        smashed_data_clone_cat = torch.cat(
                            smashed_data_clone_list, dim=1
                        )
                        # outputs_shadow = shadow_server(smashed_data_clone_cat)
                        # loss_shadow = loss_f_per(outputs_shadow, targets)
                        loss_shadow = loss_per_clone

                        # outputs_server = server_model(smashed_data_clone_cat)
                        # loss_server = loss_server_clone
                        outputs_server = server_model(smashed_data_clone_cat)
                        loss_server = loss_f_per(outputs_server, targets)
                        # loss_server = loss_f_per(outputs_server, targets)
                        # ---------------------------------- Calculate Score ------------------------------------------

                        Score = loss_shadow / (torch.abs(loss_server - loss_shadow))

                        _, score_large_indices = torch.topk(
                            Score, k=round(p * len(targets))
                        )

                        _, max_indices = torch.topk(
                            loss_per_clone,
                            k=int(settings["score_k_p_min"] * len(targets) / 2),
                        )
                        # score_large_indices = score_large_indices.to(device)
                        score_large_indices = max_indices.to(device)
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
                    with torch.no_grad():
                        smashed_data_clone_cat = torch.cat(
                            smashed_data_clone_list, dim=1
                        )
                        # outputs_shadow = shadow_server(smashed_data_clone_cat)
                        # loss_shadow = loss_f_per(outputs_shadow, targets)
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
                            Score = torch.abs(loss_student - loss_shadow)
                            # print(torch.mean(loss_student), torch.mean(torch.abs(loss_student - loss_shadow)))
                            _, score_large_indices = torch.topk(
                                Score, k=round(p * len(targets)), largest=False
                            )
                            _, score_small_indices = torch.topk(
                                Score,
                                k=round((1 - settings["score_k_p_max"]) * len(targets)),
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
                            # thred_small_indices = torch.tensor(list(range(len(targets)))).to(device)[Score < settings['score_thred']]
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

                    # smashed_data_clone_thred_small_indices = [
                    #     smashed_data_clone[thred_small_indices] for smashed_data_clone in smashed_data_clone_list
                    # ]
                    # smashed_data_clone_thred_shift = torch.cat(smashed_data_clone_thred_small_indices, dim=1)

                    if settings["unlearn_type"] == "add_label":
                        targets_shift = torch.tensor(
                            [class_num for _ in range(len(score_small_indices))],
                            dtype=targets.dtype,
                        ).to(device)

                    elif settings["unlearn_type"] == "label_shift":
                        # 设定K值，表示前K个最大值
                        targets_shift = torch.tensor(
                            [
                                generate_new_number(
                                    num.item(), lower=0, upper=class_num - 1
                                )
                                for num in targets[score_small_indices]
                            ],
                            dtype=targets.dtype,
                        ).to(device)
                    elif settings["unlearn_type"] == "LGA":
                        targets_shift = targets[score_small_indices].to(device)
                    else:
                        myLog.warning("WARNING UNLEARN TYPE ERROR!")
                    # smdata_a_unlabel, smdata_b_unlabel = smdata_a_clone[~choose_tensor], smdata_b_clone[~choose_tensor]
                    targets_score = targets[score_large_indices]

                    # if p < settings["score_k_p_max"] and settings["epochs"] - epoch >= settings["label_shift_epoch"]:
                    if settings["epochs"] - epoch >= settings["label_shift_epoch"]:
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

            # break
        myLog.info(f"c_dict_epoch: {c_dict_epoch}")

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
        myLog.info("fishes {}".format(fishes))
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
            )
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:

                val_vfl_badvfl_multi(
                    epoch=epoch,
                    model_list=model_list[:-1] + [shadow_server],
                    data_loader=val_loader,
                    device=device,
                    loss_f=loss_f,
                    myLog=myLog,
                    explain="SHADOW ASR",
                    poison=True,
                    trigger=trigger_dict,
                    split_strategy=_strategy,
                )
            # val_vfl_badvfl(epoch=epoch, model_list=[client_model1, client_model2, shadow_server], data_loader=val_loader, device=device, loss_f=loss_f, myLog=myLog, explain="SHADOW ASR", poison=True, trigger=trigger_dict)
        if epoch >= settings["distill_epochs"]:
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
            )
            if epoch >= settings["begin_embeding_swap"][settings["dataset"].lower()]:
                val_vfl_badvfl_multi(
                    epoch=epoch,
                    model_list=model_list[:-1] + [student_model],
                    data_loader=val_loader,
                    device=device,
                    loss_f=loss_f,
                    myLog=myLog,
                    explain="student ASR",
                    poison=True,
                    trigger=trigger_dict,
                    split_strategy=_strategy,
                )

                myLog.info("p {}".format(p))
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


def find_mask(grad_mean, window_size=(5, 5)):

    max_mean = float("-inf")
    max_mean_position = None

    for i in range(grad_mean.shape[1] - window_size[0] + 1):
        for j in range(grad_mean.shape[2] - window_size[1] + 1):
            window_mean = torch.mean(
                grad_mean[:, i : i + window_size[0], j : j + window_size[1]]
            )
            if window_mean > max_mean:
                max_mean = window_mean
                max_mean_position = (i, j)
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
