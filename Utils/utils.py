import torch
from Data.AuxiliaryDataset import gen_poison_data



def val_vfl(
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
    cat=True,
    half=False
):
    loss_list = []
    acc_list = []
    model_list = [
        model.eval() for model in model_list
    ]

    for idx, (input_, target_, _) in enumerate(data_loader):
        inp1, inp2 = torch.split(input_, [16, 16], dim=3)

        inp1, inp2, target_ = inp1.to(device), inp2.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data1 = model_list[0](inp1)
            smashed_data2 = model_list[1](inp2)
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                replacement = trigger.to(device=device).to(smashed_data1.dtype)
                if not half:
                    smashed_data1[:] = replacement
                else:
                    mask = (trigger != 0).unsqueeze(0).expand(smashed_data1.shape[0], -1)
                    trigger = trigger.to(smashed_data1.dtype)
                    smashed_data1[mask] = trigger.unsqueeze(0).expand(smashed_data1.shape[0], -1)[mask]
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
    top=1
):
    loss_list = []
    acc_list = []
    model_list = [
        model.eval() for model in model_list
    ]

    for idx, (input_, target_, _) in enumerate(data_loader):
        if not isinstance(input_, list):
            inp_list = torch.split(input_, split_strategy, dim=3)
        else:
            inp_list = input_

        target_ = target_.to(device)
        inp_list = [
            inp.to(device) for inp in inp_list
        ]
        smashed_list = [
            None for _ in range(len(inp_list))
        ]
        with torch.no_grad():
            # smashed_data1 = model_list[0](inp1)
            # smashed_data2 = model_list[1](inp2)
            for _c_id in range(len(smashed_list)):
                smashed_list[_c_id] = model_list[_c_id](inp_list[_c_id])
            if poison:
                target_ = torch.zeros(target_.shape).long().to(device=device)
                replacement = trigger.to(device=device).to(smashed_list[0].dtype)
                if not half:
                    smashed_list[0][:] = replacement
                else:
                    mask = (trigger != 0).unsqueeze(0).expand(smashed_list[0].shape[0], -1)
                    trigger = trigger.to(smashed_list[0].dtype)
                    smashed_list[0][mask] = trigger.unsqueeze(0).expand(smashed_list[0].shape[0], -1)[mask]
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






def val_vfl_multi(
    epoch,
    model_list,
    data_loader,
    poison=False,
    explain="",
    log_out=True,
    settings=None,
    device=None,
    loss_f=None,
    myLog=None
):
    loss_list = []
    acc_list = []
    model_list = [
        model.eval() for model in model_list
    ]

    K = len(model_list) - 1
    W = next(iter(data_loader))[0].shape[-1]

    split_strategy = [
        W//K for _ in range(K)
    ]

    for idx, (input_, target_) in enumerate(data_loader):
        inp_list = torch.split(input_, split_strategy, dim=3)
        target_ = target_.to(device)
        inp_list = [
            inp_.to(device) for inp_ in inp_list
        ]
        with torch.no_grad():
            smdata_list = [
                model_list[i+1](inp_list[i]) for i in range(K)
            ]
            outputs = model_list[0](smdata_list)
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


def val_vfl_badvfl(
    epoch,
    model_list,
    data_loader,
    poison=True,
    explain="",
    log_out=True,
    device=None,
    loss_f=None,
    myLog=None,
    trigger: dict=None,
    cat=True,
):
    assert trigger is not None
    # print(trigger)
    loss_list = []
    acc_list = []
    model_list = [
        model.eval() for model in model_list
    ]

    for idx, (input_, target_, _) in enumerate(data_loader):
        inp1, inp2 = torch.split(input_, [16, 16], dim=3)

        inp1, inp2, target_ = inp1.to(device), inp2.to(device), target_.to(device)
        if poison:
            trigger_mask = trigger["trigger_mask"]
            window_size = trigger["window_size"]
            trigger_ = trigger["trigger"]
            target_ = torch.zeros(target_.shape).long().to(device=device)
            trigger_.to(device=device, dtype=inp1.dtype)
            inp1[:, :, trigger_mask[0]: trigger_mask[0]+window_size[0], trigger_mask[1]: trigger_mask[1]+window_size[1]] = trigger_

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



def val_vfl_badvfl_multi(
    epoch,
    model_list,
    data_loader,
    poison=True,
    explain="",
    log_out=True,
    device=None,
    split_strategy=[16, 16],
    loss_f=None,
    myLog=None,
    trigger: dict=None,
    cat=True,
    targets_label=0,
):
    assert trigger is not None
    # print(trigger)
    loss_list = []
    acc_list = []
    model_list = [
        model.eval() for model in model_list
    ]

    for idx, (input_, target_, _) in enumerate(data_loader):
        if not isinstance(input_, list):
            inp_list = torch.split(input_, split_strategy, dim=3)
        else:
            inp_list = input_
        inp_list = [
            inp.to(device) for inp in inp_list
        ]
        target_ =target_.to(device)
        if poison:
            trigger_mask = trigger["trigger_mask"]
            window_size = trigger["window_size"]
            trigger_ = trigger["trigger"]
            if targets_label == 0:
                target_ = torch.zeros(target_.shape).long().to(device=device)
            elif targets_label == 1:
                target_ = torch.ones(target_.shape).long().to(device=device)
            trigger_.to(device=device, dtype=inp_list[0].dtype)
            inp_list[0][:, :, trigger_mask[0]: trigger_mask[0]+window_size[0], trigger_mask[1]: trigger_mask[1]+window_size[1]] = trigger_

        with torch.no_grad():
            smashed_data_list = [
                model_list[_c_id](inp_list[_c_id]) for _c_id in range(len(split_strategy))
            ]
            # smashed_data_clone_score = torch.cat(smashed_data_list, dim=1)

            smdata = torch.cat(smashed_data_list, dim=1).to(device)
            outputs = model_list[-1](smdata)

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



