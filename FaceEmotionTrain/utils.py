from loss import *
import torch

def build_loss_func(loss_opt, device):

    func = None
    if loss_opt == 'ce':
        func = CELoss(device=device)

    if loss_opt == "ldam":
        func = LDAMLoss(device=device)

    if loss_opt == None:
        raise NotImplementedError(f"{loss_opt} is not implemented yet.")

    return func


def compute_loss(loss_func, pred, label):
    assert pred.get_device() == label.get_device(), \
        "Prediction & label must be in same device"

    loss = loss_func(pred, label)
    return loss


def build_optim(cfg, model):
    optim_name = cfg['train']['optim']
    lr = cfg['train']['lr']
    optim_name = optim_name.lower()

    optim = None
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    if optim_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    if optim_name == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    

    if optim != None:
        return optim
    else:
        raise NotImplementedError(f"{optim_name} is not implemented yet.")


def build_scheduler(cfg, optimizer):
    scheduler_dict = cfg['train']['scheduler']

    sch_name = list(scheduler_dict.keys())[0]
    sch_settings = scheduler_dict[sch_name]
    sch_name = sch_name.lower()

    if sch_name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sch_settings['milestones'], gamma=sch_settings['gamma']
        )
    
    if sch_name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=sch_settings['gamma']
        )

    if sch_name == 'cossineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_settings['T_max'], eta_min=sch_settings['eta_min']
        )

    # Add optimizer if you want

    if sch_name != None:
        return scheduler
    else:
        raise NotImplementedError(f"{sch_name} is not implemented yet.")

def mixup_onehot(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets.type(torch.LongTensor)

def cutmix(data, targets, alpha):
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(data.size()[0])
    targets2 = targets[indices]

    W = data.size()[2]
    H = data.size()[3]
    cut_rat = np.sqrt(1. - lam) # cut ratio
    cut_w = int(W * cut_rat)  # 패치의 너비
    cut_h = int(H * cut_rat)  # 패치의 높이

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    data[:,:, bbx1:bbx2, bby1:bby2] = data[rand_idx,:, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H)) # adjust lambda
    targets = targets * lam + targets2 * (1-lam)

    return data, targets