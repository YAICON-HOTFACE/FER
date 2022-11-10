from loss import *
import torch

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import pdb

import os
import cv2
import random


def face_synthesis(image, landmark, flag=1, patch_path = "/data/FaceEmotion/AFFNet/"):
    """
    Visually synthesize the patch image with given face landmark.

    'flag': 1 sunglasses, 2 mask, 3 mask+sunglasses, 0 random synthesis,
            otherwise returns the copy of original image.
    """

    lnd = landmark.reshape(-1, 2)
    dst = image.copy()

    sun_lnd = np.float32([[350, 20], [345, 300], [755, 225], [1160, 20], [1165, 300]])
    eye_lnd = np.float32([lnd[19], lnd[41], lnd[27], lnd[24], lnd[46]])
    mask_lnd = np.float32([[200, 380], [230, 570], [350, 690], [500, 715], 
                           [650, 690], [770, 570], [800, 380], [500, 330]])
    mouth_lnd = np.float32([lnd[2], lnd[4], lnd[6], lnd[8],
                            lnd[10], lnd[12], lnd[14], lnd[30]])

    if flag == 0:
        flag = random.randrange(1, 4)

    if flag == 1:   # sunglasses
        sun = cv2.imread(os.path.join(patch_path, "sunglasses"+str(random.randrange(1,6))+".png"), cv2.IMREAD_UNCHANGED)
        M, _ = cv2.findHomography(sun_lnd, eye_lnd)
        sun = cv2.warpPerspective(sun, M, (224, 224))
        a = sun[:,:,3] / 255.0

        sun_brightness = get_avg_brightness(sun)
        img_brightness = get_avg_brightness(image)
        delta_b = 1 + (img_brightness - sun_brightness) / 255
        sun = change_brightness(sun, delta_b)
        sun_saturation = get_avg_saturation(sun)
        img_saturation = get_avg_saturation(image)
        delta_s = 1 + (img_saturation - sun_saturation) / 255
        sun = change_saturation(sun, delta_s)

        for c in range(0, 3):
            dst[:,:,c] = a * sun[:,:,c] + dst[:,:,c] * (1-a)

    elif flag == 2: # mask
        mask = cv2.imread(os.path.join(patch_path, "mask.png"), cv2.IMREAD_UNCHANGED)
        M, _ = cv2.findHomography(mask_lnd, mouth_lnd)
        mask = cv2.warpPerspective(mask, M, (224, 224))
        a = mask[:,:,3] / 255.0

        mask_brightness = get_avg_brightness(mask)
        img_brightness = get_avg_brightness(image)
        delta_b = 1 + (img_brightness - mask_brightness) / 255
        mask = change_brightness(mask, delta_b)
        mask_saturation = get_avg_saturation(mask)
        img_saturation = get_avg_saturation(image)
        delta_s = 1 + (img_saturation - mask_saturation) / 255
        mask = change_saturation(mask, delta_s)

        for c in range(0, 3):
            dst[:,:,c] = a * mask[:,:,c] + dst[:,:,c] * (1-a)

    elif flag == 3: # mask with sunglasses
        mask = cv2.imread(os.path.join(patch_path, "mask.png"), cv2.IMREAD_UNCHANGED)
        M, _ = cv2.findHomography(mask_lnd, mouth_lnd)
        mask = cv2.warpPerspective(mask, M, (224, 224))
        a = mask[:,:,3] / 255.0
        
        img_brightness = get_avg_brightness(image)
        img_saturation = get_avg_saturation(image)

        mask_brightness = get_avg_brightness(mask)
        delta_b = 1 + (img_brightness - mask_brightness) / 255
        mask = change_brightness(mask, delta_b)
        mask_saturation = get_avg_saturation(mask)
        delta_s = 1 + (img_saturation - mask_saturation) / 255
        mask = change_saturation(mask, delta_s)

        for c in range(0, 3):
            dst[:,:,c] = a * mask[:,:,c] + dst[:,:,c] * (1-a)
        
        sun = cv2.imread(os.path.join(patch_path, "sunglasses"+str(random.randrange(1,6))+".png"), cv2.IMREAD_UNCHANGED)
        M, _ = cv2.findHomography(sun_lnd, eye_lnd)
        sun = cv2.warpPerspective(sun, M, (224, 224))
        a = sun[:,:,3] / 255.0

        sun_brightness = get_avg_brightness(sun)
        delta_b = 1 + (img_brightness - sun_brightness) / 255
        sun = change_brightness(sun, delta_b)
        sun_saturation = get_avg_saturation(sun)
        delta_s = 1 + (img_saturation - sun_saturation) / 255
        sun = change_saturation(sun, delta_s)

        for c in range(0, 3):
            dst[:,:,c] = a * sun[:,:,c] + dst[:,:,c] * (1-a)

    return dst


# These codes are from aqeelanwar/MaskTheFace
def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)

def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)

def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def build_loss_func(loss_opt, device, cls_num_list=None):

    func = None
    if loss_opt == 'ce':
        func = CELoss(device=device)

    if loss_opt == "ldam":
        func = LDAMLoss(cls_num_list=cls_num_list, device=device)

    if loss_opt == "cb":
        def CB_lossFunc(logits, labelList): #defince CB loss function
            return CB_loss(labelList, logits, np.array(cls_num_list), len(cls_num_list), "focal", 0.9999, 2.0, device)
        func = CB_lossFunc

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


def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_bbox(imgsize=(224,224),beta=1.0):

    r = np.random.rand(1)
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgsize, lam)

    return [bbx1,bby1,bbx2,bby2]

def SPM(image, model,target_layers, targets = None):
    """
    image와 model, 보고싶은 model의 layer, 모델이 어떤 target을 볼지 를 통해
    Semantic percentage maps을 리턴
    img : PIL.image or tensor
    model : model 
        ex) model = resnet50(pretrained = True)
    target_layers : layers we want to see 
        ex) target_layers = [model.layer4[-1]]
    targets : ClassifierOutputTarget(pytorch_grad_cam.utils.model_targets의 함수)
    를 활용해서 보고 싶은 target을 정함
    None이면 제일 activation이 큰 target을 봄
        ex ) targets = [ClassifierOutputTarget(282)]
    
    """
    if type(image) != torch.Tensor:
        image = transforms.ToTensor()(image)#image->tensor
        
    if len(image.shape) == 3:#batch가 아니라 하나의 이미지면
        image = image.view([1,3,224,224])
        
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=image, targets=None)

    spm = grayscale_cam / grayscale_cam.sum()
    
    return spm, grayscale_cam


def visualize(input_tensor, grayscale_cam):
    """
    이미지 별로 visualize 하기 때문에 
    input_tensor.shape = (c, W, H)
    grayscale_cam.shape = (1, W, H)
    batch 로 넣으면 안됨
    """
    
    img = transforms.ToPILImage()(input_tensor)
    img_np = np.array(img)
    visualization = show_cam_on_image(img_np/255., grayscale_cam, use_rgb=True)
    return visualization

def mixup_onehot(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets.type(torch.LongTensor)

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size()[0])
    targets2 = targets[indices]

    W = data.size()[2]
    H = data.size()[3]
    bbx1,bby1, bbx2, bby2 = get_bbox((W,H), alpha)
        
    tmp = data.clone()
    tmp[:,:, bbx1:bbx2, bby1:bby2] = data[indices,:, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H)) # adjust lambda
    targets = targets * lam + targets2 * (1-lam)

    return tmp, targets.type(torch.LongTensor)

def snapmix(data, targets ,model,target_layers, beta ):
    """
    data : batch of data shape(batch_size, c, w, H)
    """

    #random indicies within batch
    indices = torch.randperm(data.size()[0])
    
    #weighted feature maps and targets according to random indicies
    wfmaps,_ = SPM(data, model, target_layers)
    wfmaps2 = wfmaps[indices,:,:]
    targets2 = targets[indices].clone()
    same_label = torch.tensor([i for i in range(data.size()[0])]) == indices
    
    
    #random bbox 
    lam = np.random.beta(beta,beta) 
    lam1 = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(data.size(), lam1)
    area = (bby2-bby1)*(bbx2-bbx1)
    area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)
    
    lam_a = torch.ones(data.size(0))
    lam_b = 1 - lam_a
    
    ret = data.clone()
    
    
    if area1 > 0 and area > 0 :
        tmp = data[indices , : ,bbx1_1:bbx2_1, bby1_1:bby2_1].clone()#data2
        tmp = torch.nn.functional.interpolate(tmp, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
        ret[:, :, bbx1:bbx2, bby1:bby2] = tmp

        
        lam_a = torch.tensor(1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1))
        lam_b = torch.tensor(wfmaps2[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1))
        
        tmpa = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmpa[same_label]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        lam_a[torch.isnan(lam_a)] = lam
        lam_b[torch.isnan(lam_b)] = 1-lam
        targets = targets * lam_a.to(targets.get_device()) + targets2 * lam_b.to(targets.get_device())

    return ret, targets.type(torch.LongTensor)
            
class MaxNorm_via_PGD():
    # learning a max-norm constrainted network via projected gradient descent (PGD) 
    def __init__(self, thresh=1.0, LpNorm=2, tau = 1):
        self.thresh = thresh
        self.LpNorm = LpNorm
        self.tau = tau
        self.perLayerThresh = []
        
    def setPerLayerThresh(self, model):
        # set per-layer thresholds
        self.perLayerThresh = []
        
        for curLayer in [model.model.fc[0].weight, model.model.fc[0].bias]: #here we only apply MaxNorm over the last two layers
            curparam = curLayer.data
            if len(curparam.shape)<=1: 
                self.perLayerThresh.append(float('inf'))
                continue
            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1).detach().unsqueeze(-1)
            curLayerThresh = neuronNorm_curparam.min() + self.thresh*(neuronNorm_curparam.max() - neuronNorm_curparam.min())
            self.perLayerThresh.append(curLayerThresh)
                
    def PGD(self, model):
        if len(self.perLayerThresh)==0:
            self.setPerLayerThresh(model)
        
        for i, curLayer in enumerate([model.model.fc[0].weight, model.model.fc[0].bias]): #here we only apply MaxNorm over the last two layers
            curparam = curLayer.data


            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)    
            curLayerThresh = self.perLayerThresh[i]
            
            idx = neuronNorm_curparam > curLayerThresh
            idx = idx.squeeze()
            tmp = curLayerThresh / (neuronNorm_curparam[idx].squeeze())**(self.tau)
            for _ in range(len(scalingVect.shape)-1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx] 

class Normalizer(): 
    def __init__(self, LpNorm=2, tau = 1):
        self.LpNorm = LpNorm
        self.tau = tau
  
    def apply_on(self, model): #this method applies tau-normalization on the classifier layer

        for curLayer in [model.model.fc[0].weight]: #change to last layer: Done
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)    
            
            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape)-1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]
