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

    return data, targets

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size()[0])
    targets2 = targets[indices]

    W = data.size()[2]
    H = data.size()[3]
    bbx1,bby1, bbx2, bby2 = get_bbox((W,H),alpha)

    data[:,:, bbx1:bbx2, bby1:bby2] = data[indices,:, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H)) # adjust lambda
    targets = targets * lam + targets2 * (1-lam)

    return data, targets

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
        
        targets = targets * lam_a + targets2 * lam_b
    
    return ret, targets


def top_k(a):
    k = 6
    idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))

def replace_attentive_regions(rand_img, image, attentive_regions,grid_size):
    """
    rand_img: the img to be replaced
    image: where the 'patches' come from
    attentive_regions: an array contains the coordinates of attentive regions
    """
    np_rand_img, np_img = np.array(rand_img), np.array(image)
    for attentive_region in attentive_regions:
        replace_attentive_region(np_rand_img, np_img, attentive_region,grid_size)
    return Image.fromarray(np_rand_img)

def replace_attentive_region(np_rand_img, np_img, attentive_region,grid_size):
    x, y = attentive_region
    x1, x2, y1, y2 = grid_size * x, grid_size * (x+1), grid_size * y, grid_size * (y+1)
    region = np_img[x1:x2, y1: y2]
    np_rand_img[x1:x2, y1:y2] = region
    

def atten_cutmix(data, targets, model):
    img_list = [transforms.ToPILImage()(tensor) for tensor in data]
    temp_model = torch.nn.Sequential(*list(model.children())[:-2])
    model = temp_model
    grid_size = 32 #224/7
    
    output = []
    out_targets = []
    for i in range(len(img_list)):
        img = img_list[i]

        rand_index = np.random.randint(0, len(img_list))
        while rand_index == i:
            rand_index = np.random.randint(0, len(img_list))
            
        rand_img = img_list[rand_index]
        attentive_regions = top_k(model(transforms.functional.to_tensor(rand_img).unsqueeze_(0))[0][-1].detach().numpy())
        img = replace_attentive_regions(img, rand_img, attentive_regions,grid_size)
        output.append(transforms.ToTensor()(img))
        out_targets.append(targets[i]*(43./49.) + targets[rand_index]*(6./49.))

    output = torch.stack(output, dim=0)
    out_targets = torch.stack(out_targets, dim = 0)
    
    return output, out_targets
