import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_model_summary
import torchvision.transforms as transforms
from model_factory import model_build
from dataset.dataset import FaceEmotionDataset
from utils import *
from loss import *
import pickle5 as pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pdb

def train(cfg, args, writer=None):
    # (0) : global
    if not cfg['emotion']:
        num_classes = 11
    elif cfg['dataset']['clsnum'] == 8:
        num_classes = 8
    elif cfg['dataset']['clsnum'] == 7:
        num_classes = 7
    else:
        raise ValueError("Not available class number")

    in_channel = 1 if cfg["gray"] else 3
    device = args.device
    batch_size = cfg['dataset']['batch']
    model_name = cfg['train']['model']

    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
                            ])

    transform_val=transforms.Compose([
                                transforms.ToTensor()
                            ])

    train_dataset = FaceEmotionDataset("train", transform, cfg["gray"], csv_file=cfg["dataset"]["train_csv"], emotion_only=cfg["emotion"], emotion_num=num_classes, masking=args.mask)
    val_dataset = FaceEmotionDataset("val", transform_val, cfg["gray"], csv_file=cfg["dataset"]["val_csv"], emotion_only=cfg["emotion"], emotion_num=num_classes, masking=args.mask)

    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    print("Calculate logit adjustment...")
    if os.path.isfile("logit_adjustment.pickle"):
        print("Load from existing file...")
        with open("logit_adjustment.pickle", "rb") as f:
            logit_adjustment = pickle.load(f)
    else:
        logit_adjustment = compute_adjustment(train_dataloader, device, 1.0)
        with open("logit_adjustment.pickle", "wb") as f:
            pickle.dump(logit_adjustment, f, protocol=pickle.HIGHEST_PROTOCOL)
    model = model_build(model_name=model_name, in_channel=in_channel, num_classes=num_classes)


    # (0) : Loss function
    if cfg['train']['loss'] not in ['ldam', 'cb']:
        loss_func = build_loss_func(cfg['train']['loss'], device=device)
    else:
        cls_num_list = train_dataset.get_cls_num()
        print(f"# of classes : {cls_num_list}")
        loss_func = build_loss_func(cfg['train']['loss'], device=device, cls_num_list=cls_num_list)

    # (1) : Optimizer & Scheduler
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    start = 0
    epochs = cfg['train']['epochs']

    # (2) : Device setting
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)

    # (3) : Create directory to save checkpoints
    os.makedirs(args.save, exist_ok=True)

    # (4) : Resume previous training
    if '.ckpt' in args.resume or '.pt' in args.resume:
        print("RESUME")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start = checkpoint['epoch']

    if args.finetune:
        if cfg['train']['loss'] == "cb":
            pgdFunc = MaxNorm_via_PGD(thresh=0.1)
            pgdFunc.setPerLayerThresh(model)
        active_layers = [model.model.fc[0].weight, model.model.fc[0].bias]
        for param in model.parameters():
            param.requires_grad = False
        
        for param in active_layers:
            param.requires_grad = True

        optimizer = build_optim(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)
        start = 0
        epochs = cfg['train']['epochs']

    print("Model configuration : ")
    print(pytorch_model_summary.summary(model,
                                torch.zeros(batch_size, in_channel, 224, 224).to(device),
                                show_input=True))
    
    for epoch in range(start, epochs):

        # (0) : Training
        training_loss, training_acc = 0.0, 0.0
        total = 0
        model.train()
        loading = tqdm(enumerate(train_dataloader), desc="training...")
        for i, (image, label) in loading:
            optimizer.zero_grad()
            if cfg["train"]["mix"]:
                choice = np.random.choice(['mixup', 'cutmix', 'naive'], p=[0.15, 0.15, 0.7])
                if choice == 'mixup':
                    image, label1, label2, lam = mixup_onehot(image, label, 1.0)
                    label1, label2 = label1.to(device), label2.to(device)
                    lam = float(lam)

                elif choice == 'cutmix':
                    image, label1, label2, lam = cutmix(image, label, 3.0)
                    label1, label2 = label1.to(device), label2.to(device)
                    lam = float(lam)
                    
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            if cfg["train"]["logit"]:
                prediction += logit_adjustment
            if choice == 'naive':
                loss = compute_loss(loss_func, prediction, label)
            else:
                loss = compute_loss(loss_func, prediction, label1)*lam + compute_loss(loss_func, prediction, label2)*(1-lam)
            loss_r = 0
            for parameter in model.parameters():
                loss_r += torch.sum(parameter ** 2)
            loss = loss + 1e-4 * loss_r
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            _, prediction = torch.max(prediction, axis=1)
            if choice == 'naive':
                accuracy = float(torch.sum(torch.eq(prediction, label)))/len(prediction)
            else:
                accuracy = (lam * prediction.eq(label1.data).cpu().sum().float()
                    + (1 - lam) * prediction.eq(label2.data).cpu().sum().float())/batch_size
                
            training_acc += accuracy
            loading.set_description(f"Loss : {training_loss/(i+1):.4f}, Acc : {100*training_acc/(i+1):.2f}%")
            # break

        print(f"Epoch #{epoch + 1} >>>> Training loss : {training_loss / len(train_dataloader):.6f}, Training acc : {100*training_acc/len(train_dataloader):.2f}%")
        if writer is not None:
            writer.add_scalar("Training loss", training_loss/len(train_dataloader), epoch)
            writer.add_scalar("Training accuracy", 100*training_acc/len(train_dataloader), epoch)
            writer.flush()
        scheduler.step()
        
        # (1): Evaluation
        model.eval()
        with torch.no_grad():
            validation_loss, val_acc = 0.0, 0.0
            for i, (image, label) in tqdm(enumerate(val_dataloader)):
                image, label = image.to(device), label.to(device)
                prediction = model(image)

                loss = compute_loss(loss_func, prediction, label)
                validation_loss += loss.item()

                _, prediction = torch.max(prediction, axis=1)
                accuracy = float(torch.sum(torch.eq(prediction, label)))/len(prediction)
                val_acc += accuracy
                # break

            print(f"Epoch #{epoch + 1} >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}, Validation acc : {100*val_acc/len(val_dataloader):.2f}%")
            if writer is not None:
                writer.add_scalar("Validation loss", validation_loss/len(val_dataloader), epoch)
                writer.add_scalar("Validation accuracy", 100*val_acc/len(val_dataloader), epoch)
                writer.flush()
        # (3) : Checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"{args.save}/checkpoint_{epoch}.ckpt")
        print(f"Epoch #{epoch + 1} >>>> SAVE .ckpt file")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='classification', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--save', type=str, default='checkpoint/classification', help='Path to save model file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    parser.add_argument('--exp', type=int, default=1, help="experiment number")
    parser.add_argument('--finetune', type=bool, default=False, help='fine tuning only fc layer')
    parser.add_argument('--mask', type=bool, default=False, help="Apply mask on data augmentation")
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    logs_base_dir = 'logs'
    os.makedirs(logs_base_dir, exist_ok = True)
    exp = f"{logs_base_dir}/ex{args.exp}"
    writer = SummaryWriter(exp)

    train(cfg, args, writer)
