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
import pickle
from torch.utils.tensorboard import SummaryWriter

def train(cfg, args, writer=None):
    # (0) : global
    num_classes = 8 if cfg['emotion'] else 11
    in_channel = 1 if cfg["gray"] else 3
    device = args.device
    batch_size = cfg['dataset']['batch']
    model_name = cfg['train']['model']

    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),
                                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                            ])


    train_dataset = FaceEmotionDataset("train", transform, cfg["gray"], csv_file=cfg["dataset"]["train_csv"], emotion_only=cfg["emotion"])
    val_dataset = FaceEmotionDataset("val", transform, cfg["gray"], csv_file=cfg["dataset"]["val_csv"], emotion_only=cfg["emotion"])
    
    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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


    print("Model configuration : ")
    print(pytorch_model_summary.summary(model,
                                torch.zeros(batch_size, in_channel, 224, 224),
                                show_input=True))


    # (0) : Loss function
    loss_func = build_loss_func(cfg['train']['loss'], device=device)

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
        optimizer = optimizer.load_state_dict(checkpoint['optimizers_state_dict'])
        scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start = checkpoint['epoch']

    for epoch in range(start, epochs):

        # (0) : Training
        training_loss, training_acc = 0.0, 0.0
        model.train()
        loading = tqdm(enumerate(train_dataloader), desc="training...")
        for i, (image, label) in loading:
            
            optimizer.zero_grad()
            if cfg["train"]["mixup"]:
                image, label = mixup_onehot(image, label, 1)
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            if cfg["train"]["logit"]:
                prediction += logit_adjustment
            loss = compute_loss(loss_func, prediction, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            _, prediction = torch.max(prediction, axis=1)
            accuracy = float(torch.sum(torch.eq(prediction, label)))/len(prediction)
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
                validation_loss += loss.item()
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
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    logs_base_dir = 'logs'
    os.makedirs(logs_base_dir, exist_ok = True)
    exp = f"{logs_base_dir}/ex{args.exp}"
    writer = SummaryWriter(exp)

    train(cfg, args, writer)