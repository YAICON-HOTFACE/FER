import numpy as np
import cv2
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class FaceEmotionDataset(Dataset):
    def __init__(self, option="train", transform=None, gray=False, csv_file=None, emotion_only=True):
        self.option = option
        self.transform = transform
        self.gray = gray
        self.emotion_only = emotion_only

        assert self.option in ["train", "val"], "Choose between 'train' and 'val'"
        
        if csv_file is None:
            print("Getting samples from directory...")
            self.images, self.labels = self._get_samples_with_labels()
        
        else:
            print("Getting samples from csvfile...")
            if self.option == "train":
                if not os.path.isfile(csv_file):
                    raise ValueError("train_dataset.csv file does not exist.")

                if "train" not in csv_file:
                    print(f"option({option}) and csv_file({csv_file}) do not match each other")
                    csv_file = csv_file.replace("train", "val")

                info = pd.read_csv(os.path.abspath(csv_file))

            elif self.option == "val":
                if not os.path.isfile(csv_file):
                    raise ValueError("val_dataset.csv file does not exist.")

                if "val" not in csv_file:
                    print(f"option({option}) and csv_file({csv_file}) do not match each other")
                    csv_file = csv_file.replace("train", "val")
                
                info = pd.read_csv(os.path.abspath(csv_file))

            else:
                raise ValueError("Not available metric")

            self.images, self.labels = info["images"], info["labels"]

        if emotion_only:
            self.images, self.labels = self._get_only_emotion()

    def __len__(self):
        assert len(self.images) == len(self.labels), "images and labels should exist as pair"
        return len(self.labels)
    
    def _get_samples_with_labels(self):
        images, labels = [], []
        if self.option == "train":
            rootdir = os.path.abspath("../AFFNet/train_set/images")
            labeldir = os.path.abspath("../AFFNet/train_set/annotations")
            for file in tqdm(os.listdir(rootdir)):
                filename, fileext = os.path.splitext(file)
                if fileext == ".jpg":
                    images += [os.path.join(rootdir, file)]                              
                    labels += [np.load(os.path.join(labeldir, filename+"_exp.npy"))]

        elif self.option == "val":
            rootdir = os.path.abspath("../AFFNet/val_set/images")
            labeldir = os.path.abspath("../AFFNet/val_set/annotations")
            for file in tqdm(os.listdir(rootdir)):
                filename, fileext = os.path.splitext(file)
                if fileext == ".jpg":
                    image += [os.path.join(rootdir, file)]                                
                    label += [np.load(os.path.join(labeldir, filename+"_exp.npy"))]

        else:
            raise ValueError("Not available metric")

        
        return images ,labels

    def print_cls_num(self):
        if self.emotion_only:
            emo_idx = ["Neutral", "Happiness", "Sadness", "Surprise",
                       "Fear", "Disgust", "Anger", "Contempt"]

            emo_num = [0]*8
            for lbl in self.labels:
                emo_num[lbl] += 1

        else:
            emo_idx = ["Neutral", "Happiness", "Sadness", "Surprise",
                       "Fear", "Disgust", "Anger", "Contempt", "None", "Uncertain", "No-Face"]

            emo_num = [0]*11
            for lbl in self.labels:
                emo_num[lbl] += 1

        for emo, num in zip(emo_idx, emo_num):
            print(f"{emo} : {num}")

    def _get_only_emotion(self):
        images, labels = [], []
        for (img, lbl) in zip(self.images, self.labels):
            if lbl<=7:
                images += [img]
                labels += [lbl]

        return images, labels

    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.labels[idx]
        image = cv2.cvtColor(cv2.imread(img_path),
                             cv2.COLOR_BGR2GRAY if self.gray else cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image).float()

        return image, torch.from_numpy(np.array(label))
