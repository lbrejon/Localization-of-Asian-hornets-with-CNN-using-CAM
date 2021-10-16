import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class DatasetTrain(object):
    def __init__(self, root, transform=None):
        self.root = root
        self.transforms = transform
        n = len(list(sorted(os.listdir(os.path.join(root, "absence")))))
        # load all image files, sorting them to
        # ensure that they are aligned
        liste = list(sorted(os.listdir(os.path.join(root, "absence")))) + list(sorted(os.listdir(os.path.join(root, "presence"))))
        self.imgs = []
        for i in range(len(liste)):
            if i<n:
                self.imgs.append((liste[i],0))
            else:
                self.imgs.append((liste[i],1))
        
    def __getitem__(self, idx):
        # load images ad masks
        target = self.imgs[idx][1]
        if (target):
            img_path = os.path.join(self.root, "presence", self.imgs[idx][0])
        else:
            img_path = os.path.join(self.root, "absence", self.imgs[idx][0])
        

        img = Image.open(img_path).convert("RGB")
        
        
        preprocess = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        

        return input_tensor,target

    def __len__(self):
        return len(self.imgs)
