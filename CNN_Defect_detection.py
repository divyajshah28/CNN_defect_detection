# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:54:38 2022

@author: divshah

This code implements a CNN based apporach for defect classification and detection
"""
#importing all constants
from constants import*
from data_loader_CNN import get_train_test_loaders
import os
from helper import train, evaluate, predict_localize
from model import CustomVGG
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 10

data_folder = os.path.join(root_dir, "bottle")
print(data_folder)

target_train_accuracy = 0.98
lr = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7
n_cv_folds = 5

#data loading part

train_loader, test_loader = get_train_test_loaders(
    data_folder, batch_size, test_size=0.2, random_state=42)

model = CustomVGG()

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)

model = train(
    train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
)

model_path = f"D:\CS230_DeepLearning\classification_model\CNN\weights\bottle_model.h5"
torch.save(model, model_path)