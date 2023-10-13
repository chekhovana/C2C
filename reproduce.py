import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import albumentations
import torch.optim as optim
from albumentations.pytorch import ToTensorV2, ToTensor

from C2C.models.resnet import *
from C2C import train
from C2C.loss import KLDLoss
from C2C.eval_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic = True
model_ft = WSIClassifier(2, bn_track_running_stats=True)
model_ft = model_ft.to(device)

data_transforms = albumentations.Compose([
    ToTensor()
])

# Cross Entropy Loss
criterion_ce = nn.CrossEntropyLoss()
criterion_kld = KLDLoss()
criterion_dic = {'CE': criterion_ce, 'KLD': criterion_kld}

# Observe that all parameters are being optimized
optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)
model_ft = train.train_model(model_ft,
                             criterion_dic,
                             optimizer,
                             df,
                             data_transforms=data_transforms,
                             alpha=1,
                             beta=0.01,
                             gamma=0.01,
                             num_epochs=1,
                             fpath='trained/checkpoint.pt',
                             topk=True)

