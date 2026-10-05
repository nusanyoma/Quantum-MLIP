import os
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import torchani
from torchani.units import hartree2kcalmol
from ani_transfer.AEV_constant import AEV_constant


def validate(model, validation):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    mse = nn.MSELoss(reduction='none')
    mse_sum = nn.MSELoss(reduction='sum')

    total_mse = 0.0
    count = 0
    model.train(False)

    true_array = np.array([])
    pred_array = np.array([])
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))

            true_array = np.append(true_array, true_energies.cpu().data.numpy())
            pred_array = np.append(pred_array, predicted_energies.cpu().data.numpy())

            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

    rmse_kcalmol = hartree2kcalmol(math.sqrt(total_mse / count))
    model.train(True)

    return rmse_kcalmol, true_array, pred_array


def validate_gpu(model, validation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mse = nn.MSELoss(reduction='none')
    mse_sum = nn.MSELoss(reduction='sum')

    total_mse = 0.0
    count = 0
    model.train(False)

    true_array = np.array([])
    pred_array = np.array([])
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))

            true_array = np.append(true_array, true_energies.cpu().data.numpy())
            pred_array = np.append(pred_array, predicted_energies.cpu().data.numpy())

            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

    rmse_kcalmol = hartree2kcalmol(math.sqrt(total_mse / count))
    model.train(True)

    return rmse_kcalmol, true_array, pred_array


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
