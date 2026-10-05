import os
import time
import math
import tqdm
import numpy as np
import pickle
from itertools import product

import torch
import torch.nn as nn

from torchani.units import hartree2kcalmol

import sys
import pathlib as _pathlib
_here = _pathlib.Path(__file__).resolve()
for _p in _here.parents:
    for _cand in (_p / "lib", _p / "code" / "code" / "lib"):
        if (_cand / "ani_transfer").is_dir():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

from ani_transfer.Model_func import validate_gpu, fix_seed
from ani_transfer.read_dataset import get_data
from ani_transfer.transfer_model import TransferANI
from ani_transfer.paths import MODEL_DIR, RESULTS_DIR


def train(
    model,
    training,
    validation,
    optimizer_weight,
    scheduler_weight,
    optimizer_bias,
    scheduler_bias,
    device,
    result_path,
    seed,
    max_epochs=100,
    early_stopping_learning_rate=1e-10,
):

    start_time = time.time()
    fix_seed(seed)

    train_dir = RESULTS_DIR / "train" / "Classic_transfer"
    valid_dir = RESULTS_DIR / "valid" / "Classic_transfer"
    model_dir = MODEL_DIR / "Classic_transfer"

    mse = nn.MSELoss(reduction='none')
    mse_sum = nn.MSELoss(reduction='sum')

    for j in range(scheduler_weight.last_epoch + 1, max_epochs + 1):
        rmse, _, _ = validate_gpu(model, validation)
        with open(valid_dir / f"valid_{result_path}.txt", "a") as f:
            f.write(str(rmse))
            f.write("\n")
        print("RMSE: ", rmse, "at epoch", scheduler_weight.last_epoch + 1)

        learning_rate = optimizer_weight.param_groups[0]["lr"]

        if learning_rate < early_stopping_learning_rate:
            break

        # Preserve model weights
        best_model_checkpoint = model_dir / f"best_{result_path}.pt"
        if scheduler_weight.is_better(rmse, scheduler_weight.best):
            torch.save(model.state_dict(), best_model_checkpoint)

        # Update parameters
        scheduler_weight.step(rmse)
        scheduler_bias.step(rmse)

        train_loss = 0.0
        total_mse = 0.0
        count = 0

        for i, properties in tqdm.tqdm(
            enumerate(training),
            total=len(training),
            desc=f"epoch {scheduler_weight.last_epoch}"
        ):
            species = properties["species"].to(device)
            coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))

            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            train_loss += loss.item()
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

            optimizer_weight.zero_grad()
            optimizer_bias.zero_grad()
            loss.backward()
            optimizer_weight.step()
            optimizer_bias.step()

        avg_train_loss = train_loss / count
        train_rmse = hartree2kcalmol(math.sqrt(total_mse / count))
        with open(train_dir / f"train_{result_path}.txt", "a") as f:
            f.write(str(train_rmse))
            f.write("\n")

    end_time = time.time()
    print("Calculation time: ", end_time - start_time)


def main():
    lr = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    species_order = ['H', 'C', 'N', 'O']
    for data_No in range(2, 3):
        train_data, valid_data, _ = get_data(data_No, species_order)

        model_params = {
            "n_qubits": (4, 8, 12),
            "n_layer": np.arange(1, 5),
            "seed": np.arange(5),
        }
        keys, values = zip(*model_params.items())
        params_product = [dict(zip(keys, p)) for p in product(*values)]

        batch_size = 256
        n_train = 5000
        n_valid = 2000
        
        try:
            path = os.path.dirname(os.path.realpath(__file__))
        except NameError:
            path = os.getcwd()

        for params in params_product:
            # n_layer, data_No, depth, seed = params.values()
            n_qubits, n_layer, seed = params.values()
            
            fix_seed(seed)
            training, _ = train_data.shuffle().split(n_train / len(train_data), None)
            validation, _ = valid_data.shuffle().split(n_valid / len(valid_data), None)
            training = training.collate(batch_size).cache()
            validation = validation.collate(batch_size).cache()

            # Transfer model
            ani_instance = TransferANI(n_layer, n_qubits, device, lr=lr)
            model, optimizer_weight, scheduler_weight, optimizer_bias, scheduler_bias = ani_instance.transfer_clayer()
            # print(model)
            model = model.to(device)

            # Training
            result_path = f"CC_s0{data_No}_{n_layer}layer_{n_qubits}qu_seed{seed}"
            train(
                model,
                training,
                validation,
                optimizer_weight,
                scheduler_weight,
                optimizer_bias,
                scheduler_bias,
                device,
                result_path,
                seed,
            )


if __name__ == "__main__":
    main()