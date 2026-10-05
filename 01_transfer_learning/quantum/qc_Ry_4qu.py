import time
import math
import tqdm
import subprocess
import numpy as np
import torch
import torch.nn as nn
from itertools import product

import torchani
from torchani.units import hartree2kcalmol
from torchani.nn import Sequential

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

from ani_transfer.Model_func import validate, fix_seed
from ani_transfer.read_dataset import get_data
from ani_transfer.quantum_model import RotCircuit, RyCircuit, RzRyCircuit
from ani_transfer.transfer_model import TransferANI
from ani_transfer.paths import MODEL_DIR, RESULTS_DIR

import pennylane as qml


def main():
    # data_No = 4
    # n_layer = 2
    # n_params = 3

    nwires = 4
    lr = 0.001
    batch_size = 256
    n_train = 5000
    n_valid = 2000
    # q_delta = 1.0

    device = torch.device("cpu")
    # quantum model
    qc_dev = qml.device("default.qubit", wires=nwires)

    # Dataset
    species_order = ['H', 'C', 'N', 'O']

    for data_No in range(1, 5):
        train_data, valid_data, _ = get_data(data_No, species_order)

        model_params = {
            "q_depth": np.arange(5, 6),
            "n_layer": np.arange(2, 5),
            "seed": np.arange(5),
        }
        keys, values = zip(*model_params.items())
        params_product = [dict(zip(keys, p)) for p in product(*values)]

        for params in params_product:
            q_depth, n_layer, seed = params.values()

            # Data
            fix_seed(seed)
            training, _ = train_data.shuffle().split(n_train / len(train_data), None)
            validation, _ = valid_data.shuffle().split(n_valid / len(valid_data), None)
            training = training.collate(batch_size).cache()
            validation = validation.collate(batch_size).cache()

            # Transfer model
            circuit = RyCircuit(qc_dev, q_depth)
            weight_shapes = {"weights": (q_depth, 2, 1, nwires)}

            ani_instance = TransferANI(n_layer, nwires, device, lr=lr)
            model, optimizer_weight, scheduler_weight = ani_instance.transfer_qlayer(
                circuit, weight_shapes
            )
            model = model.to(device)
            
            # Train
            result_path = f"QC_s0{data_No}_{n_layer}layer_{nwires}qu_d{q_depth}_seed{seed}_Ry"
            train(
                model, training, validation,
                optimizer_weight, scheduler_weight,
                result_path, device, seed
            )


def train(
    model,
    training,
    validation,
    optimizer_weight,
    scheduler_weight,
    result_path,
    device,
    seed,
    max_epochs=100,
    early_stopping_learning_rate=1.0e-5
):
    start_time = time.time()
    fix_seed(seed)

    mse = nn.MSELoss(reduction='none')
    mse_sum = nn.MSELoss(reduction='sum')
    for j in range(scheduler_weight.last_epoch + 1, max_epochs + 1):
        rmse, _, _ = validate(model, validation)
        with open(RESULTS_DIR / "valid" / "Quantum_transfer" / f"valid_{result_path}.txt", "a") as f:
            f.write(str(rmse))
            f.write("\n")
        print("RMSE: ", rmse, "at epoch", scheduler_weight.last_epoch + 1)
        
        learning_rate = optimizer_weight.param_groups[0]["lr"]
        
        if learning_rate < early_stopping_learning_rate:
            break
            
        #モデル保存
        best_model_checkpoint = MODEL_DIR / "Quantum_transfer" / f"best_{result_path}.pt"
        if scheduler_weight.is_better(rmse, scheduler_weight.best):
            torch.save(model.state_dict(), best_model_checkpoint)
            
        # パラメータアップデート
        scheduler_weight.step(rmse)

        train_loss = 0.0
        total_mse = 0.0
        count = 0
        for i, properties in tqdm.tqdm(
            enumerate(training),
            total=len(training),
            desc="epoch {}".format(scheduler_weight.last_epoch)
        ):
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()

            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))

            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            train_loss += loss.item()
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

            optimizer_weight.zero_grad()
            loss.backward()
            optimizer_weight.step()

        avg_train_loss = train_loss / count
        train_rmse = hartree2kcalmol(math.sqrt(total_mse / count))
        with open(RESULTS_DIR / "train" / "Quantum_transfer" / f"train_{result_path}.txt", "a") as f:
            f.write(str(train_rmse))
            f.write('\n')

    end_time = time.time()
    print('Calculation time: ', end_time - start_time)


if __name__ == "__main__":
    main()
