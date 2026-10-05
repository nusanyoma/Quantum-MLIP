import numpy as np
import torch
import pickle
from itertools import product
import pennylane as qml

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

from ani_transfer.Model_func import validate_gpu
from ani_transfer.read_dataset import get_data
from ani_transfer.quantum_model import RotCircuit, RyCircuit, RzRyCircuit
from ani_transfer.transfer_model import TransferANI
from ani_transfer.paths import MODEL_DIR

def main():
    device = torch.device("cuda")
    batch_size = 256

    rmse_dict = {}
    pred_dict = {}

    # Dataset
    species_order = ["H", "C", "N", "O"]
    for data_No in range(1, 5):
        _, _, test_data = get_data(data_No, species_order)
        test = test_data.collate(batch_size).cache()

        model_params = {
            "n_qubits": (4, 8, 12),
            "q_depth": (1, 5),
            "n_layer": np.arange(1, 5),
            "seed": np.arange(5),
        }
        keys, values = zip(*model_params.items())
        params_product = [dict(zip(keys, p)) for p in product(*values)]

        for params in params_product:
            n_qubits, q_depth, n_layer, seed = params.values()

            # Transfer model
            qc_dev = qml.device("default.qubit", wires=n_qubits)
            circuit = RyCircuit(qc_dev, q_depth)
            weight_shapes = {"weights": (q_depth, 2, 1, n_qubits)}

            ani_instance = TransferANI(n_layer, n_qubits, device)
            model, _, _ = ani_instance.transfer_qlayer(circuit, weight_shapes)
            model_path = MODEL_DIR / "Quantum_transfer" / f"best_QC_s0{data_No}_{n_layer}layer_{n_qubits}qu_d{q_depth}_seed{seed}_Ry.pt"
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.to(device)

            rmse, _, pred_array = validate_gpu(model, test)
            key = f"s0{data_No}_{n_qubits}qu_{n_layer}layer_{q_depth}depth"
            if key not in rmse_dict:
                rmse_dict[key] = []
                pred_dict[key] = []
            rmse_dict[key].append(rmse)
            pred_dict[key].append(pred_array)

    with open(_here.parent / f"QC_rmse_dict_Ry.pickle", "wb") as f:
        pickle.dump(rmse_dict, f)
    with open(_here.parent / f"QC_pred_dict_Ry.pickle", "wb") as f:
        pickle.dump(pred_dict, f)

if __name__ == "__main__":
    main()
