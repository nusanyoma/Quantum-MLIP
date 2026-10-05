import argparse
import sys
import pathlib
import torch
import torchani
import numpy as np
import pennylane as qml

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    for _cand in (_p / "lib", _p / "code" / "code" / "lib"):
        if (_cand / "ani_transfer").is_dir():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

from ani_transfer.transfer_model import TransferANI
from ani_transfer.quantum_model import RzRyCircuit
from ani_transfer.paths import DATA_DIR, MODEL_DIR


def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    symbols = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        s, x, y, z = line.split()
        symbols.append(s)
        coords.append([float(x), float(y), float(z)])
    return symbols, np.array(coords, dtype=np.float32)

def load_classical_transferred(data_No=1, n_layer=1, n_qubits=4, q_depth=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ani_instance = TransferANI(n_layer, n_qubits, device)

    model, _, _, _, _ = ani_instance.transfer_clayer()
    model_path = MODEL_DIR / "Classic_transfer" / f"best_CC_s0{data_No}_{n_layer}layer_{n_qubits}qu_seed0.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)

    return model

def load_quantum_transferred(data_No=1, n_layer=1, n_qubits=4, q_depth=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ani_instance = TransferANI(n_layer, n_qubits, device)

    # 量子回路
    qc_dev = qml.device("default.qubit", wires=n_qubits)
    circuit = RzRyCircuit(qc_dev, q_depth)
    weight_shapes = {"weights": (q_depth, 2, 2, n_qubits)}

    model, _, _ = ani_instance.transfer_qlayer(circuit, weight_shapes)
    model_path = MODEL_DIR / "Quantum_transfer" / f"best_QC_s0{data_No}_{n_layer}layer_{n_qubits}qu_d{q_depth}_seed0_all_RzRy.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)

    return model

def add_self_energies(residual_energy, species, self_energies=[-0.6047, -38.0684, -54.7061, -75.1796]):
    """
    残差エネルギー（推論値）にself_energiesを加算して絶対エネルギーに戻す
    - residual_energy: モデル出力の残差エネルギー（floatまたはnp.ndarray, torch.Tensor）
    - species: ['H', 'C', 'N', 'O', ...] など分子内の原子記号リスト
    - self_energies: 各原子種のself_energy（torch.Tensorやnp.ndarray, listなど）
    """
    # 合計self_energy
    total_self_energy = sum([self_energies[s] for s in species])
    # 絶対エネルギーに戻す
    abs_energy = residual_energy + total_self_energy
    return abs_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("model_type", choices=["original", "classic", "quantum"])
    args = parser.parse_args()
    file = args.file
    model_type = args.model_type

    symbols, coordinates = read_xyz(str(DATA_DIR / f"{file}.xyz"))
    origin_model = torchani.models.ANI1x()

    # モデル読み込み
    if model_type == "original":    
        model = origin_model
    elif model_type == "classic":
        model = load_classical_transferred(data_No=4, n_layer=1, n_qubits=4)
    elif model_type == "quantum":
        model = load_quantum_transferred(data_No=4, n_layer=3, n_qubits=4)

    species_to_tensor = torchani.utils.ChemicalSymbolsToInts(origin_model.species)
    species = species_to_tensor(symbols).unsqueeze(0)
    coordinates = torch.tensor([coordinates], requires_grad=True)
    
    # エネルギー計算
    energy = model((species, coordinates)).energies
    if model_type != "original":
        energy = add_self_energies(energy, species.squeeze())
    print("Energy (Hartree): ", energy.item())
    print("Energy (kcal/mol): ", energy.item() * 627.509)


if __name__ == "__main__":
    main()
