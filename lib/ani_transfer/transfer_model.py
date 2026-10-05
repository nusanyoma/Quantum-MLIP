import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict

import torchani
import pennylane as qml

from ani_transfer.AEV_constant import AEV_constant
from ani_transfer.read_dataset import get_data
from ani_transfer.Model_func import validate, fix_seed
from ani_transfer.classic_model import ClassicNet
from ani_transfer.quantum_model import RotCircuit, RyCircuit, RzRyCircuit
from ani_transfer.paths import MODEL_DIR


class TransferANI:
    def __init__(self, n_layer, nwires, device, lr=0.001):
        self.n_layer = n_layer
        self.nwires = nwires
        self.device = device
        self.lr = lr

        H_neurons = [160, 128, 96, self.nwires]
        C_neurons = [144, 112, 96, self.nwires]
        N_neurons = [128, 112, 96, self.nwires]
        O_neurons = [128, 112, 96, self.nwires]
        self.H_neurons = H_neurons[-n_layer:]
        self.C_neurons = C_neurons[-n_layer:]
        self.N_neurons = N_neurons[-n_layer:]
        self.O_neurons = O_neurons[-n_layer:]

    def define_ani(self, aev_computer):
        aev_dim = aev_computer.aev_length

        H_layer = []
        C_layer = []
        N_layer = []
        O_layer = []
        self.H_neurons.insert(0, aev_dim)
        self.C_neurons.insert(0, aev_dim)
        self.N_neurons.insert(0, aev_dim)
        self.O_neurons.insert(0, aev_dim)
        for i in range(self.n_layer):
            H_layer.append(nn.Linear(self.H_neurons[i], self.H_neurons[i + 1]))
            C_layer.append(nn.Linear(self.C_neurons[i], self.C_neurons[i + 1]))
            N_layer.append(nn.Linear(self.N_neurons[i], self.N_neurons[i + 1]))
            O_layer.append(nn.Linear(self.O_neurons[i], self.O_neurons[i + 1]))
            if i < self.n_layer - 1:
                H_layer.append(nn.CELU(0.1))
                C_layer.append(nn.CELU(0.1))
                N_layer.append(nn.CELU(0.1))
                O_layer.append(nn.CELU(0.1))
        H_layer.append(nn.Sigmoid())
        C_layer.append(nn.Sigmoid())
        N_layer.append(nn.Sigmoid())
        O_layer.append(nn.Sigmoid())
        H_layer.append(nn.Linear(self.nwires, 1))
        C_layer.append(nn.Linear(self.nwires, 1))
        N_layer.append(nn.Linear(self.nwires, 1))
        O_layer.append(nn.Linear(self.nwires, 1))
        
        H_network = nn.Sequential(*H_layer)
        C_network = nn.Sequential(*C_layer)
        N_network = nn.Sequential(*N_layer)
        O_network = nn.Sequential(*O_layer)

        ani_nn = torchani.ANIModel(OrderedDict([
            ("H", H_network),
            ("C", C_network),
            ("N", N_network),
            ("O", O_network),
        ]))

        return ani_nn

    def load_ani(self):
        device = self.device
        AEV = AEV_constant(device)
        species_order = ["H", "C", "N", "O"]
        num_species = len(species_order)
        aev_computer = torchani.AEVComputer(
            AEV.Rcr,
            AEV.Rca,
            AEV.EtaR,
            AEV.ShfR,
            AEV.EtaA,
            AEV.Zeta,
            AEV.ShfA,
            AEV.ShfZ,
            num_species
        )

        ani_model = self.define_ani(aev_computer)
        model = nn.Sequential(aev_computer, ani_model).to(device)

        l = self.n_layer + 1
        filename = f"best_Default_nnp_shuffle_l{l}_lr0.001_sigmoid_un_selfE_{self.nwires}qu.pt"
        model_path = MODEL_DIR / "Classic" / filename
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device(device)))

        return model

    def transfer_clayer(self):
        device = self.device
        model = self.load_ani()

        for param in model.parameters():
            param.requires_grad = False

        final_n = self.n_layer * 2

        # Get final layer's weights
        H_final_weight = model[1].H[final_n].weight
        C_final_weight = model[1].C[final_n].weight
        N_final_weight = model[1].N[final_n].weight
        O_final_weight = model[1].O[final_n].weight
        H_final_bias = model[1].H[final_n].bias
        C_final_bias = model[1].C[final_n].bias
        N_final_bias = model[1].N[final_n].bias
        O_final_bias = model[1].O[final_n].bias

        # Transfer
        model[1].H[final_n] = ClassicNet(self.nwires, device)
        model[1].C[final_n] = ClassicNet(self.nwires, device)
        model[1].N[final_n] = ClassicNet(self.nwires, device)
        model[1].O[final_n] = ClassicNet(self.nwires, device)
        # Initialize weight
        self.cc_init_params(model[1].H[final_n].c_net)
        self.cc_init_params(model[1].C[final_n].c_net)
        self.cc_init_params(model[1].N[final_n].c_net)
        self.cc_init_params(model[1].O[final_n].c_net)

        # Add output layer
        add_final_n = final_n + 1
        model[1].H.add_module(str(add_final_n), nn.Linear(self.nwires, 1))
        model[1].C.add_module(str(add_final_n), nn.Linear(self.nwires, 1))
        model[1].N.add_module(str(add_final_n), nn.Linear(self.nwires, 1))
        model[1].O.add_module(str(add_final_n), nn.Linear(self.nwires, 1))

        model[1].H[add_final_n].weight = H_final_weight
        model[1].C[add_final_n].weight = C_final_weight
        model[1].N[add_final_n].weight = N_final_weight
        model[1].O[add_final_n].weight = O_final_weight
        model[1].H[add_final_n].bias = H_final_bias
        model[1].C[add_final_n].bias = C_final_bias
        model[1].N[add_final_n].bias = N_final_bias
        model[1].O[add_final_n].bias = O_final_bias

        update_weight_names = [f"{final_n}.c_net.weight"]
        update_bias_names = [f"{final_n}.c_net.bias"]

        weight_to_update = []
        bias_to_update = []
        for name, param in model[1].named_parameters():
            if name[2:] in update_weight_names:
                weight_to_update.append(param)
                # print("weight name: ", name)
            elif name[2:] in update_bias_names:
                bias_to_update.append(param)
                # print("bias name: ", name)
            else:
                continue

        optimizer_weight = optim.AdamW(weight_to_update, lr=self.lr)
        optimizer_bias = optim.SGD(bias_to_update, lr=self.lr)

        scheduler_weight = lr_scheduler.ReduceLROnPlateau(
            optimizer_weight, factor=0.5, patience=100, threshold=0)
        scheduler_bias = lr_scheduler.ReduceLROnPlateau(
            optimizer_bias, factor=0.5, patience=100, threshold=0)

        return model, optimizer_weight, scheduler_weight, optimizer_bias, scheduler_bias

    def cc_init_params(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def transfer_qlayer(self, q_circuit, weight_shapes):
        qc_dev = q_circuit.qc_dev
        q_depth = q_circuit.q_depth
        device = self.device
        nwires = self.nwires
        assert nwires == q_circuit.nwires

        model = self.load_ani()

        for param in model.parameters():
            param.requires_grad = False

        final_n = self.n_layer * 2

        # Get final layer's weights
        H_final_weight = model[1].H[final_n].weight
        C_final_weight = model[1].C[final_n].weight
        N_final_weight = model[1].N[final_n].weight
        O_final_weight = model[1].O[final_n].weight
        H_final_bias = model[1].H[final_n].bias
        C_final_bias = model[1].C[final_n].bias
        N_final_bias = model[1].N[final_n].bias
        O_final_bias = model[1].O[final_n].bias

        # Define qlayer
        circuit = q_circuit.q_circuit()
        # Define initial weights
        init_method = {"weights": nn.init.zeros_}

        # Transfer
        model[1].H[final_n] = qml.qnn.TorchLayer(circuit, weight_shapes, init_method)
        model[1].C[final_n] = qml.qnn.TorchLayer(circuit, weight_shapes, init_method)
        model[1].N[final_n] = qml.qnn.TorchLayer(circuit, weight_shapes, init_method)
        model[1].O[final_n] = qml.qnn.TorchLayer(circuit, weight_shapes, init_method)

        # Add output layer
        add_final_n = final_n + 1
        model[1].H.add_module(str(add_final_n), nn.Linear(nwires, 1))
        model[1].C.add_module(str(add_final_n), nn.Linear(nwires, 1))
        model[1].N.add_module(str(add_final_n), nn.Linear(nwires, 1))
        model[1].O.add_module(str(add_final_n), nn.Linear(nwires, 1))

        model[1].H[add_final_n].weight = H_final_weight
        model[1].C[add_final_n].weight = C_final_weight
        model[1].N[add_final_n].weight = N_final_weight
        model[1].O[add_final_n].weight = O_final_weight
        model[1].H[add_final_n].bias = H_final_bias
        model[1].C[add_final_n].bias = C_final_bias
        model[1].N[add_final_n].bias = N_final_bias
        model[1].O[add_final_n].bias = O_final_bias

        # model[1] = nn.Sequential()

        update_weight_names = [f"{final_n}.weights"]

        weight_to_update = []
        for name, param in model[1].named_parameters():
            if name[2:] in update_weight_names:
                weight_to_update.append(param)
            else:
                continue

        optimizer_weight = optim.AdamW(weight_to_update, lr=self.lr)

        scheduler_weight = lr_scheduler.ReduceLROnPlateau(
            optimizer_weight, factor=0.5, patience=100, threshold=0)

        return model, optimizer_weight, scheduler_weight


def main():
    # for test
    device = torch.device("cpu")
    nwires = 4
    n_layer = 3

    inst = TransferANI(n_layer, nwires, device)

    
    # CC test
    model, _, _, _, _ = inst.transfer_clayer()
    print(model)
    for param in model.parameters():
        print(param)
    """

    # QC test
    q_depth = 2
    qc_dev = qml.device("default.qubit", wires=nwires)
    q_circuit = RotCircuit(qc_dev, q_depth)
    weight_shapes = {"weights": (q_depth, 2, nwires, 3)}
    model, _, _ = inst.transfer_qlayer(q_circuit, weight_shapes)
    print(model)

    #for param in model.parameters():
    #    print(param)
    """

    batch_size = 256
    n_valid = 100
    _, valid_data, _ = get_data(data_No=1, species_order=["H", "C", "N", "O"])
    validation, _ = valid_data.shuffle().split(n_valid / len(valid_data), None)
    validation = validation.collate(batch_size).cache()

    rmse, _, _ = validate(model, validation)
    print("RMSE: ", rmse)


if __name__ == "__main__":
    main()
