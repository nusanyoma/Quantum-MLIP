import os
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def RZ_layer(w):
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)


def Rot_layer(w):
    # w = w.reshape(-1, 3)
    for idx, element in enumerate(w):
        qml.Rot(*element, wires=idx)


def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def inverse_entangling_layer(nqubits):
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


class BaseCircuit:
    def __init__(self, qc_dev, q_depth):
        self.qc_dev = qc_dev
        self.nwires = len(qc_dev.wires)
        self.q_depth = q_depth


class RotCircuit(BaseCircuit):
    def q_circuit(self):
        nwires = self.nwires
        qc_dev = self.qc_dev
        q_depth = self.q_depth

        @qml.qnode(qc_dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # q_weights = weights.reshape(q_depth, 2, nwires, 3)
            inputs_trans = torch.acos(inputs)
            
            # RY_layer(inputs)
            # RZ_layer(inputs)
            qml.AngleEmbedding(inputs_trans, wires=range(nwires), rotation="Y")
            qml.AngleEmbedding(inputs_trans, wires=range(nwires), rotation="Z")

            for q_weight in weights:
                entangling_layer(nwires)
                Rot_layer(q_weight[0])
                inverse_entangling_layer(nwires)
                Rot_layer(q_weight[1])

            return [qml.expval(qml.PauliZ(j)) for j in range(nwires)]

        return circuit

    def initialize_weights(self):
        weights = nn.Parameter(torch.zeros(self.q_depth * 2 * self.nwires * 3))
        return weights


class RzRyCircuit(BaseCircuit):
    def q_circuit(self):
        nwires = self.nwires
        qc_dev = self.qc_dev
        q_depth = self.q_depth

        @qml.qnode(qc_dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # q_weights = weights.reshape(q_depth, 2, 2, nwires)
            inputs_trans = torch.acos(inputs)

            # RY_layer(inputs)
            # RZ_layer(inputs)
            qml.AngleEmbedding(inputs_trans, wires=range(nwires), rotation="Y")
            qml.AngleEmbedding(inputs_trans, wires=range(nwires), rotation="Z")
            for q_weight in weights:
                entangling_layer(nwires)
                RZ_layer(q_weight[0][0])
                RY_layer(q_weight[0][1])
                inverse_entangling_layer(nwires)
                RZ_layer(q_weight[1][0])
                RY_layer(q_weight[1][1])

            return [qml.expval(qml.PauliZ(j)) for j in range(nwires)]
            
        return circuit

    def initialize_weights(self):
        weights = nn.Parameter(torch.zeros(self.q_depth * 2 * 2 * self.nwires))
        return weights


class RyCircuit(BaseCircuit):
    def q_circuit(self):
        nwires = self.nwires
        qc_dev = self.qc_dev
        q_depth = self.q_depth

        @qml.qnode(qc_dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # q_weights = weights.reshape(q_depth, 2, 1, nwires)
            inputs_trans = -torch.asin(inputs)

            H_layer(nwires)
            qml.AngleEmbedding(inputs_trans, wires=range(nwires), rotation="Y")
            for q_weight in weights:
                entangling_layer(nwires)
                RY_layer(q_weight[0][0])
                inverse_entangling_layer(nwires)
                RY_layer(q_weight[1][0])

            return [qml.expval(qml.PauliZ(j)) for j in range(nwires)]
            
        return circuit

    def initialize_weights(self):
        weights = nn.Parameter(torch.zeros(self.q_depth * 2 * 1 * self.nwires))
        return weights
