import os
import numpy as np
import random
import torch
from collections import Counter

import torchani
from torchani.data import TransformableIterable, IterableAdapterWithLength

from ani_transfer.paths import DATA_DIR


def subtract_self_energies_wo_ref(reenterable_iterable, self_energies, species_order):
    counts = {}
    Y = []
    for n, d in enumerate(reenterable_iterable):
        species = d["species"]
        count = Counter()
        for s in species:
            count[s] += 1
        for s, c in count.items():
            if s not in counts:
                counts[s] = [0] * n
            counts[s].append(c)
        for s in counts:
            if len(counts[s]) != n + 1:
                counts[s].append(0)
        Y.append(d["energies"])
        
    species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))
    shifter = {s: e for s, e in zip(species, self_energies)}
    X = [counts[s] for s in species]

    def reenterable_iterable_factory():
        for d in reenterable_iterable:
            e = 0
            for s in d["species"]:
                e += shifter[s]
            d["energies"] -= e
            yield d
    return IterableAdapterWithLength(reenterable_iterable_factory, n)


def get_data(data_No, species_order, seed=42):
    energy_shifter = torchani.utils.EnergyShifter(None)
    energy_shifter.self_energies = torch.Tensor([-0.6047, -38.0684, -54.7061, -75.1796])

    dspath = DATA_DIR / f"ani_gdb_s0{data_No}.h5"
    data = torchani.data.load(str(dspath))
    data_sub = TransformableIterable(subtract_self_energies_wo_ref(data, energy_shifter.self_energies, species_order))

    fix_seed(seed)
    train, valid, test = data_sub.species_to_indices(species_order).shuffle().split(0.6, 0.2, 0.2)
    return train, valid, test


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
