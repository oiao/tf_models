import numpy as np
import hashlib

class Mol:
    def __init__(self, xyz, numbers, name=None):
        assert len(numbers) == len(xyz)
        self.numbers = np.array(numbers)
        self.xyz     = np.array(xyz)
        assert self.xyz.ndim == 2
        assert self.xyz.shape[-1] == 3
        self.name = name

    def __len__(self):
        return len(self.xyz)

    def __str__(self):
        s = ''
        for num,coords in zip(self.numbers, self.xyz):
            _  = ''.join(f"{i:>10.6f}" for i in coords)
            s += f"{num:<4}{_}\n"
        return s



    def dm(self, round=None) -> np.ndarray:
        """ Return the distance matrix of self """
        dm = np.sum((np.expand_dims(self.xyz,1)-np.expand_dims(self.xyz,0))**2, -1)**0.5
        if round is not None:
            dm = np.round(dm, round)
        return dm

    def hash(self, hashf='blake2b', round=None) -> str:
        assert hashf in hashlib.algorithms_available, f"{hashf} is not known to hashlib. Choose from {hashlib.algorithms_available}."
        hashf = getattr(hashlib, hashf)
        b = self.numbers.tobytes() + self.dm(round).tobytes()
        return hashf(b).hexdigest()
