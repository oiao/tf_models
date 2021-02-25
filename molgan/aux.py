from typing import List
from mol import Mol

def read_sdf(fpaths:List[str], exclude_h=True) -> List[Mol]:
    """
    Read pubchem sdf from `fpaths`, return a list of mol.Mol
    """
    import gzip
    from io import StringIO
    from ase import io as aseio, Atoms

    atoms = []
    for fpath in fpaths:
        with gzip.open(fpath) as f:
            data = f.read()
        data = data.decode('utf-8')
        data = data.split('$$$$\n')[:-1]
        for d in data:
            io = StringIO(d+'M  END\n')
            try:
                atom = aseio.read(io, format='sdf')
                if exclude_h:
                    numbers = atom.numbers[atom.numbers != 1]
                    coords  = atom.positions[atom.numbers != 1]
                    atom    = Atoms(numbers=numbers, positions=coords)
                atom.name = int(d.split('\n')[0])
                atoms.append(atom)
            except KeyError:
                continue
    return [Mol(i.positions, i.numbers, name=i.name) for i in atoms]
