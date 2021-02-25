from glob import glob
from aux import read_sdf
from mol import Mol

# if __name__ == '__main__':
mols = read_sdf(glob('data/Pub*'))
hashes = np.array([mol.hash(round=1) for mol in mols])



lens = [len(mol) for mol in mols]
mi,ma= min(lens), max(lens)
numbers = set.union(*[set(i.numbers) for i in mols])
numbers = np.array(list(numbers))

k    = np.random.randint(mi,ma+1)
nums = np.random.choice(np.array(numbers), size=k)
xyz  = np.round(np.random.uniform(0, k/2, size=(k,3)), 1)
mol  = Mol(xyz, nums)
mol.dm(round=1)




hashes = np.array([i.hash() for i in mols])
veclayer = get_vectorization_layer(hashes)
