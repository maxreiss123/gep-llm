from tqdm import *

from geppy_reg import *

file_name = 'airfoil.csv'

unit_map = {
    'x1': np.array([0, 0, -1, 0, 0, 0, 0]),
    'x2': np.zeros(7),
    'x3': np.array([0, 1, 0, 0, 0, 0, 0]),
    'x4': np.array([0, 1, -1, 0, 0, 0, 0]),
    'x5': np.array([0, 1, 0, 0, 0, 0, 0])
}

ngens_ = 100
for seed in tqdm(range(0, 25, 1)):
    for mem_ in [True,False]:
        for k, [v1, v2] in target_airfoil_UCI.items():
            log_book = run(seed, v2, k, pop_size_=100, ngens_=ngens_,
                           mem=mem_, file_name=file_name, prop=0.9,
                           log_genes=True, primitives=v1, prop_mem=0.1,
                           log_name='gene1_target.csv', unit_maps=unit_map)
