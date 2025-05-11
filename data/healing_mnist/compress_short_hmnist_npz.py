# load the official short HMNIST npz data (downloaded), and then compress them to save storage

import numpy as np
from pathlib import Path


file_name = 'hmnist_temporal_neg.npz'
output_file_name = 'hmnist_temporal_neg_compressed.npz'
if Path(output_file_name).exists():
    raise FileExistsError(f'Output file {output_file_name} already exists')

data_dict = np.load(file_name)
print(f"====== loaded {file_name} ======")
for key in data_dict.keys():
    print(f'{key}: {data_dict[key].shape}')

print("\n====== start compression ====== ")
np.savez_compressed(output_file_name, **data_dict)
print(f"====== compression finished! saved {output_file_name} ======\n")

print("====== check compessed npz file ======")
data_dict_comp = np.load(output_file_name)
for key in data_dict_comp.keys():
    print(f'{key}: {data_dict_comp[key].shape}')

