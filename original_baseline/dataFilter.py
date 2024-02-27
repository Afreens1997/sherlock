import json
import os
import sys

if __name__ =="__main__":
    MODE = sys.argv[1]
    if MODE == 'train':
        FILE_PATH = '../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/sherlock_train_v1_1.json'
        OUT_PATH = '../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/train.json'
    elif MODE == 'val':
        FILE_PATH = '../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/sherlock_val_with_split_idxs_v1_1.json'
        OUT_PATH = '../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/val.json'

    with open(FILE_PATH, 'r') as file:
        data = json.load(file)

    print(len(data))
    updated_data = []
    for idx in data:
        if 'vcr1images' not in idx['inputs']['image']['url']:
            updated_data.append(idx)
    print(len(updated_data))

    with open(OUT_PATH, "w") as out_file:
        json.dump(updated_data, out_file)
    
    print(f'{(len(updated_data) / len(data))*100} of data retained in {MODE}')