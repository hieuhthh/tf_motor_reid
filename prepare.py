import os
from glob import glob
import pandas as pd
import numpy as np

route = 'unzip/MoRe_Dataset'

def get_paths_ids_more_dataset(filename='train_files.txt'):
    paths = []
    ids = []
    cams = []
    txt_path = os.path.join(route, filename)
    with open(txt_path, "r") as f:
        data = f.read().splitlines()
        for line in data:
            full_line = os.path.join(route, line)
            name = line.split('/')[-1]
            get_id = int(name.split('_')[2])
            if not os.path.exists(full_line):
                print(full_line)
            else:
                paths.append(full_line)
                ids.append(get_id)
                cams.append(name[3])
    return paths, ids, cams

if __name__ == '__main__':
    train_paths, train_ids, train_cams = get_paths_ids_more_dataset('train_files.txt')
    # print(train_paths)
    # print(train_ids)
    # print(train_cams)

    test_paths, test_ids, test_cams = get_paths_ids_more_dataset('test_files.txt')
    # print(test_paths)
    # print(test_ids)
    # print(test_cams)