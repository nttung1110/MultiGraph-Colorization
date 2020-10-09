import numpy as np
import os
root = "../data/Geek"
data = os.path.join(root_folder, "annotations")
out_path_split = os.path.join(root_folder, "split")

def gen_split(root_folder, data_folder, out_split):
    data = []
    for folder_name in os.listdir(data_folder):
        for file_name in os.listdir(os.path.join(data_folder, folder_name)):
            data.append(folder_name+"/"+file_name)

    train = np.array([data], dtype=object)    
    test = np.array([data], dtype=object)
    np.savez_compressed(out_split+"/geek_pairs", train=train, test=test)

if __name__ == "__main__":
    gen_split(root, data, out_path_split)
