import json 
from natsort import natsorted

def build_pairs_path(file_name1, file_name2):
    l = [file_name1, file_name2]
    l = natsorted(l)
    return l[0]+"__"+l[1]

def read_matching_info(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return data