import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import sys 
import os
import random
from utils.build_graphs import build_graphs

from utils.config import cfg
from torch_geometric.data import Data, Batch

sys.path.append("../"+os.path.join(os.path.curdir))

class TestDataset(Dataset):
    def __init__(self, length, data_folder, processor):
        self.data_folder = data_folder
        self.length = length
        self.processor = processor
        self.num_graphs_in_matching_instance = None

    def set_num_graphs(self, num_graphs_in_matching_instance):
        self.num_graphs_in_matching_instance = num_graphs_in_matching_instance

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.num_graphs_in_matching_instance is None:
            raise ValueError("Num_graphs has to be set to an integer value.")

        idx = idx 

        anno_list = self.processor.prepare_all()


        points_gt = [np.array([(kp["x"], kp["y"]) for kp in anno_dict["keypoints"]]) for anno_dict in anno_list]
        n_points_gt = [len(p_gt) for p_gt in points_gt]
        # Geek
        # print(anno_list)
        image_names = [anno_dict["image_name"] for anno_dict in anno_list]
        types = [anno_dict["type_name"] for anno_dict in anno_list]
        kp_label_name = [[kp["name"] for kp in anno_dict["keypoints"]] for anno_dict in anno_list]
        folder_name = [anno_dict["folder_name"] for anno_dict in anno_list]
        components = [anno_dict["components"] for anno_dict in anno_list]

        graph_list = []
        for p_gt, n_p_gt in zip(points_gt, n_points_gt):
            edge_indices, edge_features = build_graphs(p_gt, n_p_gt)

            # Add dummy node features so the __slices__ of them is saved when creating a batch
            pos = torch.tensor(p_gt).to(torch.float32) / 256.0
            assert (pos > -1e-5).all(), p_gt
            graph = Data(
                edge_attr=torch.tensor(edge_features).to(torch.float32),
                edge_index=torch.tensor(edge_indices, dtype=torch.long),
                x=pos,
                pos=pos,
            )
            graph.num_nodes = n_p_gt
            graph_list.append(graph)

        # print(graph_list)
        ret_dict = {
            "Ps": [torch.Tensor(x) for x in points_gt],
            "ns": [torch.tensor(x) for x in n_points_gt],
            #geek
            "image_names": image_names,
            "types": types,
            "kp_label_name": kp_label_name,
            "folder_names": folder_name,
            "gt_perm_mat": None,
            "edges": graph_list,
            "components": components
        }

        imgs = [anno["image"] for anno in anno_list]
        tyler_imgs = [anno["tyler_image"] for anno in anno_list]
        if imgs[0] is not None:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
            imgs = [trans(img) for img in imgs]
            ret_dict["images"] = imgs
            ret_dict["tyler_imgs"] = tyler_imgs
        elif "feat" in anno_list[0]["keypoints"][0]:
            feat_list = [np.stack([kp["feat"] for kp in anno_dict["keypoints"]], axis=-1) for anno_dict in anno_list]
            ret_dict["features"] = [torch.Tensor(x) for x in feat_list]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str or type(inp[0]) == int or inp[0] == None:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            ret = inp
        return ret

    ret = stack(data)
    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, bs, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
    )
