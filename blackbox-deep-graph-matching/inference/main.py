from geek_infer import *
from dataloader import *

from utils_zed.config_zed import cfg_zed
import sys
import torch
import numpy as np
import os

sys.path.append("/mnt/ai_filestore/home/zed/multi-graph-matching/blackbox-deep-graph-matching")

from transfer_color import vis_color_sketch_transfer, build_binary_combination
def predict_color_sketch(s_pred_list, inputs):
    '''
        s_pred_list: combinatorial matching pairs containing (n-1)! pairs with n is the number
        of graph for matching
    '''

    image_names = inputs["image_names"]
    types = inputs["types"]
    kp_label_name = inputs["kp_label_name"]
    folder_name = inputs["folder_names"][0][0]
    image_names = inputs["image_names"]
    components = inputs["components"]

    triplet_im_name = [name for name in image_names]
    triplet_kp_name = [kp_name for kp_name in kp_label_name]

    image_names_cb = build_binary_combination(image_names)
    types_cb = build_binary_combination(types)
    kp_label_name_cb = build_binary_combination(kp_label_name)

    triplet_matching = []
    for idx, each_pair in enumerate(s_pred_list):
        
        triplet_matching.append(each_pair)

        images_pair = image_names_cb[idx]
        
        types_pair = types_cb[idx]
        kp_label_name_pair = kp_label_name_cb[idx]
        components_pair = components[idx]

        if types_pair[0][0] == types_pair[1][0]:
            # refuse to transfer color for sketch-sketch or color-color
            continue

        i_name_1, i_name_2 = images_pair[0], images_pair[1]
        type_1, type_2 = types_pair[0][0], types_pair[1][0]
        kp_name_1, kp_name_2 = kp_label_name_pair[0], kp_label_name_pair[1]
        each_pair = each_pair[0]
        components_1, components_2 = components[0], components[1]

        if type_1 == "color" and type_2 == "sketch":
            i_name_1, i_name_2 = images_pair[1], images_pair[0]
            type_1, type_2 = types_pair[1][0], types_pair[0][0]
            kp_name_1, kp_name_2 = kp_label_name_pair[1], kp_label_name_pair[0]
            components_1, components_2 = components[1], components[0]
            # Transpose permutation matrix to become sketch->color mapping
            # each_pair = np.array()
            each_pair = each_pair.T
            each_pair = each_pair.tolist()


        vis_color_sketch_transfer(each_pair, 
                                (i_name_1, i_name_2),
                                (kp_name_1, kp_name_2),
                                (components_1, components_2))
        
        print("On colorizing:", i_name_1, i_name_2)

if __name__ == "__main__":
    # load model and checkpoint
    print("Loading model")
    model = Net()
    model = model.cuda()

    path_checkpoint = "../results/geek_alpha_test_1/params/0007"
    params_path = os.path.join(path_checkpoint, "params.pt")
    model.load_state_dict(torch.load(params_path))

    # process data
    # data_folder = "./test_inference"
    processor = DataProcessor(cfg_zed.data_folder)

    len_test_data = 2
    batch_size = 1
    num_graph_matching = 2

    my_data = TestDataset(len_test_data, cfg_zed.data_folder, processor)
    my_data = get_dataloader(my_data, batch_size)

    my_data.dataset.set_num_graphs(num_graph_matching)

    for k, inputs in enumerate(my_data):
        data_list = [_.cuda() for _ in inputs["images"]]
        tyler_imgs = inputs["tyler_imgs"]
        points_gt = [_.cuda() for _ in inputs["Ps"]]
        n_points_gt = [_.cuda() for _ in inputs["ns"]]
        # print("Num points:", n_points_gt)
        # print(inputs["edges"])
        edges = [_.to("cuda") for _ in inputs["edges"]]

        perm_mat_list = [perm_mat for perm_mat in inputs["gt_perm_mat"]]

        # geek
        image_names = inputs["image_names"]
        # print(image_names)
        types = inputs["types"]
        # print(types)
        kp_label_name = inputs["kp_label_name"]
        folder_names = inputs["folder_names"]
        image_names = inputs["image_names"]

        with torch.set_grad_enabled(False):
            s_pred_list = model(
                data_list,
                points_gt,
                edges,
                n_points_gt,
                perm_mat_list,
                types,
                tyler_imgs,
                folder_names,
                image_names,
                is_training=False
            )
        
        predict_color_sketch(s_pred_list, inputs)
        
    
    