import time
import cv2
import sys
import torch
import numpy as np
import os

sys.path.append("../"+os.path.join(os.path.curdir))

from utils.config import cfg
from torchvision import transforms
from PIL import Image
from BB_GM.model_geek import Net
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg
from utils.utils import lexico_iter
from utils.build_graphs import build_graphs
from torch_geometric.data import Data, Batch



                                
sys.path.append("/mnt/ai_filestore/home/zed/multi-graph-matching/hades_painting")
from rules.color_component_matching import ComponentWrapper, get_component_color

'''
    This scripts is used for inferencing(including both matching and colorizing) for
    a specific folder following the below format:
    + folder_name
        * sketch: store sketch images
        * color: store color images as reference for colorizing sketch folder
'''      

def j_2(path1, path2):
    return os.path.join(path1, path2)


class DataProcessor():
    def __init__(self, folder_path):
        '''
            Parameters:
                + folder_path: path to folder_name as above
        '''
        self.num_sketch = 1
        self.sketch_folder = j_2(folder_path, "sketch")
        self.color_folder = j_2(folder_path, "color")

    def read_img(self, path, type_image):
        '''
            Read img and return two things:
                + tyler_image: input for tyler model
                + zed_image: input for zed model
        '''
        with Image.open(str(path)) as img:

            if type_image == "color":
                tyler_input_img = img
                tyler_input_img = tyler_input_img.convert("RGB")
                tyler_input_img = cv2.cvtColor(np.array(tyler_input_img), cv2.COLOR_RGB2BGR)

            elif type_image == "sketch":
                tyler_input_img = np.array(img)

            ori_sizes = img
            
            zed_img = img.resize((256, 256), resample=Image.BICUBIC, box=(0, 0, 256, 256))
        return tyler_input_img, zed_img

    def get_points(self, anno_list):
        '''
            This function performs 2 tasks:
                + Extracting components of each image in list_img
                + Use keypoints to represent components 
            Parameters:
                anno_list
            Returned:
                +anno_list:
        '''
        # get points
        component_wrapper = ComponentWrapper()

        for annot in anno_list:
            type_img = annot["type_name"]
            img = annot["tyler_image"]

            kp_list = []
            if type_img == "color":
                mask, components = component_wrapper.process(img, None, "extract_color")
                get_component_color(components, img)

            elif type_img == "sketch":
                mask, components = component_wrapper.process(None, img, "extract_sketch")

            components_tmp = {}
            for component in components:
                centroid = component["centroid"]
                name = component["label"]
                components_tmp[int(name)] = component
                x = centroid[0]
                y = centroid[1]
                kp_list.append({"x": x, "y": y, "name": name})
            
            annot["keypoints"] = kp_list
            annot["components"] = components_tmp

        return anno_list

        # convert to cuda

        # # build graphs
        # graph_list = []
        # for p_gt, n_p_gt in zip(p_list, n_p):
        #     edge_indices, edge_features = build_graphs(p_gt, n_p_gt)

        #     # Add dummy node features so the __slices__ of them is saved when creating a batch
        #     pos = torch.tensor(p_gt).to(torch.float32) / 256.0
        #     assert (pos > -1e-5).all(), p_gt
        #     graph = Data(
        #         edge_attr=torch.tensor(edge_features).to(torch.float32),
        #         edge_index=torch.tensor(edge_indices, dtype=torch.long),
        #         x=pos,
        #         pos=pos,
        #     )
        #     graph.num_nodes = n_p_gt
        #     graph_list.append(graph)

        # # to cuda
        # p_list = [torch.Tensor(_).cuda() for _ in p_list]
        # n_p = [torch.Tensor(_).cuda() for _ in n_p]
        # edges = [_.to("cuda") for _ in graph_list]

        # return p_list, n_p, graph_list

    def prepare_all(self):
        print("Preparing data")
        #-----image, image_names, folder_names-----
        anno_list = []
        zed_img_list = []
        tyler_img_list = []
        image_names = []
        folder_names = []
        types = []
        perm_mat = None


        # read sketch img first 
        sketch_file = os.listdir(self.sketch_folder)[0]
        tyler_sketch, zed_sketch = self.read_img(j_2(self.sketch_folder, sketch_file), "sketch")

        zed_img_list.append(zed_sketch)
        tyler_img_list.append(tyler_sketch)
        image_names.append(sketch_file[:-4])
        folder_names.append(None)
        types.append("sketch")

        annot = {}
        annot["image"] = zed_sketch
        annot["tyler_image"] = tyler_sketch
        annot["image_name"] = sketch_file[:-4]
        annot["type_name"] = "sketch"
        annot["folder_name"] = None
        
        anno_list.append(annot)
        # read color imgs
        for color_file in os.listdir(self.color_folder):
            tyler_color, zed_color = self.read_img(j_2(self.color_folder, color_file), "color")
        
            annot = {}
            annot["image"] = zed_color
            annot["tyler_image"] = tyler_color
            annot["image_name"] = color_file[:-4]
            annot["type_name"] = "color"
            annot["folder_name"] = None

            anno_list.append(annot)
        
        # if zed_img_list[0] is not None:
        #     trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
        #     zed_img_list = [trans(img).unsqueeze(0) for img in zed_img_list]

        # zed_img_list = [_.cuda() for _ in zed_img_list]
        # get points, n_points, graphs 
        anno_list = self.get_points(anno_list)

        return anno_list
        # return zed_img_list, p_list, graph_list, n_p, perm_mat, types, tyler_img_list, folder_names, image_names
        
    
if __name__ == "__main__":
    # load model and checkpoint
    # print("Loading model")
    # model = Net()
    # model = model.cuda()

    # path_checkpoint = "./results/geek_alpha_test_1/params/0007"
    # params_path = os.path.join(path_checkpoint, "params.pt")
    # model.load_state_dict(torch.load(params_path))

    # process data
    data_folder = "./test_inference"
    processor = DataProcessor(data_folder)

    print(processor.prepare_all())
    # data_list, \
    # points_gt, \
    # edges, \
    # n_points_gt, \
    # perm_mat_list, \
    # types, \
    # tyler_imgs, \
    # folder_names, \
    # image_names = processor.prepare_all()
    print(edges)
    print("Running inference")
    with torch.set_grad_enabled(False):
        s_pred_list = model(
            data_list,
            points_gt,
            [edges],
            n_points_gt,
            perm_mat_list,
            types,
            tyler_imgs,
            folder_names,
            image_names)
    print(s_pred_list)

    









# def predict_color_sketch(s_pred_list, image_names, types, kp_label_name, folder_names):
#     '''
#         s_pred_list: combinatorial matching pairs containing (n-1)! pairs with n is the number
#         of graph for matching
#     '''
#     # print(len(s_pred_list))
#     triplet_im_name = [name for name in image_names]
#     triplet_kp_name = [kp_name for kp_name in kp_label_name]

#     image_names_cb = build_binary_combination(image_names)
#     types_cb = build_binary_combination(types)
#     kp_label_name_cb = build_binary_combination(kp_label_name)

#     triplet_matching = []
#     folder_name = folder_names[0][0]
#     for idx, each_pair in enumerate(s_pred_list):
        
#         triplet_matching.append(each_pair)

#         images_pair = image_names_cb[idx]
        
#         types_pair = types_cb[idx]
#         kp_label_name_pair = kp_label_name_cb[idx]

#         if types_pair[0][0] == types_pair[1][0]:
#             # refuse to transfer color for sketch-sketch or color-color
#             continue

#         i_name_1, i_name_2 = images_pair[0], images_pair[1]
#         type_1, type_2 = types_pair[0][0], types_pair[1][0]
#         kp_name_1, kp_name_2 = kp_label_name_pair[0], kp_label_name_pair[1]
#         each_pair = each_pair[0]
#         if type_1 == "color" and type_2 == "sketch":
#             i_name_1, i_name_2 = images_pair[1], images_pair[0]
#             type_1, type_2 = types_pair[1][0], types_pair[0][0]
#             kp_name_1, kp_name_2 = kp_label_name_pair[1], kp_label_name_pair[0]
#             # Transpose permutation matrix to become sketch->color mapping
#             # each_pair = np.array()
#             each_pair = each_pair.T
#             each_pair = each_pair.tolist()


#         vis_color_sketch_transfer(each_pair, 
#                                 (i_name_1, i_name_2),
#                                 (kp_name_1, kp_name_2),
#                                 folder_name)
        
#         print("On colorizing:", i_name_1, i_name_2)

    

