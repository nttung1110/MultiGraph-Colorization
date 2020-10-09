#!/usr/bin/python
import numpy as np 
import cv2 
import os
import json
import shutil
import sys
import argparse
from PIL import Image
from gen_json2xml import gen_xml_file


def read_component_saving(file_name, mode, cfg):


    if mode == "extract_color":
        path_anno = os.path.join(cfg.path_anno_component_folder, "color")
        path_image_folder = cfg.path_image_folder_color
    else:
        path_anno = os.path.join(cfg.path_anno_component_folder, "sketch")
        path_image_folder = cfg.path_image_folder_sketch

    with open(os.path.join(path_anno, file_name+".json"), "r") as f_p:
        data = json.load(f_p)
    
    image = Image.open(os.path.join(path_image_folder, file_name+".png"))
    w, h = image.size

    res = {}

    for component_key in data:
        component = data[component_key]

        #get centroid as keypoint
        centroid= data[component_key]["centroid"]
        x = centroid[0]
        y = centroid[1]

        res[component_key] = [y, x]

    return res, w, h

def mkdirs(path):
    if os.path.isdir(path) == False:
        os.mkdir(path) 

def transform_image(img_name, is_transform, is_color, cfg):

    targ_img_root = os.path.join(cfg.geek_root_target_component_folder, "color")

    mkdirs(targ_img_root)

    if is_color:
        img = Image.open(os.path.join(cfg.path_image_folder_color, img_name))
    else:
        img = Image.open(os.path.join(cfg.path_image_folder_sketch, img_name))

    if is_transform:
        transform_img = img.rotate(45, expand=True)
    else:
        transform_img = img

    transform_img.save(os.path.join(targ_img_root, img_name))
    transform_img.save(os.path.join(targ_img_root, img_name[:-4]+".png"))

def transform_annot_component(file_name, is_color, cfg):
    targ_annot_component_root = os.path.join(cfg.geek_root_target_component_folder, "annotations")

    mkdirs(targ_annot_component_root)

    targ_annot_kp_root = os.path.join(targ_annot_component_root, "character")

    mkdirs(targ_annot_kp_root)

    if is_color:
        mode = "extract_color"
    else:
        mode = "extract_sketch"

    res, w, h = read_component_saving(file_name, mode, cfg)
    gen_xml_file(res, w, h, file_name, targ_annot_kp_root, is_color)

def gen_split_component(cfg):
    data_folder = os.path.join(cfg.geek_root_target_component_folder, "annotations")
    out_path_split = os.path.join(cfg.geek_root_target_component_folder, "split")

    mkdirs(out_path_split)
    data = []
    for folder_name in os.listdir(data_folder):
        for file_name in os.listdir(os.path.join(data_folder, folder_name)):
            data.append(folder_name+"/"+file_name)

    train = np.array([data], dtype=object)    
    test = np.array([data], dtype=object)
    np.savez_compressed(out_path_split+"/geek_pairs", train=train, test=test)


def transform(cfg):

    is_transform = False
    is_color = False

    for img_name in os.listdir(cfg.path_image_folder_color):
        print("File name:", img_name[:-4])
        
        transform_image(img_name, is_transform, is_color, cfg)
        
        if is_color:
            path_image_folder = cfg.path_image_folder_color

        else:
            path_image_folder = cfg.path_image_folder_sketch

        # transform_annot(img_name[:-4], is_transform, path_image_folder, is_color)
        transform_annot_component(img_name[:-4], is_color, cfg)
        is_color = not is_color
        # is_transform = not is_transform

    gen_split_component(cfg)
    # print("Generating splitted data:")
    # transform_split()

def build_cfg(folder_name):
    class CFG:
        def __init__(self):
            pass 

    cfg = CFG()
    # cfg["root_folder"] = ""
    # cfg["path_anno_component_folder"] = ""
    # cfg["geek_root_target_component_folder"] = ""
    # cfg["root_data_folder"] = ""
    # cfg["path_image_folder_sketch"] = ""
    # cfg["cfg.path_image_folder_color"] = ""
    # cfg["targ_img_root"] = ""

    cfg.root_folder = "/mnt/ai_filestore/home/zed/multi-graph-matching"
    # path_ori_data_folder = "../data/Geek_data/out_res_hor_data/HOR02/output_res_hor02/hor02_031"
    # path_anno_folder = os.path.join(path_ori_data_folder, "labels_v1")

    cfg.path_anno_component_folder = os.path.join(cfg.root_folder, 
                                                "components_saving",
                                                "HOR02", 
                                                folder_name)

                    

    cfg.geek_root_target_component_folder = os.path.join(cfg.root_folder,
                                        "blackbox-deep-graph-matching","data",
                                        "Geek_components_matching_"+folder_name)

    cfg.root_data_folder = os.path.join(cfg.root_folder, "Geek_data", "processed_HOR02")

    cfg.path_image_folder_sketch = os.path.join(cfg.root_data_folder, folder_name, "sketch")
    cfg.path_image_folder_color = os.path.join(cfg.root_data_folder, folder_name, "color")

    cfg.targ_img_root = os.path.join(cfg.geek_root_target_component_folder, "color")

    return cfg


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--folder_name', type=str,
                        help='folder')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # print(sys.argv)
    # folder_name = sys.argv[1]

    chosen_folder = ["hor02_016_019",
                    "hor02_025",
                    "hor02_029",
                    "hor02_033_072",
                    "hor02_034",
                    "hor02_040"]

    for folder_name in chosen_folder:
        cfg = build_cfg(folder_name)

        mkdirs(cfg.geek_root_target_component_folder)
        print("Transform Geek:", folder_name)
        transform(cfg)






# def read_json_rotate_keypoint(file_name, path_image_folder, is_transform):
#     with open(os.path.join(path_anno_folder, file_name+".json"), "r") as f_p:
#         data = json.load(f_p)
    
#     image = Image.open(os.path.join(path_image_folder, file_name+".tga"))
#     w, h = image.size

#     shape_list = data["shapes"]

#     res = {}
#     for each_shape in shape_list:
#         label = each_shape["label"]
#         if is_transform:
#             x = w - each_shape["points"][0][0]
#             y = h - each_shape["points"][0][1]
#         else:
#             x = each_shape["points"][0][0]
#             y = each_shape["points"][0][1]
#         res[label] = [x, y]
    
#     return res, w, h

# def transform_annot(file_name, is_transform, path_image_folder):
#     targ_annot_root = os.path.join(geek_root_target_folder, "annotations")

#     mkdirs(targ_annot_root)

#     targ_annot_kp_root = os.path.join(targ_annot_root, "character")

#     mkdirs(targ_annot_kp_root)

    
#     res, w, h = read_json_rotate_keypoint(file_name, path_image_folder, is_transform)
#     gen_xml_file(res, w, h, file_name, targ_annot_kp_root)

# def gen_split(cfg):
#     data_folder = os.path.join(geek_root_target_folder, "annotations")
#     out_path_split = os.path.join(geek_root_target_folder, "split")

#     mkdirs(out_path_split)
#     data = []
#     for folder_name in os.listdir(data_folder):
#         for file_name in os.listdir(os.path.join(data_folder, folder_name)):
#             data.append(folder_name+"/"+file_name)

#     train = np.array([data], dtype=object)    
#     test = np.array([data], dtype=object)
#     np.savez_compressed(out_path_split+"/geek_pairs", train=train, test=test)