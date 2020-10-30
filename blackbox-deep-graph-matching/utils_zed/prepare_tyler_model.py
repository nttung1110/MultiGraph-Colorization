import sys
import os
import argparse
import glob
import numpy as np
import pickle
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from natsort import natsorted
from PIL import Image
from utils.config import cfg


from hades_painting.rules.color_component_matching import ComponentWrapper, get_component_color
from hades_painting.rules.component_wrapper import get_moment_features_v1, resize_mask, build_neighbor_graph, get_moment_features
from hades_painting.models.shallow_net import UNet
from utils_zed.config_zed import cfg_zed

def get_input_data_v1(folder_name, image_name):
    '''
        Get component extracted and mask from annotation file to input for tyler model
        Train mode
    '''
    features_list, masks_list, components_list = [], [], []
    for each_fol, each_img in zip(folder_name, image_name):
        path_pkl = os.path.join(cfg.Geek.ROOT_DIR,"annotations",
                            "character","components",
                            each_fol,
                            each_img+".pkl")

        size = (768, 512)
        # color_image = read_image(path)
        # print(size)
        mean = np.array([2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]).reshape(8, 1)
        std = np.array([0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]).reshape(8, 1)
        
        info_components = pickle.load(open(path_pkl, "rb"), fix_imports=True, encoding="latin1")
        mask, components = info_components["mask"], info_components["components"]
        # print(components["image"])
        # import pdb
        # pdb.set_trace()
        mask = resize_mask(mask, components, size).astype(np.int32)

        # print(mask.shape)
        # get features
        
        features = get_moment_features_v1(components, mask)
        for i in range(8):
            features[i] = (features[i] - mean[i]) / std[i]

        features = torch.tensor(features).float().unsqueeze(0)
        mask = torch.tensor(mask).long().unsqueeze(0)
        
        features_list.append(features)
        masks_list.append(mask)
        components_list.append(components)

    return features_list, masks_list, components_list
    

def get_input_data(color_images, type_imgs, image_names, type_extract):
    '''
        Inference mode
    '''
    features_list, masks_list, components_list = [], [], []
    for color_image, type_img, image_name in zip(color_images, type_imgs, image_names):
        component_wrapper = ComponentWrapper()

        size = (768, 512)
        # color_image = read_image(path)
        # print(size)
        mean = np.array([2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]).reshape(8, 1)
        std = np.array([0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]).reshape(8, 1)

        # print(color_image.shape)

        # get by file
        if type_extract == "pkl":
            path_pkl = os.path.join(cfg_zed.data_folder, "annot", type_img, image_name+".pkl")
            info = pickle.load(open(path_pkl, "rb"), fix_imports=True, encoding="latin1")
            mask, components = info["mask"], info["components"]

        # get by model
        elif type_extract == "no pkl":
            if type_img == "color":
                mask, components = component_wrapper.process(color_image, None, "extract_color")

            elif type_img == "sketch":
                mask, components = component_wrapper.process(None, color_image, "extract_sketch")

        mask = resize_mask(mask, components, size).astype(np.int32)

        # print(mask.shape)
        # get features
        features = get_moment_features(components, mask)

        for i in range(8):
            features[i] = (features[i] - mean[i]) / std[i]

        features = torch.tensor(features).float().unsqueeze(0)
        mask = torch.tensor(mask).long().unsqueeze(0)

        if type_img == "color":
            get_component_color(components, color_image)
        
        features_list.append(features)
        masks_list.append(mask)
        components_list.append(components)

    return features_list, masks_list, components_list