#!/usr/bin/python
import numpy as np 
import cv2
import sys 
import os 
import sys
import json
import argparse
from PIL import Image
sys.path.append("/mnt/ai_filestore/home/zed/multi-graph-matching/hades_painting")

from rules.component_wrapper import ComponentWrapper, get_component_color

component_save_folder_all = "/mnt/ai_filestore/home/zed/multi-graph-matching/components_saving/HOR02"
def mkdirs(path):
    if os.path.isdir(path) == False:
        os.mkdir(path) 

def save_component_to_json(image, json_img_path, mode_extract):
    component_wrapper = ComponentWrapper()
    if mode_extract == "extract_color":
        input_label, input_components = component_wrapper.process(image, None, mode_extract)
        get_component_color(input_components, image, mode_extract)

    elif mode_extract == "extract_sketch":
        input_label, input_components = component_wrapper.process(None, image, mode_extract)
    # add color value to components
    # if mode_extract == "extract_color":

    idx = 1
    res = {}
    for component in input_components:
        for key, value in component.items():
            if isinstance(value, int) == False and isinstance(value, tuple) == False and isinstance(value, list) == False:
                # print(value)
                component[key] = value.tolist()
        
        res[str(idx)] = component
        idx += 1

    with open(json_img_path, "w") as fp:
        json.dump(res, fp, indent=4)

def process_type(folder_path, mode):
    component_save_folder = os.path.join(component_save_folder_all, folder_path.split("/")[-2])

    mkdirs(component_save_folder)
    component_save_folder = os.path.join(component_save_folder, folder_path.split("/")[-1])
    mkdirs(component_save_folder)

    for image_name in os.listdir(folder_path):
        
        if image_name[-4] == ".jpg":
            continue
        print(image_name)
        img_path = os.path.join(folder_path, image_name)

        if mode == "extract_color":
            image = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
            # image = cv2.imread(img_path)
        else:
            image = np.array(Image.open(img_path))
            # image = cv2.imread(img_path)
        json_img_path = os.path.join(component_save_folder, image_name[:-4]+".json")

        save_component_to_json(image, json_img_path, mode)

def process_all_HOR02():
    data_path = "/mnt/ai_filestore/home/zed/multi-graph-matching/Geek_data/processed_HOR02"
    chosen_folder = ["hor02_016_019",
                    "hor02_025",
                    "hor02_029",
                    "hor02_033_072",
                    "hor02_034",
                    "hor02_040"]
    for each_folder in os.listdir(data_path):
        
        full_sub_path = os.path.join(data_path, each_folder)
        if os.path.isdir(full_sub_path) == False:
            continue

        if each_folder not in chosen_folder:
            continue
        
        print("Processing folder:", each_folder)
        color_folder = os.path.join(full_sub_path, "color")
        sketch_folder = os.path.join(full_sub_path, "sketch")

        print("Process color")
        process_type(color_folder, "extract_color")
        print("Process sketch")
        process_type(sketch_folder, "extract_sketch")

def process_folder(folder_name):
    data_path = "/mnt/ai_filestore/home/zed/multi-graph-matching/Geek_data/processed_HOR02/"
    # data_path = "/mnt/ai_filestore/home/zed/multi-graph-matching/Geek_data/HOR02_Full_Formated"
    full_sub_path = os.path.join(data_path, folder_name)

    color_folder = os.path.join(full_sub_path, "color")
    sketch_folder = os.path.join(full_sub_path, "sketch")

    print("Process color")
    process_type(color_folder, "extract_color")
    print("Process sketch")
    process_type(sketch_folder, "extract_sketch")


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--folder_name', type=str,
                        help='folder')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # folder_name = sys.argv[1]
    
    # print("Generate kp from component on folder of HOR02:", folder_name)
    # process_folder(folder_name)
    process_all_HOR02()