from preprocess_hor02 import get_sketch
from PIL import Image
from lxml import etree 
from skimage import measure
from natsort import natsorted

import os
import numpy as np 
import glob 
import shutil
import json
import shutil
import pickle
import cv2
import matplotlib.pyplot as plt

drop_folder = []

def join_two(p1, p2):
    return os.path.join(p1, p2)

def join_three(p1, p2, p3):
    return os.path.join(p1, p2, p3)

def mkdirs(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)


def gen_xml_file(res, w, h, file_name, out_xml_folder, is_color):
    '''
        Generating xml file storing keypoint information
    '''
    annot_el = etree.Element('annotation')

    # You can do this in a loop and keep adding new elements
    # Note: A deepcopy will be required for subsequent items
    image_el = etree.SubElement(annot_el, "image")
    image_el.text = file_name

    type_el = etree.SubElement(annot_el, "type")
    if is_color:
        type_text = "color"
    else:
        type_text = "sketch" 
    type_el.text = type_text

    category_el = etree.SubElement(annot_el, "category")
    category_el.text = "character"

    subcategory_el = etree.SubElement(annot_el, "subcategory")
    subcategory_el.text = "character"

    visible_bounds_el = etree.SubElement(annot_el, "visible_bounds", height = str(h), width = str(w), xmin="0.0", ymin="0.0")

    keypoints_el = etree.SubElement(annot_el, "keypoints")
    
    for element in res:
        keypoints_sub_el = etree.SubElement(keypoints_el, "keypoint", name = element, visible="1", 
                                            x = str(res[element][0]), y = str(res[element][1]), z="0.00")

    
    xml_object = etree.tostring(annot_el,
                                pretty_print=True,
                                xml_declaration=True,
                                encoding='UTF-8')

    with open(os.path.join(out_xml_folder, file_name+".xml"), "wb") as writter:
        writter.write(xml_object)

def read_img(path_img):
    return np.array(Image.open(path_img))

def convert_components(components):
    components_dict = {}
    for idx, component in components.items():
        label = component["label"]
        components_dict[int(label)] = component 
    
    return components_dict

def read_pkl_and_gen_xml(path_pkl, path_annot_out, is_color, folder_name):
    '''
        Read pkl annotation file and perform two task:
            + Copy that pkl file to target data folder for later \
                colorizing (annotations/components)
            + Generate xml file to target data folder for later \
                training (annotations/keypoints)

        Paramerters:
            + path_pkl: path to pickle file
            + path_annot_out: annotations/character
            + is_color: use color or sketch image as training data
    '''
    file_name = path_pkl.split("/")[-1][:-4] # remove extension
    components_vs_mask = pickle.load(open(path_pkl, "rb"), encoding="latin1")
    info_components = components_vs_mask["components"]
    res = {}

    # read image to get size
    image = Image.open(path_pkl[:-4]+".tga")
    w, h = image.size
    
    if len(info_components) <= 3:
        if folder_name not in drop_folder:
            drop_folder.append(folder_name)
        return
    # for idx, component in info_components.items():
    #     centroid = component["centroid"]

    #     x, y = centroid[0], centroid[1]
    #     res[str(idx)] = [int(y), int(x)]
    info_components = convert_components(info_components)
    
    for idx, component in info_components.items():
        centroid = component["centroid"]
        label = component["label"]
        x, y = centroid[0], centroid[1]
        res[str(label)] = [int(y), int(x)]
    
    # copy pkl file first or dump new converted components
    targ_annot_component = join_three(path_annot_out, "components", folder_name)
    mkdirs(targ_annot_component)

    targ_annot_component = join_two(targ_annot_component, file_name+".pkl")

    out_pkl = {}
    out_pkl["mask"] = components_vs_mask["mask"]
    out_pkl["components"] = info_components

    pickle.dump(out_pkl, open(targ_annot_component, "wb+"))
    # copy = shutil.copy(path_pkl, targ_annot_component)

    # then generate xml_file
    targ_annot_kp_root = join_three(path_annot_out, "keypoints", folder_name)
    mkdirs(targ_annot_kp_root)
    gen_xml_file(res, w, h, file_name, targ_annot_kp_root, is_color)
    # access by result["mask"], result["components"]


def prepare_image(path_image_all, path_image_out):
    '''
        Preparing image including both color and sketch(extracted from color) for
        standardized training data
    '''
    num_folder = 0
    num_image = 0

    for folder_dir in natsorted(os.listdir(path_image_all)):
        if not os.path.isdir(join_two(path_image_all, folder_dir)):
            continue 

        if folder_dir in drop_folder:
            continue

        print("Process folder:", folder_dir)
        color_path_in = join_three(path_image_all, folder_dir, "color")

        color_path_out = join_three(path_image_out, "color", folder_dir)
        sketch_path_out = join_three(path_image_out, "sketch", folder_dir)

        mkdirs(color_path_out)
        mkdirs(sketch_path_out)

        allowed_ext = [".tga", ".png"]
        for each_image in os.listdir(color_path_in):
            ext = each_image[-4:]
            if ext not in allowed_ext:
                continue 
            full_path_img = join_two(color_path_in, each_image)
            color_img = read_img(full_path_img)
            out_sketch, out_color = get_sketch(color_img)
            
            shutil.copy(full_path_img, join_two(color_path_out, each_image[:-4]+".png"))
            cv2.imwrite(join_two(sketch_path_out, each_image[:-4]+".png"), out_sketch)

            num_image += 1
        
        num_folder += 1

        if num_folder == 10:
            break

    print("Finish processing " + str(num_folder) +" folders")
    print("Finish processing " + str(num_image) +" images")

def prepare_annot(path_annot_all, path_annot_out):
    '''
        This function performs two tasks:
            + Copy pickle file storing information of components for image into standardized training data
            + Generate keypoints xml file storing information of keypoints for image into standardized training data
    '''
    num_folder = 0

    for folder_dir in natsorted(os.listdir(path_annot_all)):
        if not os.path.isdir(join_two(path_annot_all, folder_dir)):
            continue 

        folder_path = join_three(path_annot_all, folder_dir, "color")
        is_color = True

        for each_annot in os.listdir(folder_path):
            if not each_annot.endswith(".pkl"):
                continue 

            full_path_anno_pkl = join_two(folder_path, each_annot)
            info_img = read_pkl_and_gen_xml(full_path_anno_pkl, path_annot_out, is_color, folder_dir)
            # Randomly choosing color and sketch for training(1:1 ratio)
            # is_color = not is_color
        
        num_folder += 1
        if num_folder == 10:
            break

def prepare_split(data_folder, path_root_out):
    '''
        Prepare split file storing information of which folder should be used for:
            + Training
            + Evaluation
            + Testing
    '''
    out_path_split = join_two(path_root_out, "split")
    mkdirs(out_path_split)

    data = []
    data_folder = join_two(data_folder, "keypoints")

    num_folder = 0
    for folder_name in natsorted(os.listdir(data_folder)):
        if folder_name in drop_folder:
            continue
        print(folder_name)
        folder_list = []
        for file_name in os.listdir(join_two(data_folder, folder_name)):
            folder_list.append(folder_name+"/"+file_name)
        data.append(folder_list)
        
        num_folder += 1
        if num_folder == 10:
            break

    train = np.array(data, dtype=object)    
    test = np.array(data, dtype=object)
    np.savez_compressed(out_path_split+"/geek_pairs", train=train, test=test)

def prepare_matching(path_root_in, path_root_out):
    '''
        Copy json pair matching results from labeled tools into standardized training data
    '''
    path_root_out = join_two(path_root_out, "matching_info")
    mkdirs(path_root_out)

    num_folder = 0

    for folder in natsorted(os.listdir(path_root_in)):
        if not os.path.isdir(join_two(path_root_in, folder)):
            continue 

        if folder in drop_folder:
            continue

        folder_path = join_two(path_root_in, folder)
        annot_matching_path = join_three(folder_path, "PD15_training", "annotations")

        # move annotation to target folder
        des_folder_path = join_two(path_root_out, folder)
        mkdirs(des_folder_path)

        for pairs_fname in os.listdir(annot_matching_path):
            if not pairs_fname.endswith(".json"):
                continue 

            source_file = join_two(annot_matching_path, pairs_fname)
            des_file = join_two(des_folder_path, pairs_fname)
            copy = shutil.copy(source_file, des_file)
        
        num_folder += 1
        if num_folder == 10:
            break

    

if __name__ == "__main__":
    # specify and build path only

    # root path to train data
    path_root_in = "../../stuff/zed_self_prepare" #non-standardized folder
    # path_root_in = "../exp/tmp" #non-standardized folder
    path_root_out = "../data/test_color_only" #standardized folder
    mkdirs(path_root_out)
    #----------------annotation---------
    print("Preparing annotation")
    path_annot_in = path_root_in #same path as image
    path_annot_out = join_two(path_root_out, "annotations")
    mkdirs(path_annot_out)

    path_annot_out = join_two(path_annot_out, "character")
    mkdirs(path_annot_out)
    
    # xml file store keypoint information only(used for loading training data)
    path_annot_xml_kp_out = join_two(path_annot_out, "keypoints")
    mkdirs(path_annot_xml_kp_out)

    # pickle file store component information(used for colorizing process)
    path_annot_pkl_out = join_two(path_annot_out, "components")
    mkdirs(path_annot_pkl_out)

    prepare_annot(path_annot_in, path_annot_out)

    #----------------image--------------
    print("Preparing image")
    path_image_in = path_root_in
    path_image_out = join_two(path_root_out, "image")
    mkdirs(path_image_out)

    sketch_out = join_two(path_image_out, "sketch")
    color_out = join_two(path_image_out, "color")
    
    mkdirs(sketch_out)
    mkdirs(color_out)
    prepare_image(path_image_in, path_image_out)

    #---------------split--------------
    print("Preparing split")
    prepare_split(path_annot_out, path_root_out)

    print("Drop folder:", drop_folder)
    #---------------matching_info----------
    print("Preparing matching_info")
    prepare_matching(path_root_in, path_root_out)

