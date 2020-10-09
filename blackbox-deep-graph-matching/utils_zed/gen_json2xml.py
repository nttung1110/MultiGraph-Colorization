import numpy as np
import os 
import cv2
import json
from PIL import Image
from lxml import etree

path_data_folder = "../data/Geek_data/out_res_hor_data/HOR01/output_res/hor01_020_k_A"
path_anno_folder = os.path.join(path_data_folder, "labels_v1")
path_image_folder = os.path.join(path_data_folder, "color")

root_folder = "../data" 
out_xml = os.path.join(root_folder, "Geek", "annotations", "character")

def read_json(file_name):
    with open(os.path.join(path_anno_folder, file_name+".json"), "r") as f_p:
        data = json.load(f_p)
    
    image = Image.open(os.path.join(path_image_folder, file_name+".tga"))
    w, h = image.size

    shape_list = data["shapes"]

    res = {}
    for each_shape in shape_list:
        label = each_shape["label"]
        x = each_shape["points"][0][0]
        y = each_shape["points"][0][1]
        res[label] = [x, y]
    
    return res, w, h

def gen_xml_file(res, w, h, file_name, out_xml_folder, is_color):
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
        
if __name__ == "__main__":
    for each_json_file in os.listdir(path_anno_folder):
        print("Processing file ", each_json_file)
        file_name = each_json_file[:-5]
        res, w, h = read_json(file_name)
        gen_xml_file(res, w, w, file_name, out_xml)
    