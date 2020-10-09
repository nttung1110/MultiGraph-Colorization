import numpy as np 
import cv2
import json 
from itertools import combinations
import os 
import sys 
import pickle
sys.path.append("../")
from utils_zed.config_zed import cfg_zed

def put_color(reference_color, sketch_img, list_position):
    reference_color = (reference_color[0][0], reference_color[1][0], reference_color[2][0])
    for [x, y] in list_position:
        sketch_img[x][y] = reference_color
    return sketch_img

def read_img_pairs(im_name_sketch, im_name_color):
    path_color = os.path.join(cfg_zed.data_folder, "color", im_name_color+".png")
    path_sketch = os.path.join(cfg_zed.data_folder, "sketch", im_name_sketch+".png")


    color_img = cv2.imread(path_color)
    sketch_img = cv2.imread(path_sketch)

    return sketch_img, color_img
    
def get_mask(region, input_image):
    mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)
    mask[region[:, 0], region[:, 1]] = 255

    return mask

def process_list_point(list_point):
    list_point = list_point[0]
    list_point = np.array(list_point)
    list_point = list_point[:, :2]
    return list_point

def mkdirs(path_in):
    if os.path.isdir(path_in) is False:
        os.mkdir(path_in)

def debug_component_matching(mask_c, mask_s, c_kp_id, s_kp_id, c_img_name, s_img_name, img_c, img_s, p_c, p_s):
    # mkdirs(path_save_vis_transfer_color)
    # mkdirs(path_debug_pair_matching)

    path_save_vis_transfer_color = os.path.join(cfg_zed.data_folder, "color_transfer_res")
    mkdirs(path_save_vis_transfer_color)
    

    path_debug_pair_matching = os.path.join(cfg_zed.data_folder, "debug_pair_component")
    mkdirs(path_debug_pair_matching)


    folder_save = os.path.join(path_debug_pair_matching, s_img_name+"_"+c_img_name)

    mkdirs(folder_save)



    # print(mask_c.shape)
    # print(mask_s.shape)

    img_name = "s"+s_kp_id+"_c"+c_kp_id+".jpg"
    vis_mask = cv2.hconcat([mask_s, mask_c])
    vis_mask = cv2.merge((vis_mask, vis_mask, vis_mask))

    width = img_s.shape[1]
    # img_s_converted = cv2.merge((img_s))
    # print(img_s.shape)
    # print(img_c.shape)
    
    vis_img = cv2.hconcat([img_s, img_c])

    start_point = (p_s[1], p_s[0])
    end_point = (width + p_c[1], p_c[0])
    color = (0, 0, 255) 
    thickness = 3

    cv2.line(vis_img, start_point, end_point, color, thickness)

    # print(vis_mask.shape)
    # print(vis_img.shape)
    vis = np.vstack((vis_mask, vis_img))
    cv2.imwrite(os.path.join(folder_save, img_name), vis)



def vis_color_sketch_transfer(pair_matching, pair_im_name, pair_kp_name, pair_components):


    [im_name_sketch, im_name_color] = pair_im_name
    [kp_names_sketch, kp_names_color] = pair_kp_name
    [components_retrieve_sketch, components_retrieve_color] = pair_components

    im_name_sketch = im_name_sketch[0]
    im_name_color = im_name_color[0]

    # print(im_name_sketch)
    # print(im_name_color)
    sketch_img, color_img = read_img_pairs(im_name_sketch, im_name_color)
    for i in range(len(pair_matching)):
        for j in range(len(pair_matching[i])):
            if pair_matching[i][j] == 1:
                name_sketch_kp = kp_names_sketch[i][0]
                name_color_kp = kp_names_color[j][0]

                components_color = components_retrieve_color[int(name_color_kp)]
                components_sketch = components_retrieve_sketch[int(name_sketch_kp)]
                # print(name_color_kp, name_sketch_kp)

                # pixel locations to put color
                # print(components_color)
                list_position_sketch = process_list_point(components_sketch["coords"])
                list_position_color = process_list_point(components_color["coords"])

                color_mask = get_mask(list_position_color, color_img)
                sketch_mask = get_mask(list_position_sketch, sketch_img)

                color_centroid = (int(components_color["centroid"][0][0]), int(components_color["centroid"][0][1]))
                sketch_centroid = (int(components_sketch["centroid"][0][0]), int(components_sketch["centroid"][0][1]))

                color_reference = components_color["color"]
                sketch_img = put_color(color_reference, sketch_img, list_position_sketch)
                debug_component_matching(color_mask, 
                                        sketch_mask, 
                                        str(j), 
                                        str(i), 
                                        im_name_color, 
                                        im_name_sketch,
                                        color_img,
                                        sketch_img,
                                        color_centroid,
                                        sketch_centroid)
    
    # for each in sketch_img:
    #     for each_ in each:
    #         print(each_)
    print("Colorize sketch "+im_name_sketch+" with reference "+im_name_color)
    print("Final sketch unique", np.unique(sketch_img))
    cv2.imwrite(os.path.join(cfg_zed.data_folder, "color_transfer_res",
                            im_name_sketch+"_"+im_name_color+".jpg"), 
                            sketch_img)

def build_binary_combination(input):
    res = [] 
    for x, y in combinations(input, 2):
        res.append([x, y])

    return res

def test():
    a = [[1,2,3], [4,5,6]]
    a = np.array(a)
    a = a[:, :2]
    print(a)


if __name__ == "__main__":
    test()
    


