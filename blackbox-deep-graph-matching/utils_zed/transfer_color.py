import numpy as np 
import cv2
import json 
from itertools import combinations
import os 
import sys 
import pickle
sys.path.append("../")
from utils.config import cfg

folder_name = (cfg.Geek.ROOT_DIR.split("/")[-2]).split("_")[3:]
folder_name = "_".join(folder_name)

path_root_data = cfg.Geek.ROOT_DIR

path_components_retrieve = "../components_saving/HOR02/"+folder_name

path_img = "./data/"+cfg.Geek.ROOT_DIR.split("/")[-2]+"/color"
path_save_vis_transfer_color = "../color_transfer_res/"+folder_name
path_debug_pair_matching = "../debug_pair_components/"+folder_name

path_triplet_matching = "../debug_triplet_components/"+folder_name


def load_components_retrieve_triplet(im_name_sketch, im_name_color_1, im_name_color_2):
    path_sketch = os.path.join(path_components_retrieve, "sketch", im_name_sketch+".json")
    path_color_1 = os.path.join(path_components_retrieve, "color", im_name_color_1+".json")
    path_color_2 = os.path.join(path_components_retrieve, "color", im_name_color_2+".json")

    with open(path_sketch, "r") as f_p:
        components_retrieve_sketch = json.load(f_p)

    with open(path_color_1, "r") as f_p:
        components_retrieve_color_1 = json.load(f_p)

    with open(path_color_2, "r") as f_p:
        components_retrieve_color_2 = json.load(f_p)

    return components_retrieve_sketch, components_retrieve_color_1, components_retrieve_color_2

def load_components_retrieve(im_name_sketch, im_name_color):
    path_sketch = os.path.join(path_components_retrieve, "sketch", im_name_sketch+".json")
    path_color = os.path.join(path_components_retrieve, "color", im_name_color+".json")
    with open(path_sketch, "r") as f_p:
        components_retrieve_sketch = json.load(f_p)

    with open(path_color, "r") as f_p:
        components_retrieve_color = json.load(f_p)
    return components_retrieve_sketch, components_retrieve_color


def load_components_retrieve_pkl(im_name_sketch, im_name_color, folder_name):
    path_sketch = os.path.join(cfg.Geek.ROOT_DIR, "annotations", "character", "components", folder_name, im_name_sketch+".pkl")
    path_color = os.path.join(cfg.Geek.ROOT_DIR, "annotations", "character", "components", folder_name, im_name_color+".pkl")


    info_components_sketch = pickle.load(open(path_sketch, "rb"), fix_imports=True, encoding="latin1")
    info_components_color = pickle.load(open(path_color, "rb"), fix_imports=True, encoding="latin1")

    info_components_sketch = info_components_sketch["components"]
    info_components_color = info_components_color["components"]

    components_retrieve_sketch = {}
    components_retrieve_color = {}

    for c_s in info_components_sketch:
        components_retrieve_sketch[c_s["label"]] = c_s 

    for c_c in info_components_color:
        components_retrieve_color[c_c["label"]] = c_c
    
    return components_retrieve_sketch, components_retrieve_color

def read_img_pairs(im_name_sketch, im_name_color, folder_name):
    path_color = os.path.join(cfg.Geek.ROOT_DIR, "image", "color", folder_name, im_name_color+".png")
    path_sketch = os.path.join(cfg.Geek.ROOT_DIR, "image", "sketch", folder_name, im_name_sketch+".png")


    color_img = cv2.imread(path_color)
    sketch_img = cv2.imread(path_sketch)

    return sketch_img, color_img

def read_img_triplet(s1_name, c1_name, c2_name):
    path_color_1 = os.path.join(path_img, c1_name+".png")
    path_color_2 = os.path.join(path_img, c2_name+".png")
    path_sketch = os.path.join(path_img, s1_name+".png")


    color_img_1 = cv2.imread(path_color_1)
    color_img_2 = cv2.imread(path_color_2)

    sketch_img = cv2.imread(path_sketch)

    return sketch_img, color_img_1, color_img_2

def put_color(reference_color, sketch_img, list_position):
    for [x, y] in list_position:
        sketch_img[x][y] = reference_color
    return sketch_img

def get_mask(region, input_image):
    mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)
    mask[region[:, 0], region[:, 1]] = 255

    return mask

def process_list_point(list_point):
    list_point = np.array(list_point)
    list_point = list_point[:, :2]
    return list_point

def mkdirs(path_in):
    if os.path.isdir(path_in) is False:
        os.mkdir(path_in)

def debug_component_matching(mask_c, mask_s, c_kp_id, s_kp_id, c_img_name, s_img_name, img_c, img_s, p_c, p_s, folder_name):
    # mkdirs(path_save_vis_transfer_color)
    # mkdirs(path_debug_pair_matching)

    path_save_vis_transfer_color = os.path.join(cfg.Geek.ROOT_DIR, "color_transfer_res")
    mkdirs(path_save_vis_transfer_color)
    path_save_vis_transfer_color = os.path.join(path_save_vis_transfer_color, folder_name)
    mkdirs(path_save_vis_transfer_color)

    path_debug_pair_matching = os.path.join(cfg.Geek.ROOT_DIR, "debug_pair_component")
    mkdirs(path_debug_pair_matching)
    path_debug_pair_matching = os.path.join(path_debug_pair_matching, folder_name)
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

def debug_triplet_matching(triplet_mask, triplet_kp_id, triplet_im_name, triplet_im, triple_p):
    # mkdirs(path_save_vis_transfer_color)
    # mkdirs(path_debug_pair_matching)
    mkdirs(path_triplet_matching)

    [s_img_name, c_img_name_1, c_img_name_2] = triplet_im_name
    folder_save = os.path.join(path_triplet_matching, s_img_name+"_"+c_img_name_1+"_"+c_img_name_2)

    mkdirs(folder_save)



    # print(mask_c.shape)
    # print(mask_s.shape)

    [s_kp_id, c_kp_id_1, c_kp_id_2] = triplet_kp_id

    img_name = "s_a_" + s_kp_id

    if c_kp_id_1 != None:
        img_name += "_c_b" + c_kp_id_1

    if c_kp_id_2 != None:
        img_name += "_c_c" + c_kp_id_2

    img_name += ".jpg"

    [mask_sketch, mask_color_1, mask_color_2] = triplet_mask
    # create 3 channel mask
    mask_sketch = cv2.merge((mask_sketch, mask_sketch, mask_sketch))

    if mask_color_1 is None:
        mask_color_1 = np.ones(mask_sketch.shape)*129
    else:
        mask_color_1 = cv2.merge((mask_color_1, mask_color_1, mask_color_1))

    if mask_color_2 is None:
        mask_color_2 = np.ones(mask_sketch.shape)*129
    else:
        mask_color_2 = cv2.merge((mask_color_2, mask_color_2, mask_color_2))

    [sketch, color_1, color_2] = triplet_im

    # mask_vs_img_sketch = cv2.hconcat(mask_sketch, sketch)
    # mask_vs_img_color_1 = cv2.hconcat(mask_color_1, color_1)
    # mask_vs_img_color_2 = cv2.hconcat(mask_color_2, color_2)


    mask_vs_img_sketch = np.concatenate((mask_sketch, sketch), axis=1)
    mask_vs_img_color_1 = np.concatenate((mask_color_1, color_1), axis=1)
    mask_vs_img_color_2 = np.concatenate((mask_color_2, color_2), axis=1)


    # print("Mask", mask_vs_img_sketch.shape)

    empty_white = np.ones(mask_vs_img_sketch.shape)*129

    # print("White", empty_white.shape)
    upper_part = np.concatenate((np.concatenate((mask_vs_img_color_1, empty_white), axis=1), mask_vs_img_color_2), axis=1)
    lower_part = np.concatenate((np.concatenate((empty_white, mask_vs_img_sketch), axis=1), empty_white), axis=1)

    (c1_h, c1_w, c) = color_1.shape 
    (c2_h, c2_w, c) = color_2.shape
    (s_h, s_w, c) = sketch.shape
    
    # img_s_converted = cv2.merge((img_s))
    # print(img_s.shape)
    # print(img_c.shape)
    vis_img = np.vstack((upper_part, lower_part))
    
    # print(vis_img.shape)
    [p_s, p_c_1, p_c_2] = triple_p

    start_point = (p_s[1] + 2*c1_w + s_w, p_s[0] + c1_h)
    # from sketch to color b
    if p_c_1 is not None:
        end_point = (c1_w + p_c_1[1], p_c_1[0])
        color = (0, 0, 255) 
        thickness = 3
        cv2.line(vis_img, start_point, end_point, color, thickness)

    # from sketch to color c
    if p_c_2 is not None:
        end_point = (2*c1_w + 2*s_w + c2_w + p_c_2[1], p_c_2[0])
        color = (0, 0, 255) 
        thickness = 3
        cv2.line(vis_img, start_point, end_point, color, thickness)

    # from color c to color b
    if p_c_1 is not None and p_c_2 is not None:
        start_point = (2*c1_w + 2*s_w + c2_w + p_c_2[1], p_c_2[0])
        end_point = (c1_w + p_c_1[1], p_c_1[0])
        color = (0, 0, 255) 
        thickness = 3
        cv2.line(vis_img, start_point, end_point, color, thickness)

    # print(vis_mask.shape)
    # print(vis_img.shape)
    # vis = np.vstack((vis_mask, vis_img))
    cv2.imwrite(os.path.join(folder_save, img_name), vis_img)

def cycle_checking(idx_a, idx_b, pair_matching_2, pair_matching_3):
    '''
        b- - - -c
         '     '
          '   '
           'a'   
        a is sketch
        b is color
        c is color

        pair_matching_2: matching result from a->c
        pair_matching_3: matching result from c->b

        Condition a[idx_a]->b[idx_b]
        <=> there exists idx_c such that (a[idx_a]->c[idx_c] and c[idx_c]->b[idx_b])      
    '''
    # Find idx_c 
    find_candidate = False
    idx_c_found = None
    for idx_c in range(len(pair_matching_2[0])):
        if pair_matching_2[idx_a][idx_c] == 1 and pair_matching_3[idx_c][idx_b] == 1:
            find_candidate = True 
            idx_c_found = idx_c
            break

    return idx_c_found, find_candidate


def vis_sketch_triple_graph_v2(triplet_matching, triplet_im_name, triplet_kp_name):
    '''
        b<- - - -c
         ^     ^
          '   '
           'a'  
        a->b: pair_matching_1
        a->c: pair_matching_2
        c->b: pair_matching_3

        list_pair_matching: list of pair_matching from source to des (source is sketch and des is 
        color)
        list_pair_im_name: list of image_name in pairs from source to des
        list_pair_kp_name: list of pair_kp_name in pairs from source to des
    '''

    [im_name_sketch, im_name_color_1, im_name_color_2] = triplet_im_name
    [kp_name_sketch, kp_name_color_1, kp_name_color_2] = triplet_kp_name
    [pair_matching_1, pair_matching_2, pair_matching_3] = triplet_matching #should check oder care full

    # print(pair_matching_1, pair_matching_2, pair_matching_3)
    '''
        pair_matching_1 matchs sketch_a to color_b
        pair_matching_2 matches sketch_a to color_c
        pair_matching_3 matches color_c to color_b
    '''

    im_name_sketch = im_name_sketch[0]
    im_name_color_1 = im_name_color_1[0]
    im_name_color_2 = im_name_color_2[0]

        
    components_retrieve_sketch, \
    components_retrieve_color_1, \
    components_retrieve_color_2 = load_components_retrieve_triplet(im_name_sketch, \
                                                                    im_name_color_1, \
                                                                    im_name_color_2)

    sketch_img, color_img_1, color_img_2 = read_img_triplet(im_name_sketch,\
                                                            im_name_color_1, \
                                                            im_name_color_2)


    '''
        b<- - - -c
         ^     ^
          '   '
           'a'  
        a->b: pair_matching_1
        a->c: pair_matching_2
        c->b: pair_matching_3

        list_pair_matching: list of pair_matching from source to des (source is sketch and des is 
        color)
        list_pair_im_name: list of image_name in pairs from source to des
        list_pair_kp_name: list of pair_kp_name in pairs from source to des
    '''

    for i in range(len(pair_matching_1)):
        # accumulate matching results from both graph b and graph c

        # name
        name_sketch_kp = kp_name_sketch[i][0]
        name_color_kp_1 = None
        name_color_kp_2 = None

        # component info
        components_sketch = components_retrieve_sketch[name_sketch_kp]
        list_position_sketch = process_list_point(components_sketch["coords"])
        sketch_mask = get_mask(list_position_sketch, sketch_img)
        sketch_centroid = (int(components_sketch["centroid"][0]), int(components_sketch["centroid"][1]))
        a_id = i
        b_id = None 
        c_id = None

        for j in range(len(pair_matching_1[i])):
            if pair_matching_1[i][j] == 1:
                b_id = j 
                name_color_kp_1 = kp_name_color_1[j][0]
                break
        
        for k in range(len(pair_matching_2[i])):
            if pair_matching_2[i][k] == 1:
                c_id = k 
                name_color_kp_2 = kp_name_color_2[k][0]
                break

        # only colorize when criteria are met

        if name_color_kp_1 != None:
            components_color_1 = components_retrieve_color_1[name_color_kp_1]
            list_position_color_1 = process_list_point(components_color_1["coords"])
            color_mask_1 = get_mask(list_position_color_1, color_img_1)
            color__centroid_1 = (int(components_color_1["centroid"][0]), int(components_color_1["centroid"][1]))
            color_reference_1 = components_color_1["color"]
            color_reference_1 = components_color_1["color"]

        else:
            color_mask_1 = None 
            color__centroid_1 = None

        if name_color_kp_2 != None:
            components_color_2 = components_retrieve_color_2[name_color_kp_2]
            list_position_color_2 = process_list_point(components_color_2["coords"])
            color_mask_2 = get_mask(list_position_color_2, color_img_2)
            color__centroid_2 = (int(components_color_2["centroid"][0]), int(components_color_2["centroid"][1]))
            color_reference_2 = components_color_2["color"]
            color_reference_2 = components_color_2["color"]
        

        else:
            color_mask_2 = None 
            color__centroid_2 = None

        if b_id == None and c_id == None:
            # no matching component found from two graph
            continue
        elif b_id is not None and c_id is not None:
            if color_reference_1 != color_reference_2:
                continue
                
        if b_id is not None:
            sketch_img = put_color(color_reference_1, sketch_img, list_position_sketch)
        
        if c_id is not None:
            sketch_img = put_color(color_reference_2, sketch_img, list_position_sketch)


        # build triplet
        triplet_mask = (sketch_mask, color_mask_1, color_mask_2)
        triplet_kp_id = (str(a_id), str(b_id), str(c_id))
        triplet_im_name = (im_name_sketch, im_name_color_1, im_name_color_2)
        triplet_im = (sketch_img, color_img_1, color_img_2) 
        triple_p = (sketch_centroid, color__centroid_1, color__centroid_2)

        debug_triplet_matching(triplet_mask, \
                                        triplet_kp_id, \
                                        triplet_im_name, \
                                        triplet_im, \
                                        triple_p)



def vis_sketch_triple_graph(triplet_matching, triplet_im_name, triplet_kp_name):
    '''
        b<- - - -c
         ^     ^
          '   '
           'a'  
        a->b: pair_matching_1
        a->c: pair_matching_2
        c->b: pair_matching_3

        list_pair_matching: list of pair_matching from source to des (source is sketch and des is 
        color)
        list_pair_im_name: list of image_name in pairs from source to des
        list_pair_kp_name: list of pair_kp_name in pairs from source to des
    '''

    [im_name_sketch, im_name_color_1, im_name_color_2] = triplet_im_name
    [kp_name_sketch, kp_name_color_1, kp_name_color_2] = triplet_kp_name
    [pair_matching_1, pair_matching_2, pair_matching_3] = triplet_matching #should check oder care full

    # print(pair_matching_1, pair_matching_2, pair_matching_3)
    '''
        pair_matching_1 matchs sketch_a to color_b
        pair_matching_2 matches sketch_a to color_c
        pair_matching_3 matches color_c to color_b
    '''

    im_name_sketch = im_name_sketch[0]
    im_name_color_1 = im_name_color_1[0]
    im_name_color_2 = im_name_color_2[0]

        
    components_retrieve_sketch, \
    components_retrieve_color_1, \
    components_retrieve_color_2 = load_components_retrieve_triplet(im_name_sketch, \
                                                                    im_name_color_1, \
                                                                    im_name_color_2)

    sketch_img, color_img_1, color_img_2 = read_img_triplet(im_name_sketch,\
                                                            im_name_color_1, \
                                                            im_name_color_2)

    # go through pair_1 first

    for i in range(len(pair_matching_1)):
        exist_paired_components = False 
        for j in range(len(pair_matching_1[i])):
            if pair_matching_1[i][j] == 1: 

                k, is_consistency = cycle_checking(i, j, pair_matching_2, pair_matching_3)

                if is_consistency == False:
                    continue


                name_sketch_kp = kp_name_sketch[i][0]
                name_color_kp_1 = kp_name_color_1[j][0]
                name_color_kp_2 = kp_name_color_2[k][0]

                components_color_1 = components_retrieve_color_1[name_color_kp_1]
                components_color_2 = components_retrieve_color_2[name_color_kp_2]
                components_sketch = components_retrieve_sketch[name_sketch_kp]

                # pixel locations to put color
                list_position_sketch = process_list_point(components_sketch["coords"])
                list_position_color_1 = process_list_point(components_color_1["coords"])
                list_position_color_2 = process_list_point(components_color_2["coords"])

                color_mask_1 = get_mask(list_position_color_1, color_img_1)
                color_mask_2 = get_mask(list_position_color_2, color_img_2)
                sketch_mask = get_mask(list_position_sketch, sketch_img)

                color__centroid_1 = (int(components_color_1["centroid"][0]), int(components_color_1["centroid"][1]))
                color__centroid_2 = (int(components_color_2["centroid"][0]), int(components_color_2["centroid"][1]))
                sketch_centroid = (int(components_sketch["centroid"][0]), int(components_sketch["centroid"][1]))

                color_reference_1 = components_color_1["color"]
                color_reference_2 = components_color_2["color"]

                if color_reference_1 != color_reference_2:
                    continue

                exist_paired_components = True
                # color_reference = color_img[color__centroid[0]][color__centroid[1]]
                sketch_img = put_color(color_reference_1, sketch_img, list_position_sketch)

                # build triplet
                triplet_mask = (sketch_mask, color_mask_1, color_mask_2)
                triplet_kp_id = (str(i), str(j), str(k))
                triplet_im_name = (im_name_sketch, im_name_color_1, im_name_color_2)
                triplet_im = (sketch_img, color_img_1, color_img_2) 
                triple_p = (sketch_centroid, color__centroid_1, color__centroid_2)




                debug_triplet_matching(triplet_mask, \
                                        triplet_kp_id, \
                                        triplet_im_name, \
                                        triplet_im, \
                                        triple_p)


        # if exist_paired_components == False:
    



def vis_color_sketch_transfer(pair_matching, pair_im_name, pair_kp_name, folder_name, mode="training"):


    [im_name_sketch, im_name_color] = pair_im_name
    [kp_names_sketch, kp_names_color] = pair_kp_name
    

    im_name_sketch = im_name_sketch[0]
    im_name_color = im_name_color[0]

    # print(im_name_sketch)
    # print(im_name_color)
    if mode == "training":
        components_retrieve_sketch, components_retrieve_color = load_components_retrieve_pkl(im_name_sketch, im_name_color, folder_name)
    sketch_img, color_img = read_img_pairs(im_name_sketch, im_name_color, folder_name)
    # print(len(pair_matching))
    # print(len(pair_matching[0]))
    for i in range(len(pair_matching)):
        for j in range(len(pair_matching[i])):
            if pair_matching[i][j] == 1:
                name_sketch_kp = kp_names_sketch[i][0]
                name_color_kp = kp_names_color[j][0]

                components_color = components_retrieve_color[int(name_color_kp)]
                components_sketch = components_retrieve_sketch[int(name_sketch_kp)]
                # print(name_color_kp, name_sketch_kp)

                # pixel locations to put color
                list_position_sketch = process_list_point(components_sketch["coords"])
                list_position_color = process_list_point(components_color["coords"])

                color_mask = get_mask(list_position_color, color_img)
                sketch_mask = get_mask(list_position_sketch, sketch_img)

                color__centroid = (int(components_color["centroid"][0]), int(components_color["centroid"][1]))
                sketch_centroid = (int(components_sketch["centroid"][0]), int(components_sketch["centroid"][1]))

                color_reference = components_color["color"]
                # color_reference = color_img[color__centroid[0]][color__centroid[1]]
                sketch_img = put_color(color_reference, sketch_img, list_position_sketch)

                debug_component_matching(color_mask, 
                                        sketch_mask, 
                                        str(j), 
                                        str(i), 
                                        im_name_color, 
                                        im_name_sketch,
                                        color_img,
                                        sketch_img,
                                        color__centroid,
                                        sketch_centroid,
                                        folder_name)
    
    print("Colorize sketch "+im_name_sketch+" with reference "+im_name_color)
    cv2.imwrite(os.path.join(cfg.Geek.ROOT_DIR, "color_transfer_res", folder_name,
                            im_name_sketch+"_"+im_name_color+".jpg"), 
                            sketch_img)

def triplet_predict_color_sketch(s_pred_list, image_names, types, kp_label_name):
    triplet_im_name = [name[0] for name in image_names]
    triplet_kp_name = [kp_name for kp_name in kp_label_name]
    triplet_matching = [None, None, None]

    print("On colorizing:", triplet_im_name)

    sketch_pos = None
    color_1_pos = None 
    color_2_pos = None 

    for idx, type_im in enumerate(types):
        if type_im[0] == "sketch":
            sketch_pos = idx 
        else:
            if color_1_pos == None:
                color_1_pos = idx 
            else:
                color_2_pos = idx

    true_pos = {0: None, 1: None, 2: None}
    true_pos[sketch_pos] = 0
    true_pos[color_1_pos] = 1
    true_pos[color_2_pos] = 2

    true_pos_triplet_pair = {0: {0: None, 1: None, 2: None},\
                        1: {0: None, 1: None, 2: None},\
                        2: {0: None, 1: None, 2: None}}

    # print(sketch_pos, color_1_pos, color_2_pos)
    true_pos_triplet_pair[true_pos[sketch_pos]][true_pos[color_1_pos]] = 0
    true_pos_triplet_pair[true_pos[color_1_pos]][true_pos[sketch_pos]] = 0

    true_pos_triplet_pair[true_pos[sketch_pos]][true_pos[color_2_pos]] = 1
    true_pos_triplet_pair[true_pos[color_2_pos]][true_pos[sketch_pos]] = 1

    combinatorial = []
    combinatorial.append([0, 1])
    combinatorial.append([0, 2])
    combinatorial.append([1, 2])

    if sketch_pos != None and color_1_pos != None and color_2_pos != None:
        triplet_im_name = [image_names[sketch_pos], \
                            image_names[color_1_pos], \
                            image_names[color_2_pos]]

        triplet_kp_name = [kp_label_name[sketch_pos], \
                            kp_label_name[color_1_pos], \
                            kp_label_name[color_2_pos]]

        for i in range(len(combinatorial)):
            pair_matching = s_pred_list[i][0]

            from_raw_index, to_raw_index = combinatorial[i][0], combinatorial[i][1]
            from_true_index, to_true_index = true_pos[from_raw_index], true_pos[to_raw_index]

            # print(from_true_index, to_true_index)
            if (from_true_index == 2 and to_true_index == 1):
                triplet_matching[2] = pair_matching
            
            elif (from_true_index == 1 and to_true_index == 2):
                pair_matching = pair_matching.T 
                pair_matching = pair_matching.tolist()
                triplet_matching[2] = pair_matching

            else:
                if from_true_index > to_true_index:
                    tmp = from_true_index
                    from_true_index = to_true_index
                    to_true_index = tmp 
                    pair_matching = pair_matching.T
                    pair_matching = pair_matching.tolist()
            
                true_idx_match = true_pos_triplet_pair[from_true_index][to_true_index]
                    
                triplet_matching[true_idx_match] = pair_matching

        vis_sketch_triple_graph_v2(triplet_matching, triplet_im_name, triplet_kp_name)





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
    


