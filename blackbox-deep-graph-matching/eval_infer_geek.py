import time
from pathlib import Path
import cv2

import torch
import numpy as np

from utils.config import cfg
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg
from utils.utils import lexico_iter
from utils_zed.transfer_color import vis_color_sketch_transfer, \
                                    build_binary_combination, \
                                    vis_sketch_triple_graph,\
                                    vis_sketch_triple_graph_v2,\
                                    triplet_predict_color_sketch
                                    

def predict_color_sketch(s_pred_list, image_names, types, kp_label_name, folder_names):
    '''
        s_pred_list: combinatorial matching pairs containing (n-1)! pairs with n is the number
        of graph for matching
    '''
    # print(len(s_pred_list))
    triplet_im_name = [name for name in image_names]
    triplet_kp_name = [kp_name for kp_name in kp_label_name]

    image_names_cb = build_binary_combination(image_names)
    types_cb = build_binary_combination(types)
    kp_label_name_cb = build_binary_combination(kp_label_name)

    triplet_matching = []
    folder_name = folder_names[0][0]
    for idx, each_pair in enumerate(s_pred_list):
        
        triplet_matching.append(each_pair)

        images_pair = image_names_cb[idx]
        
        types_pair = types_cb[idx]
        kp_label_name_pair = kp_label_name_cb[idx]

        if types_pair[0][0] == types_pair[1][0]:
            # refuse to transfer color for sketch-sketch or color-color
            continue

        i_name_1, i_name_2 = images_pair[0], images_pair[1]
        type_1, type_2 = types_pair[0][0], types_pair[1][0]
        kp_name_1, kp_name_2 = kp_label_name_pair[0], kp_label_name_pair[1]
        each_pair = each_pair[0]
        if type_1 == "color" and type_2 == "sketch":
            i_name_1, i_name_2 = images_pair[1], images_pair[0]
            type_1, type_2 = types_pair[1][0], types_pair[0][0]
            kp_name_1, kp_name_2 = kp_label_name_pair[1], kp_label_name_pair[0]
            # Transpose permutation matrix to become sketch->color mapping
            # each_pair = np.array()
            each_pair = each_pair.T
            each_pair = each_pair.tolist()


        vis_color_sketch_transfer(each_pair, 
                                (i_name_1, i_name_2),
                                (kp_name_1, kp_name_2),
                                folder_name)
        
        print("On colorizing:", i_name_1, i_name_2)

    
    # vis_sketch_triple_graph_v2(triplet_matching, triplet_im_name, triplet_kp_name)
    # print("On colorizing:", triplet_im_name)

    # for p1 in points_gt:
    #     print("Len p1:", len(p1))
    #     for p2 in p1:
    #         print("Len p2:", len(p2))
    #         # for p3 in p2:
            #     print("Len p3:", p3)




    # print(len(s_pred_list))
    # for s1 in s_pred_list:
    #     print("Out s1:", len(s1))
    #     for s1_1 in s1:
    #         print("Num nodes 1:", len(s1_1))
    #         for s1_1_1 in s1_1:
    #             print("Num nodes 2:", len(s1_1_1))
    #             break
    #             # for s1_1_1_1 in s1_1_1:
    #             #     print("Out s1_1_1_1:", len(s1_1_1_1))
   

def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / "params" / "{:04}".format(eval_epoch) / "params.pt")
        print("Loading model parameters from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    ds.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]
            tyler_imgs = inputs["tyler_imgs"]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            # print("Num points:", n_points_gt)
            # print(inputs["edges"])
            edges = [_.to("cuda") for _ in inputs["edges"]]

            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)
            # geek
            image_names = inputs["image_names"]
            # print(image_names)
            types = inputs["types"]
            # print(types)
            kp_label_name = inputs["kp_label_name"]
            folder_names = inputs["folder_names"]
            image_names = inputs["image_names"]

            iter_num = iter_num + 1

            visualize = k == 0 and cfg.visualize    
            visualization_params = {**cfg.visualization_params, **dict(string_info=cls, true_matchings=perm_mat_list)}
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
                    visualize_flag=visualize,
                    visualization_params=visualization_params,
                )

            # predict_color_sketch(s_pred_list, image_names, types, kp_label_name, folder_names)

            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_list[0], perm_mat_list[0])
            _tp, _fp, _fn = get_pos_neg(s_pred_list[0], perm_mat_list[0])

            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            tp += _tp
            fp += _fp
            fn += _fn

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        f1_scores[i] = f1_score(tp, fp, fn)
        if verbose:
            print("Class {} acc = {:.4f} F1 = {:.4f}".format(cls, accs[i], f1_scores[i]))

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    return accs, f1_scores
