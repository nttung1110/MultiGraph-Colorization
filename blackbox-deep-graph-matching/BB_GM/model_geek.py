import torch
import cv2
import numpy as np

import utils.backbone
from BB_GM.affinity_layer import InnerProductWithWeightsAffinity
from BB_GM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
from utils.config import cfg
from utils_zed.config_zed import cfg_zed
from utils.feature_align import feature_align
from torch import nn
from utils.utils import lexico_iter
from utils.visualization import easy_visualize
from utils_zed.prepare_tyler_model import *



# tyler part


from hades_painting.models.shallow_net import UNet
from hades_painting.rules.color_component_matching import ComponentWrapper, get_component_color
from hades_painting.rules.component_wrapper import get_moment_features, resize_mask, build_neighbor_graph

path_tyler_model = "/mnt/ai_filestore/home/zed/multi-graph-matching/blackbox-deep-graph-matching/BB_GM/new_weight.pth"
def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def concat_features_v1(embeddings):
    res = torch.cat(embeddings, dim=0)
    return res

class Net(utils.backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()

        # num_feature = 64#geek padding from 64
        # # num_feature = 1024 #pascalvoc

        node_feature = 64 # 64 for Geek, 1024 for pascal voc
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=node_feature)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        
        # self.affinity_weights = nn.init.uniform_(torch.empty(2, 512)).cuda()

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        perm_mats,
        types,
        tyler_imgs,
        folder_names,
        image_names,
        type_extract,
        is_training=True,
        visualize_flag=False,
        visualization_params=None,
    ):

        global_list = []
        orig_graph_list = []
        
        for image, p, n_p, graph, cur_type, tyler_image, folder_name, image_name in zip(images, points, n_points, graphs, types, tyler_imgs, folder_names, image_names):
            # print("Num:", n_p)
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            # global_list.append(self.affinity_weights)
            
            # nodes = normalize_over_channels(nodes)
            # edges = normalize_over_channels(edges)
            '''
                Folder_name is only used for training(because hades has already saved the annotated file
                in pkl => just loaded)
                If folder_name is None => extract again:
                Considering to exchange these two functions:
                    get_input_data: extract
                    get_input_data_v1: read
            '''

            # insert tyler model here                
            feature_extractor_tyler = UNet(8, 0.0)
            feature_extractor_tyler.load_state_dict(torch.load(path_tyler_model)["model"])

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            feature_extractor_tyler.to(device)
            feature_extractor_tyler.eval()

            if folder_name[0] is None: #inference mode
                image_tyler_input = tyler_image.cpu().data.numpy()
                image_name = image_name
                features_list, masks_list, components_list = get_input_data(image_tyler_input, cur_type, image_name, type_extract)

            else:# train mode
                # image_name = image_name[0]
                features_list, masks_list, components_list = get_input_data_v1(folder_name, image_name)
            
            feature_tyler = []
            with torch.no_grad():
                for features, mask in zip(features_list, masks_list):
                    features = features.float().to(device)
                    mask = mask.to(device)
                    feature_tyler.append(feature_extractor_tyler(features, mask)[0])

            node_features = feature_tyler
            node_features = concat_features_v1(node_features)
            graph.x = node_features

            
            # print(folder_name, image_name)
            
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)
            
            

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_costs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # unary_costs_list = [
        #     self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2])
        #     for (g_1, g_2) in lexico_iter(orig_graph_list)
        # ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]
        if is_training:
            unary_costs_list = [
                [
                    x + 1.0*gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, perm_mat, (ns_src, ns_tgt) in zip(unary_costs_list, perm_mats, lexico_iter(n_points))
            ]

        quadratic_costs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # quadratic_costs_list = [
        #     self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2])
        #     for (g_1, g_2) in lexico_iter(orig_graph_list)
        # ]

        # Aimilarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]
        if is_training == False:
            
            # pair
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solvers = [
                GraphMatchingModule(
                    all_left_edges,
                    all_right_edges,
                    ns_src,
                    ns_tgt,
                    cfg_zed.BB_GM.lambda_val,
                    cfg_zed.BB_GM.solver_params,
                )
                for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                    lexico_iter(all_edges), lexico_iter(n_points)
                )
            ]
            matchings = [
                gm_solver(unary_costs, quadratic_costs)
                for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
            ]

        else:
            if cfg.BB_GM.solver_name == "lpmp":
                all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
                gm_solvers = [
                    GraphMatchingModule(
                        all_left_edges,
                        all_right_edges,
                        ns_src,
                        ns_tgt,
                        cfg.BB_GM.lambda_val,
                        cfg.BB_GM.solver_params,
                    )
                    for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                        lexico_iter(all_edges), lexico_iter(n_points)
                    )
                ]
                matchings = [
                    gm_solver(unary_costs, quadratic_costs)
                    for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
                ]
            elif cfg.BB_GM.solver_name == "multigraph":
                all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
                gm_solver = MultiGraphMatchingModule(
                    all_edges, n_points, cfg.BB_GM.lambda_val, cfg.BB_GM.solver_params)
                matchings = gm_solver(unary_costs_list, quadratic_costs_list)
            else:
                raise ValueError(f"Unknown solver {cfg.BB_GM.solver_name}")

        if visualize_flag:
            easy_visualize(
                orig_graph_list,
                points,
                n_points,
                images,
                unary_costs_list,
                quadratic_costs_list,
                matchings,
                **visualization_params,
            )

        return matchings



# feature_tyler = torch.flatten(output, start_dim=1)
            # print(feature_tyler.shape)
            # print(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)).shape)
            # print(output.shape)
            # arrange features
            # U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            # F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            # node_features = torch.cat((U, F), dim=-1)
            # print(node_features.shape)
            # use tyler feature instead 
            # m = nn.ZeroPad2d((0, 960, 0, 0)) # padding 1024-64=960
            # padding_feature_tyler = m(feature_tyler)
