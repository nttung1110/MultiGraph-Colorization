"""Graph matching config system."""

import os
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:

cfg_zed = __C

__C.data_folder = "./hor02_055"
__C.combine_classes = False
# added
__C.BB_GM = edict()
__C.BB_GM.solver_name = "lpmp"
__C.BB_GM.lambda_val = 80.0
__C.BB_GM.solver_params = edict()
__C.BB_GM.solver_params.timeout = 1000
__C.BB_GM.solver_params.primalComputationInterval = 10
__C.BB_GM.solver_params.maxIter = 100

# __C.BB_GM.solver_params.maxIter = 200
# __C.BB_GM.solver_params.innerIteration = 10
# __C.BB_GM.solver_params.presolveIterations = 30
# __C.BB_GM.solver_params.primalCheckingTriplets = 100
# __C.BB_GM.solver_params.multigraphMatchingRoundingMethod = "MCF_PS"
# __C.BB_GM.solver_params.tighten = ""
# __C.BB_GM.solver_params.tightenIteration = 50
# __C.BB_GM.solver_params.tightenInterval = 20
# __C.BB_GM.solver_params.tightenConstraintsPercentage = 0.1
# __C.BB_GM.solver_params.tightenReparametrization = "uniform:0.5"
# Geek Dataset

__C.Geek = edict()
__C.Geek.KPT_ANNO_DIR = "./data/Geek_prepare_data_kan/annotations/character/keypoints"
__C.Geek.ROOT_DIR = "./data/Geek_prepare_data_kan/"
__C.Geek.SET_SPLIT = "./data/Geek_prepare_data_kan/split/geek_pairs.npz"
__C.Geek.CLASSES = ["character"]



# VOC2011-Keypoint Dataset
__C.VOC2011 = edict()
__C.VOC2011.KPT_ANNO_DIR = "./data/downloaded/PascalVOC/annotations/"  # keypoint annotation
__C.VOC2011.ROOT_DIR = "./data/downloaded/PascalVOC/VOC2011/"  # original VOC2011 dataset
__C.VOC2011.SET_SPLIT = "./data/split/voc2011_pairs.npz"  # set split path
__C.VOC2011.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Willow-Object Dataset
__C.WILLOW = edict()
__C.WILLOW.ROOT_DIR = "./data/downloaded/WILLOW/WILLOW-ObjectClass"
__C.WILLOW.CLASSES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
__C.WILLOW.KPT_LEN = 10
__C.WILLOW.TRAIN_NUM = 20
__C.WILLOW.TRAIN_OFFSET = 0

# SPair Dataset
__C.SPair = edict()
__C.SPair.ROOT_DIR = "./data/downloaded/SPair-71k"
__C.SPair.size = "large"
__C.SPair.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]


#
# Training options
#

__C.TRAIN = edict()
__C.TRAIN.difficulty_params = {}
# Iterations per epochs

__C.EVAL = edict()
__C.EVAL.difficulty_params = {}

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = "data/cache"

# random seed used for data loading
__C.RANDOM_SEED = 123
