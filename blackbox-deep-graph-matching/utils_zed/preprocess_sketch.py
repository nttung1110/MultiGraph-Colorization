import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from natsort import natsorted
from PIL import Image
from scipy.optimize import linear_sum_assignment

import sys 

sys.path.append("../../hades_paiting")

from training.sketch_infer import *