import sys
import os
import time
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from natsort import natsorted
from PIL import Image
from scipy.optimize import linear_sum_assignment

from rules.color_component_matching import ComponentWrapper, get_component_color
from rules.component_wrapper import get_moment_features, resize_mask, build_neighbor_graph
from models.new_shallow_net import UNet
from models.utils import cosine_distance
from rules.complex_matching import complex_matching


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    return args


def get_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_image(path):
    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def get_input_data(path, size, mean, std, reference=True):
    component_wrapper = ComponentWrapper(min_area=15, min_size=3)

    color_image = read_image(path)
    # extract components
    mask, components = component_wrapper.process(color_image, None, ComponentWrapper.EXTRACT_COLOR)
    graph = build_neighbor_graph(mask)
    mask = resize_mask(mask, components, size).astype(np.int32)

    # get features
    features = get_moment_features(components, mask)
    features = (features - mean) / std
    features = torch.tensor(features).float().unsqueeze(0)
    mask = torch.tensor(mask).long().unsqueeze(0)

    if reference:
        get_component_color(components, color_image, ComponentWrapper.EXTRACT_COLOR)
        return features, mask, components, graph, color_image

    black_values = [0, 5, 10, 15]
    frame = np.full_like(color_image, 255)
    for x in black_values:
        frame = np.where(color_image == [x, x, x], np.zeros_like(color_image), frame)
    return features, mask, components, graph, color_image, frame


def draw_components(image, target, source, pairs):
    image = np.full_like(image, 255)

    for index, component in enumerate(target):
        match_index = [p[1] for p in pairs if p[0] == index]
        if len(match_index) == 0:
            continue

        match_index = match_index[0]
        match_part = source[match_index]

        match_color = np.array(match_part["color"])
        coords = component["coords"]

        if component["label"] == 236:
            image[coords[:, 0], coords[:, 1], :] = [255, 255, 255]
        else:
            image[coords[:, 0], coords[:, 1], :] = match_color
    return image


def linear_matching(output_a, output_b):
    pairs = []

    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])

        distances = cosine_distance(region_b, output_a)
        min_index = torch.argmin(distances).item()
        pairs.append([index_b, min_index])
    return pairs


def hungarian_matching(output_a, output_b):
    distance = np.zeros((output_b.shape[1], output_a.shape[1]))

    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])
        distance[index_b, :] = cosine_distance(region_b, output_a)

    pairs = linear_sum_assignment(distance)
    pairs = [[a, b] for a, b in zip(pairs[0], pairs[1])]
    return pairs


def pick_reference_sketch(character_dir):
    component_wrapper = ComponentWrapper(min_area=15, min_size=3)
    image_paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))

    candidates = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        components = component_wrapper.process(image, None, ComponentWrapper.EXTRACT_COLOR)[1]
        candidates.append((len(components)))

    final_candidate = int(np.argmax(candidates))
    return final_candidate


def main(args):
    device = torch.device("cuda:0")

    image_size = (768, 512)
    mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
    std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
    mean = np.array(mean)[:, np.newaxis][:, np.newaxis]
    std = np.array(std)[:, np.newaxis][:, np.newaxis]

    # Initialize model
    model = UNet(8, 0.0)
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.to(device)
    model.eval()

    character_dirs = natsorted(glob.glob(os.path.join(args.input_dir, "*")))
    for character_dir in character_dirs:
        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        reference_index = pick_reference_sketch(character_dir)
        reference_path = paths[reference_index]

        character_name = os.path.basename(character_dir)
        if not os.path.exists(os.path.join(args.output_dir, character_name)):
            os.makedirs(os.path.join(args.output_dir, character_name))

        for path in paths:
            if path == reference_path:
                continue

            image_name = get_base_name(path)
            # Get data
            features_a, mask_a, components_a, graph_a, color_a = get_input_data(reference_path, image_size, mean, std)
            features_b, mask_b, components_b, graph_b, color_b, frame = get_input_data(
                path, image_size, mean, std, False)
            if torch.max(mask_a).item() == 0 or torch.max(mask_b).item() == 0:
                print(path, reference_path)
                continue

            # Run the model
            start = time.time()
            with torch.no_grad():
                features_a = features_a.float().to(device)
                mask_a = mask_a.to(device)
                features_b = features_b.float().to(device)
                mask_b = mask_b.to(device)

                output_a = model(features_a, mask_a)
                output_b = model(features_b, mask_b)
                pairs = complex_matching(output_a, output_b, graph_a, graph_b, components_a, components_b)
            print(time.time() - start)

            output_image = draw_components(frame, components_b, components_a, pairs)
            output_image[np.where((frame == [0, 0, 0]).all(axis=-1))] = [0, 0, 0]
            cv2.imwrite(os.path.join(args.output_dir, character_name, "%s.png" % image_name), output_image)
    return


if __name__ == "__main__":
    main(parse_arguments())
