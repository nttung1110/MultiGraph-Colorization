import sys
import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from natsort import natsorted
from PIL import Image

from rules.color_component_matching import ComponentWrapper, get_component_color
from rules.component_wrapper import get_moment_features, resize_mask, build_neighbor_graph
from models.new_shallow_net import UNet
from training.run_model_output import linear_matching, draw_components
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
    mask, components = component_wrapper.process(color_image)
    graph = build_neighbor_graph(mask)
    mask = resize_mask(mask, components, size).astype(np.int32)

    # get features
    features = get_moment_features(components, mask)
    features = (features - mean) / std
    features = torch.tensor(features).float().unsqueeze(0)
    mask = torch.tensor(mask).long().unsqueeze(0)

    if reference:
        get_component_color(components, color_image)
        return features, mask, components, graph, color_image

    black_values = [0, 5, 10, 15, 20, 25, 30]
    frame = np.full_like(color_image, 255)
    for x in black_values:
        frame = np.where(color_image == [x, x, x], np.zeros_like(color_image), frame)
    return features, mask, components, graph, color_image, frame


def main(args):
    device = torch.device("cpu")

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
    print(model.gamma)

    character_dirs = natsorted(glob.glob(os.path.join(args.input_dir, "*")))
    exist_names = natsorted(glob.glob(os.path.join(args.output_dir, "*.png")))
    exist_names = [get_base_name(path) for path in exist_names]

    for character_dir in character_dirs:
        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        character_name = os.path.basename(character_dir)

        if not os.path.exists(os.path.join(args.output_dir, character_name)):
            os.makedirs(os.path.join(args.output_dir, character_name))

        for path, next_path in zip(paths[:-1], paths[1:]):
            image_name = get_base_name(next_path)
            name = character_name + "_" + image_name
            print(name)

            # Get data
            features_a, mask_a, components_a, graph_a, color_a = get_input_data(path, image_size, mean, std)
            features_b, mask_b, components_b, graph_b, color_b, frame = get_input_data(
                next_path, image_size, mean, std, False)
            if torch.max(mask_a).item() == 0 or torch.max(mask_b).item() == 0:
                print(path, next_path)
                continue

            # Run the model
            with torch.no_grad():
                features_a = features_a.float().to(device)
                mask_a = mask_a.to(device)
                features_b = features_b.float().to(device)
                mask_b = mask_b.to(device)

                output_a = model(features_a, mask_a)
                output_b = model(features_b, mask_b)
                pairs = complex_matching(output_a, output_b, graph_a, graph_b, components_a, components_b)

            output_image = draw_components(frame, components_b, components_a, pairs)
            output_image = np.where(frame == 0, frame, output_image)
            output_image = np.concatenate([color_a, color_b, output_image], axis=1)
            cv2.imwrite(os.path.join(args.output_dir, character_name, "%s.png" % image_name), output_image)
    return


if __name__ == "__main__":
    main(parse_arguments())
