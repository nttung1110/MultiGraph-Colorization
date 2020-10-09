import os
import copy
import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from PIL import Image

from kan.component_wrapper import ComponentWrapper
from rules.component_wrapper import ComponentWrapper as ColorComponent
from rules.component_wrapper import get_moment_features, resize_mask, build_neighbor_graph
from models.new_shallow_net import UNet
from rules.complex_matching import complex_matching_with_distance


def read_image(path):
    image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)[1]
    return image


def adapt_component_mask(mask, components, min_area=15, min_size=3):
    components = [components[i] for i in range(0, len(components))]
    new_mask = np.zeros_like(mask)
    new_components = []
    index = 1
    mapping = list()

    for component in components:
        if component["area"] < min_area:
            continue

        box = component["bbox"]
        h, w = abs(box[2] - box[0]), abs(box[3] - box[1])
        if h < min_size or h >= mask.shape[0]:
            continue
        if w < min_size or w >= mask.shape[1]:
            continue

        mapping.append(component["label"])
        new_component = copy.deepcopy(component)
        new_component["label"] = index
        coords = new_component["coords"]
        new_mask[coords[:, 0], coords[:, 1]] = index
        new_components.append(new_component)
        index += 1
    return new_mask, new_components, mapping


def extract_features(mask, components, mean, std, size):
    mask, components, mapping = adapt_component_mask(mask, components)
    graph = build_neighbor_graph(mask)
    mask = resize_mask(mask, components, size).astype(np.int32)

    # get features
    features = get_moment_features(components, mask)
    features = (features - mean) / std
    features = torch.tensor(features).float().unsqueeze(0)
    mask = torch.tensor(mask).long().unsqueeze(0)
    return mask, components, graph, features, mapping


def get_input_data(ref_mask, ref_components, tgt_mask, tgt_components, size, mean, std):
    # Extract features
    ref_mask, ref_components, ref_graph, ref_features, ref_mapping = extract_features(
        ref_mask, ref_components, mean, std, size)
    tgt_mask, tgt_components, tgt_graph, tgt_features, tgt_mapping = extract_features(
        tgt_mask, tgt_components, mean, std, size)

    ref_list = [ref_features, ref_mask, ref_components, ref_graph, ref_mapping]
    tgt_list = [tgt_features, tgt_mask, tgt_components, tgt_graph, tgt_mapping]
    return ref_list, tgt_list


def get_component_color(components, color_image):
    for component in components.values():
        coords = component["coords"]
        points = color_image[coords[:, 0], coords[:, 1]]

        unique, counts = np.unique(points, return_counts=True, axis=0)
        max_index = np.argmax(counts)
        color = unique[max_index].tolist()
        component["color"] = color
    return


class ColorWrapper:
    def __init__(self, weight_path):
        self.image_size = (768, 512)
        self.mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
        self.std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]

        self.mean = np.array(self.mean)[:, np.newaxis][:, np.newaxis]
        self.std = np.array(self.std)[:, np.newaxis][:, np.newaxis]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet(8, 0.0)
        self.model.load_state_dict(torch.load(weight_path)["model"])
        self.model.to(self.device)
        self.model.eval()

        self.pairs = self.distances = None

    def process(self, ref_mask, ref_components, tgt_mask, tgt_components):
        new_tgt_components = [tgt_components[i] for i in range(0, len(tgt_components))]
        new_ref_components = [ref_components[i] for i in range(0, len(ref_components))]

        max_tgt_label = max([c["label"] for c in new_tgt_components])
        max_ref_label = max([c["label"] for c in new_ref_components])
        self.distances = np.full([max_tgt_label + 1, max_ref_label + 1], np.inf)

        # Get data
        ref_list, tgt_list = get_input_data(
            ref_mask, new_ref_components, tgt_mask, new_tgt_components, self.image_size, self.mean, self.std)
        ref_features, ref_mask, new_ref_components, ref_graph, ref_mapping = ref_list
        tgt_features, tgt_mask, new_tgt_components, tgt_graph, tgt_mapping = tgt_list

        # Run the model
        with torch.no_grad():
            ref_features = ref_features.float().to(self.device)
            ref_mask = ref_mask.to(self.device)
            tgt_features = tgt_features.float().to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            ref_latent = self.model(ref_features, ref_mask)
            tgt_latent = self.model(tgt_features, tgt_mask)

        pairs, distances = complex_matching_with_distance(
            ref_latent, tgt_latent, ref_graph, tgt_graph, new_ref_components, new_tgt_components)
        self.pairs = dict()

        for tgt_index, ref_index in pairs:
            self.pairs[tgt_mapping[tgt_index]] = ref_mapping[ref_index]
            self.distances[tgt_mapping[tgt_index], ref_mapping[ref_index]] = distances[tgt_index, ref_index]
        return self.pairs, self.distances


def main():
    weight_path = "D:/Data/GeekToys/output/checkpoints/full_size/model_hyper.pth"
    sketch_path = "D:/Data/GeekToys/coloring_data/PD15_samples/PD15_132_R_k_a_R/sketch/a0002.tga"
    ref_sketch_path = "D:/Data/GeekToys/coloring_data/PD15_samples/PD15_132_R_k_a_R/sketch/a0001.tga"
    ref_path = "D:/Data/GeekToys/coloring_data/PD15_samples/PD15_132_R_k_a_R/color/a0001.tga"
    output_dir = "D:/Data/GeekToys/output/rules"

    sketch = read_image(sketch_path)
    tgt_mask, tgt_components = ComponentWrapper().process(sketch, is_gray=True)[:2]

    ref_image = cv2.cvtColor(np.array(Image.open(ref_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "input.png"), ref_image)
    ref_mask, ref_components = ComponentWrapper().process(ref_image, is_gray=False)[:2]
    get_component_color(ref_components, ref_image)

    wrapper = ColorWrapper(weight_path)
    pairs, distances = wrapper.process(ref_mask, ref_components, tgt_mask, tgt_components)
    print(len(pairs))
    print(distances.shape, distances.min(), distances.max())

    frame = cv2.cvtColor(np.array(Image.open(sketch_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    for tgt_index, ref_index in pairs.items():
        tgt_component = [c for c in tgt_components.values() if c["label"] == tgt_index][0]
        ref_component = [c for c in ref_components.values() if c["label"] == ref_index][0]

        coords = tgt_component["coords"]
        frame[coords[:, 0], coords[:, 1], :] = ref_component["color"]
    cv2.imwrite(os.path.join(output_dir, "output.png"), frame)


if __name__ == "__main__":
    main()
