import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
from natsort import natsorted
from PIL import Image
from rules.color_component_matching import ComponentWrapper, ShapeMatchingWrapper, resize_mask
from rules.color_component_matching import get_component_color
from rules.component_wrapper import get_moment_features


def get_image_by_index(paths, index):
    if index is None:
        return None
    path = paths[index]

    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image, path


def draw_component_image(components, mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for component in components:
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = component["color"]

    cv2.imwrite("%d.png" % len(components), image)


def match_components_three_stage(components_a, components_b, matcher, is_removed):
    pairs = []

    for index_a, a in enumerate(components_a):
        matches = [(b, matcher.process(a, b)) for b in components_b]
        count_true = len([1 for match in matches if match[1][0]])
        if count_true == 0:
            continue

        distances = np.array([match[1][1] for match in matches])
        index_b = int(np.argmin(distances))
        pairs.append([index_a, index_b])

    if len(pairs) == 0:
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, threshold=0.2))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    if len(pairs) == 0 and (not is_removed):
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, pos_filter=False, threshold=0.6))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    return pairs


def loader_collate(batch):
    assert len(batch) == 1
    batch = batch[0]

    features_a = torch.tensor(batch[0]).unsqueeze(0).float()
    mask_a = torch.tensor(batch[1]).unsqueeze(0).float()
    features_b = torch.tensor(batch[2]).unsqueeze(0).float()
    mask_b = torch.tensor(batch[3]).unsqueeze(0).float()

    positive_pairs = torch.tensor(batch[4]).unsqueeze(0).int()
    colors_a = torch.tensor(batch[7]).unsqueeze(0).int()
    colors_b = torch.tensor(batch[8]).unsqueeze(0).int()

    return (features_a, mask_a, colors_a, batch[5]), (features_b, mask_b, colors_b, batch[6]), positive_pairs


def random_remove_component(mask, components, max_removed=2):
    if len(components) < 10 or random.random() < 0.6:
        return mask, components, False

    index = 1
    new_components = []
    new_mask = np.zeros(mask.shape, dtype=np.int)
    removed = 0

    for component in components:
        if random.random() > 0.05 or removed >= max_removed:
            component["label"] = index
            new_components.append(component)
            new_mask[component["coords"][:, 0], component["coords"][:, 1]] = index
            index += 1
        else:
            removed += 1
    return new_mask, new_components, removed > 0


def add_random_noise(features, mask):
    noise = np.random.normal(loc=0.0, scale=0.02, size=features.shape)
    bool_mask = mask > 0
    features = features + noise * bool_mask
    return features


class PairAnimeDataset(data.Dataset):
    def __init__(self, root_dir, size, mean, std):
        super(PairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size
        self.mean = mean
        self.std = std

        self.paths = {}
        self.lengths = {}
        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))

        self.component_wrapper = ComponentWrapper(min_area=10, min_size=3)
        self.matcher = ShapeMatchingWrapper()

        for sub_dir in dirs:
            dir_name = os.path.basename(sub_dir)
            self.paths[dir_name] = {}

            for set_name in ["sketch_v3", "color"]:
                paths = []
                for sub_type in ["png", "jpg", "tga"]:
                    paths.extend(glob.glob(os.path.join(sub_dir, set_name, "*.%s" % sub_type)))
                self.paths[dir_name][set_name] = natsorted(paths)

            self.lengths[dir_name] = len(self.paths[dir_name]["color"])
        return

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def get_component_mask(self, color_image, sketch, path, extract_prob=0.4):
        is_pd = any([(w in path) for w in ["PD09", "PD10"]])
        if is_pd:
            method = ComponentWrapper.EXTRACT_COLOR
        else:
            if random.random() < extract_prob:
                method = ComponentWrapper.EXTRACT_SKETCH
            else:
                method = ComponentWrapper.EXTRACT_COLOR

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            mask, components = self.component_wrapper.process(color_image, sketch, method)
            get_component_color(components, color_image, ComponentWrapper.EXTRACT_COLOR)
            save_data = {"mask": mask, "components": components}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            mask, components = save_data["mask"], save_data["components"]

        mask, components, is_removed = random_remove_component(mask, components)
        mask = resize_mask(mask, components, self.size).astype(np.int32)
        return mask, components, is_removed

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        length = len(self.paths[name]["color"])
        k = random.choice([1, 1, 1, 2])
        next_index = max(index - k, 0) if index == length - 1 else min(index + k, length - 1)

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)
        is_pd = any([(w in path_a and w in path_b) for w in ["PD09", "PD10"]])

        if is_pd:
            sketch_a = sketch_b = None
        else:
            sketch_a = get_image_by_index(self.paths[name]["sketch_v3"], index)[0]
            sketch_b = get_image_by_index(self.paths[name]["sketch_v3"], next_index)[0]

        # extract components
        mask_a, components_a, is_removed_a = self.get_component_mask(color_a, sketch_a, path_a)
        mask_b, components_b, is_removed_b = self.get_component_mask(color_b, sketch_b, path_b)
        is_removed = is_removed_a or is_removed_b

        # component matching
        positive_pairs = match_components_three_stage(components_a, components_b, self.matcher, is_removed)
        positive_pairs = np.array(positive_pairs)

        # component color
        colors_a = [a["color"] for a in components_a]
        colors_b = [b["color"] for b in components_b]
        colors_a, colors_b = np.array(colors_a), np.array(colors_b)

        if len(positive_pairs) == 0 or len(components_a) == 0 or len(components_b) == 0:
            print(name, index, next_index)
        if np.max(mask_a) == 0 or np.max(mask_b) == 0:
            print(name, index)

        # get features
        features_a = get_moment_features(components_a, mask_a)
        features_b = get_moment_features(components_b, mask_b)
        features_a = (features_a - self.mean) / self.std
        features_b = (features_b - self.mean) / self.std
        features_a = add_random_noise(features_a, mask_a)
        features_b = add_random_noise(features_b, mask_b)

        return features_a, mask_a, features_b, mask_b, positive_pairs, components_a, components_b, colors_a, colors_b
