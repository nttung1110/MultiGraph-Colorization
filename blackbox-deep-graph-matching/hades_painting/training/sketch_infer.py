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
from scipy.optimize import linear_sum_assignment

from kan.denoise import pre_denoise
from kan.closing import ClosingModel
from kan.cat_utils import huge_preprocess_images, get_horizontal_bound
from kan.component_wrapper import ComponentWrapper

from rules.color_component_matching import get_component_color
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


def adapt_component_mask(mask, components, y_bound, x_bound, min_area=15, min_size=3):
    components = [components[i] for i in range(0, len(components))][1:]
    new_mask = np.zeros_like(mask)
    new_components, small_components = [], []
    index = 1

    for component in components:
        if component["area"] < min_area:
            small_components.append(component)
            continue

        box = component["bbox"]
        h, w = abs(box[2] - box[0]), abs(box[3] - box[1])
        if h < min_size or w < min_size:
            small_components.append(component)
            continue
        if h >= mask.shape[0] or w >= mask.shape[1]:
            continue
        if box[2] < y_bound[0] or box[0] > y_bound[1]:
            continue
        if box[3] < x_bound[0] or box[1] > x_bound[1]:
            continue

        component["old_label"] = component["label"]
        component["label"] = index
        coords = component["coords"]
        new_mask[coords[:, 0], coords[:, 1]] = index
        new_components.append(component)
        index += 1

    removed_data = {"mask": mask, "components": small_components}
    return new_mask, new_components, removed_data


def close_sketch(sketch):
    closing_model = ClosingModel()
    sketch = np.ascontiguousarray(sketch)
    color = 0

    pair_points = closing_model.process(sketch)[0]
    for p1, p2 in pair_points:
        cv2.line(sketch, (p1[1], p1[0]), (p2[1], p2[0]), color=color, thickness=1)
    return sketch


def close_sketch_with_color_image(ref_sketch, ref_color_image, k=0.9):
    from skimage.measure import find_contours
    from skimage.morphology import skeletonize

    b, g, r = cv2.split(ref_color_image)
    b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)
    contour_image = np.zeros_like(ref_sketch)
    invert_sketch = (255 - ref_sketch) / 255

    processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
    uniques = np.unique(processed_image)
    bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 255]]

    for unique in uniques:
        if unique in bad_values:
            continue

        rows, cols = np.where(processed_image == unique)
        image_temp = np.zeros_like(processed_image)
        image_temp[rows, cols] = 255
        image_temp = np.array(image_temp, dtype=np.uint8)

        contours = find_contours(image_temp, level=0)
        part_contour_image = np.zeros_like(ref_sketch)
        for contour in contours:
            contour = contour.astype(np.int)
            part_contour_image[contour[:, 0], contour[:, 1]] = 1
            part_contour_image = skeletonize(part_contour_image)

            total_points = np.sum(part_contour_image)
            overlap_points = np.sum(part_contour_image * invert_sketch)
            ratio = overlap_points / total_points

            if ratio > k:
                contour_image = np.clip(contour_image + part_contour_image, a_min=0, a_max=1)

    ref_sketch = np.where(contour_image, np.zeros_like(ref_sketch), ref_sketch)
    return ref_sketch


def extract_features(sketch, mean, std, size, y_bound, x_bound):
    component_wrapper = ComponentWrapper()
    # extract components
    raw_mask, raw_components = component_wrapper.process(sketch, is_gray=True)[:2]
    mask, components, removed = adapt_component_mask(raw_mask, raw_components, y_bound, x_bound)
    graph = build_neighbor_graph(mask)
    mask = resize_mask(mask, components, size).astype(np.int32)

    # get features
    features = get_moment_features(components, mask)
    features = (features - mean) / std
    features = torch.tensor(features).float().unsqueeze(0)
    mask = torch.tensor(mask).long().unsqueeze(0)
    return mask, components, graph, features, removed


def pad_sketch(sketch):
    sketch = cv2.copyMakeBorder(sketch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255)
    return sketch


def get_input_data(path, sketch_path, reference_path, ref_sketch_path, size, mean, std):
    closing, padding = True, True

    sketch = read_image(sketch_path)
    color_image = read_image(path)
    ref_sketch = read_image(ref_sketch_path)
    ref_color_image = read_image(reference_path)

    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY) if sketch.ndim == 3 else sketch
    sketch = cv2.threshold(sketch, 250, 255, cv2.THRESH_BINARY)[1]
    ref_sketch = cv2.cvtColor(ref_sketch, cv2.COLOR_BGR2GRAY) if ref_sketch.ndim == 3 else ref_sketch
    ref_sketch = cv2.threshold(ref_sketch, 250, 255, cv2.THRESH_BINARY)[1]
    ref_sketch = close_sketch_with_color_image(ref_sketch, ref_color_image)

    # Pre-processing
    if closing:
        sketch, color_image = pre_denoise(sketch, color_image)
        ref_sketch, ref_color_image = pre_denoise(ref_sketch, ref_color_image)
        sketch, ref_sketch = close_sketch(sketch), close_sketch(ref_sketch)

        # Close huge openings
        image_list, y_bound = huge_preprocess_images(sketch, ref_sketch, ref_color_image)
        sketch, ref_sketch, ref_color_image = image_list
        x_bound = get_horizontal_bound(sketch, ref_color_image)
    else:
        y_bound = [0, sketch.shape[0]]
        x_bound = [0, sketch.shape[1]]

    # Padding for background component
    if padding:
        sketch = pad_sketch(sketch)
        ref_sketch = pad_sketch(ref_sketch)
        color_image = cv2.copyMakeBorder(color_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        ref_color_image = cv2.copyMakeBorder(ref_color_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (255, 255, 255))

    # Extract features
    mask_a, components_a, graph_a, features_a = extract_features(ref_sketch, mean, std, size, y_bound, x_bound)[:4]
    mask_b, components_b, graph_b, features_b, removed = extract_features(sketch, mean, std, size, y_bound, x_bound)
    get_component_color(components_a, ref_color_image)
    get_component_color(components_b, color_image)

    frame = np.stack([sketch] * 3, axis=-1)
    a_list = [features_a, mask_a, components_a, graph_a, ref_sketch, ref_color_image]
    b_list = [features_b, mask_b, components_b, graph_b, sketch, color_image]
    return a_list, b_list, frame, removed


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
        component["predicted_color"] = match_color
        image[coords[:, 0], coords[:, 1], :] = match_color
    return image


def draw_small_components(image, removed):
    old_mask, small_components = removed["mask"], removed["components"]

    for small_component in small_components:
        coords = small_component["coords"]
        image[coords[:, 0], coords[:, 1], :] = np.array([0, 0, 0])
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
    min_area, min_size = 15, 3
    component_wrapper = ComponentWrapper()
    image_paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))

    candidates = []
    for image_path in image_paths:
        image_name = get_base_name(image_path)
        sketch_path = os.path.join(os.path.dirname(image_path), "..", "sketch", "%s.tga" % image_name)

        sketch = read_image(sketch_path)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY) if sketch.ndim == 3 else sketch
        sketch = cv2.threshold(sketch, 250, 255, cv2.THRESH_BINARY)[1]

        sketch = pre_denoise(sketch, np.stack([sketch] * 3, axis=-1))[0]
        sketch = close_sketch(sketch)
        sketch = cv2.copyMakeBorder(sketch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255)

        components = component_wrapper.process(sketch, is_gray=True)[1]
        components = [components[i] for i in range(0, len(components))][1:]
        new_components = []

        for component in components:
            if component["area"] < min_area:
                continue

            box = component["bbox"]
            h, w = abs(box[2] - box[0]), abs(box[3] - box[1])
            if h < min_size or w < min_size:
                continue
            new_components.append(component)

        candidates.append((len(new_components)))

    print(candidates)
    final_candidate = int(np.argmax(candidates))
    return final_candidate


def evaluate_colorization(components):
    true_count = 0
    for component in components:
        if all(component["color"] == component["predicted_color"]):
            true_count += 1
    return true_count, len(components)


def main(args):
    device = torch.device("cuda:0")
    total_true_count, total_length = 0, 0

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
    print(len(character_dirs))
    for character_dir in character_dirs:
        if "hor02_182" not in character_dir:
            continue

        character_name = os.path.basename(character_dir)
        if not os.path.exists(os.path.join(args.output_dir, character_name)):
            os.makedirs(os.path.join(args.output_dir, character_name))

        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        reference_index = pick_reference_sketch(character_dir)
        reference_path = paths[reference_index]
        print(character_dir, reference_index, reference_path)

        for path in paths:
            if path == reference_path:
                ref_image = read_image(reference_path)
                cv2.imwrite(os.path.join(args.output_dir, character_name, "reference.png"), ref_image)
                continue
            print(path)

            image_name = get_base_name(path)
            reference_name = get_base_name(reference_path)
            sketch_path = os.path.join(os.path.dirname(path), "..", "sketch", "%s.tga" % image_name)
            reference_sketch_path = os.path.join(os.path.dirname(path), "..", "sketch", "%s.tga" % reference_name)
            if not os.path.exists(sketch_path):
                continue

            # Get data
            a_list, b_list, frame, removed = get_input_data(
                path, sketch_path, reference_path, reference_sketch_path, image_size, mean, std)
            features_a, mask_a, components_a, graph_a, sketch_a, color_a = a_list
            features_b, mask_b, components_b, graph_b, sketch_b, color_b = b_list
            if torch.max(mask_a).item() == 0 or torch.max(mask_b).item() == 0:
                print(path, reference_path)
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
            output_image[np.where((frame == [0, 0, 0]).all(axis=-1))] = [0, 0, 0]
            output_image = output_image[1:-1, 1:-1]
            cv2.imwrite(os.path.join(args.output_dir, character_name, "%s.png" % image_name), output_image)

            true_count, length = evaluate_colorization(components_b)
            print(true_count / length)
            total_true_count += true_count
            total_length += length

    print("Summary:", total_true_count / total_length)


if __name__ == "__main__":
    main(parse_arguments())
