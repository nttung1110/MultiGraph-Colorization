import numpy as np
import cv2
import torch
from PIL import Image
from models.utils import cosine_distance


def find_nearest_match(anchor, components, graph):
    min_index = -1

    for index, component in components.items():
        is_neighbor = graph[anchor["label"], component["label"]]
        if is_neighbor:
            min_index = index
            break

    if min_index == -1:
        return None, min_index
    return components[min_index], min_index


def read_as_sketch(path):
    image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    image = np.stack([image] * 3, axis=-1)
    return image


def draw_component(component, image):
    box = component["bbox"]
    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 0, 255))
    centroid = component["centroid"]
    cv2.circle(image, (int(centroid[1]), int(centroid[0])), 2, (0, 0, 255), 2)
    return image


def complex_matching(output_a, output_b, graph_a, graph_b, components_a, components_b, k=4):
    top_k_per_region, pairs = [], []
    k = max(min(k, output_a.shape[1] - 2), 1)

    min_area_ratio = 0.6
    distance_threshold = 0.3

    # find top-k for each region
    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])

        distances = cosine_distance(region_b, output_a)
        min_distances, min_indices = torch.topk(distances, k, largest=False, dim=-1)

        top_k_per_region.append([])
        for min_distance, min_index in zip(min_distances.tolist()[0], min_indices.tolist()[0]):
            top_k_per_region[-1].append([min_index, min_distance])

    # highly-likely matching
    for index_b, top_k in enumerate(top_k_per_region):
        component_b = components_b[index_b]
        component_a = components_a[top_k[0][0]]
        if top_k[0][1] > distance_threshold:
            continue

        area_ratio = component_b["area"] / component_a["area"]
        area_ratio = area_ratio if area_ratio < 1 else (1 / area_ratio)
        if area_ratio < min_area_ratio:
            continue
        pairs.append([index_b, top_k[0][0]])

    # neighbor matching
    while True:
        old_len = len(pairs)
        done_indices = [p[0] for p in pairs]
        done_components = {i: components_b[i] for i in done_indices}

        for index_b, top_k in enumerate(top_k_per_region):
            if index_b in done_indices:
                continue

            component_b = components_b[index_b]
            neighbor, index_n = find_nearest_match(component_b, done_components, graph_b)
            if neighbor is None:
                continue

            neighbor_pair = [p for p in pairs if p[0] == index_n][0]
            neighbor_match = components_a[neighbor_pair[1]]

            min_index = -1
            for index_a, _ in top_k:
                component_a = components_a[index_a]
                is_neighbor = graph_a[component_a["label"], neighbor_match["label"]]

                if is_neighbor:
                    min_index = index_a
                    break

            if min_index != -1:
                pairs.append([index_b, min_index])

        new_len = len(pairs)
        if new_len - old_len <= 0:
            break

    # remaining matching
    done_indices = [p[0] for p in pairs]
    for index_b, top_k in enumerate(top_k_per_region):
        if index_b not in done_indices:
            pairs.append([index_b, top_k[0][0]])

    return pairs


def complex_matching_with_distance(output_a, output_b, graph_a, graph_b, components_a, components_b, k=4):
    top_k_per_region, pairs = [], []
    k = max(min(k, output_a.shape[1] - 2), 1)
    print(output_b.shape, output_a.shape)

    min_area_ratio = 0.6
    distance_threshold = 0.3
    all_distances = np.zeros([output_b.shape[1], output_a.shape[1]])

    # find top-k for each region
    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])

        distances = cosine_distance(region_b, output_a)
        all_distances[index_b, :] = distances.cpu().numpy()
        min_distances, min_indices = torch.topk(distances, k, largest=False, dim=-1)

        top_k_per_region.append([])
        for min_distance, min_index in zip(min_distances.tolist()[0], min_indices.tolist()[0]):
            top_k_per_region[-1].append([min_index, min_distance])

    # highly-likely matching
    for index_b, top_k in enumerate(top_k_per_region):
        component_b = components_b[index_b]
        component_a = components_a[top_k[0][0]]
        if top_k[0][1] > distance_threshold:
            continue

        area_ratio = component_b["area"] / component_a["area"]
        area_ratio = area_ratio if area_ratio < 1 else (1 / area_ratio)
        if area_ratio < min_area_ratio:
            continue
        pairs.append([index_b, top_k[0][0]])

    # neighbor matching
    while True:
        old_len = len(pairs)
        done_indices = [p[0] for p in pairs]
        done_components = {i: components_b[i] for i in done_indices}

        for index_b, top_k in enumerate(top_k_per_region):
            if index_b in done_indices:
                continue

            component_b = components_b[index_b]
            neighbor, index_n = find_nearest_match(component_b, done_components, graph_b)
            if neighbor is None:
                continue

            neighbor_pair = [p for p in pairs if p[0] == index_n][0]
            neighbor_match = components_a[neighbor_pair[1]]

            min_index = -1
            for index_a, _ in top_k:
                component_a = components_a[index_a]
                is_neighbor = graph_a[component_a["label"], neighbor_match["label"]]

                if is_neighbor:
                    min_index = index_a
                    break

            if min_index != -1:
                pairs.append([index_b, min_index])

        new_len = len(pairs)
        if new_len - old_len <= 0:
            break

    # remaining matching
    done_indices = [p[0] for p in pairs]
    for index_b, top_k in enumerate(top_k_per_region):
        if index_b not in done_indices:
            pairs.append([index_b, top_k[0][0]])

    return pairs, all_distances
