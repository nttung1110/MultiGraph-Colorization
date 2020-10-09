import os
import glob
import random
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
import sys 
sys.path.append("/mnt/ai_filestore/home/zed/multi-graph-matching/hades_painting")
from hades_painting.rules.component_wrapper import ComponentWrapper, get_component_color, resize_mask
from hades_painting.rules.shape_matching_wrapper import ShapeMatchingWrapper


def match_components_three_stage(components_a, components_b, matcher):
    pairs = []
    for a in components_a:
        print(a["label"])
        matches = [(b, matcher.process(a, b)) for b in components_b]
        count_true = len([1 for match in matches if match[1][0]])
        if count_true == 0:
            continue

        distances = np.array([match[1][1] for match in matches])
        index = int(np.argmin(distances))
        pairs.append((a, matches[index][0]))

    if len(pairs) == 0:
        for a in components_a:
            matches = [(b, matcher.process(a, b, area_filter=False, threshold=0.2))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index = int(np.argmin(distances))
            pairs.append((a, matches[index][0]))

    if len(pairs) == 0:
        for a in components_a:
            matches = [(b, matcher.process(a, b, area_filter=False, pos_filter=False, threshold=0.6))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index = int(np.argmin(distances))
            pairs.append((a, matches[index][0]))

    return pairs


def get_part_and_color(component):
    image = component["image"]

    color_block = np.tile(component["color"], (image.shape[0], image.shape[1], 1))
    image = np.where(
        np.stack([image == 0] * 3, axis=-1), np.zeros_like(color_block), color_block)
    image = np.ascontiguousarray(image.astype(np.uint8))
    return image


def main():
    root_dir = "D:/Data/GeekToys/coloring_data/complete_data"
    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    character_dir = random.choice(character_dirs)

    sketch_image_paths = natsorted(glob.glob(os.path.join(character_dir, "sketch_v3", "*.png")))[:2]
    color_image_paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))[:2]

    output_dir = "D:/Data/GeekToys/output/rules"
    size = (768, 512)
    component_wrapper = ComponentWrapper()
    matcher = ShapeMatchingWrapper()

    # read images
    sketch_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in sketch_image_paths]
    color_images = [
        cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR) for path in color_image_paths]

    color_images = [np.where(
        np.stack([sketch_image == 0] * 3, axis=-1), np.zeros_like(color_image), color_image)
        for sketch_image, color_image in zip(sketch_images, color_images)]
    color_images = [cv2.resize(image, (size[0], size[1])) for image in color_images]

    # extract components
    mask_a, components_a = component_wrapper.process(color_images[0])
    get_component_color(components_a, color_images[0])
    mask_b, components_b = component_wrapper.process(color_images[1])
    get_component_color(components_b, color_images[1])

    # component matching
    pairs = match_components_three_stage(components_a, components_b, matcher)

    # draw matches
    print(len(pairs), len(components_a))
    for pair in pairs:
        first_image = get_part_and_color(pair[0])
        second_image = get_part_and_color(pair[1])
        second_image = cv2.resize(second_image, (first_image.shape[1], first_image.shape[0]))

        pair_image = np.concatenate([first_image, second_image], axis=1)
        name = "%d_%d.png" % (pair[0]["label"], pair[1]["label"])
        cv2.imwrite(os.path.join(output_dir, name), pair_image)

    cv2.imwrite(os.path.join(output_dir, "mask_a.png"), mask_a)
    cv2.imwrite(os.path.join(output_dir, "mask_b.png"), mask_b)


def check_part_size():
    root_dir = "D:/Data/GeekToys/coloring_data/complete_data"
    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))

    size = (768, 512)
    component_wrapper = ComponentWrapper()

    for index, character_dir in enumerate(character_dirs):
        sketch_image_paths = natsorted(glob.glob(os.path.join(character_dir, "sketch_v3", "*.png")))
        color_image_paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        print(index)

        for sketch_image_path, color_image_path in zip(sketch_image_paths, color_image_paths):
            sketch_image = cv2.imread(sketch_image_path, cv2.IMREAD_GRAYSCALE)
            color_image = cv2.cvtColor(np.array(Image.open(color_image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

            color_image = np.where(
                np.stack([sketch_image == 0] * 3, axis=-1), np.zeros_like(color_image), color_image)
            color_image = cv2.resize(color_image, (size[0], size[1]))
            mask, components = component_wrapper.process(color_image)

            if len(components) == 0 or np.max(mask) == 0:
                print(character_dir, color_image_path)
    return


def move_data():
    import shutil

    input_dir = "D:/Data/GeekToys/coloring_data/different_data"
    output_dir = "D:/Data/GeekToys/coloring_data/move"

    character_dirs = natsorted(glob.glob(os.path.join(input_dir, "*")))
    character_dirs = [d for d in character_dirs if "hor01" in d]
    print(len(character_dirs))

    random_dirs = random.choices(character_dirs, k=1)
    for character_dir in random_dirs:
        shutil.move(character_dir, os.path.join(output_dir, os.path.basename(character_dir)))
    return


if __name__ == "__main__":
    move_data()
