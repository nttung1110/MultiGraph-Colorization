import os
import glob
import cv2
import numpy as np
from natsort.natsort import natsorted
from PIL import Image
from rules.component_wrapper import ComponentWrapper, get_component_color


def accuracy_by_component(target, output):
    component_wrapper = ComponentWrapper(min_area=15, min_size=3)
    mask_gt, components_gt = component_wrapper.process(target)
    get_component_color(components_gt, target)
    acc = 0

    for i in range(0, len(components_gt)):
        coords = components_gt[i]["coords"]
        output_pixels = output[coords[:, 0], coords[:, 1], :]
        output_colors = np.unique(output_pixels, axis=0).tolist()

        bad_values = [[x, x, x] for x in [0, 255]]
        good_colors = [color for color in output_colors if color not in bad_values]

        if len(good_colors) == 1:
            output_color = good_colors[0]
        elif len(good_colors) == 0:
            output_color = [0, 0, 0]
        else:
            output_color = [0, 0, 0]

        if output_color == components_gt[i]["color"]:
            acc += 1

    acc = acc / len(components_gt)
    return acc


def accuracy_by_pixel(target, output):
    h, w, _ = target.shape
    b, g, r = cv2.split(output)
    processed_output = b + 300 * (g + 1) + 300 * 300 * (r + 1)

    b, g, r = cv2.split(target)
    processed_target = b + 300 * (g + 1) + 300 * 300 * (r + 1)

    out = processed_target - processed_output
    coord = np.where(out == 0)
    acc = len(coord[0]) / (h * w)
    return acc


def evaluate_one_cut(character_dir, output_dir):
    acc_component = acc_pixel = count = 0
    list_acc_component, list_acc_pixel = [], []

    character_name = os.path.basename(character_dir)
    output_paths = natsorted(glob.glob(os.path.join(output_dir, character_name, "*.png")))

    for output_path in output_paths:
        if "target" in output_path:
            continue

        target_path = os.path.join(character_dir, "color", os.path.basename(output_path).replace("png", "tga"))
        target = cv2.cvtColor(np.array(Image.open(target_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        output = cv2.imread(output_path)
        step_acc_component = accuracy_by_component(target, output)
        step_acc_pixel = accuracy_by_pixel(target, output)

        acc_component += step_acc_component
        acc_pixel += step_acc_pixel
        count += 1
        list_acc_component.append(step_acc_component)
        list_acc_pixel.append(step_acc_pixel)

    return acc_component, acc_pixel, count, list_acc_component, list_acc_pixel


def main():
    test_dir = "D:/Data/GeekToys/coloring_data/different_data"
    output_dir = "D:/Data/GeekToys/output/hades"

    character_dirs = natsorted(glob.glob(os.path.join(test_dir, "*")))
    acc_component = acc_pixel = count = 0
    list_acc_component, list_acc_pixel = [], []

    for character_dir in character_dirs:
        cut_acc_component, cut_acc_pixel, cut_count, lc, lp = evaluate_one_cut(character_dir, output_dir)
        acc_component += cut_acc_component
        acc_pixel += cut_acc_pixel
        count += cut_count

        list_acc_component.extend(lc)
        list_acc_pixel.extend(lp)

    acc_component = acc_component / count
    acc_pixel = acc_pixel / count
    print()
    print("Accuracy by component:", acc_component, count)
    print("Accuracy by pixel:", acc_pixel, count)

    print(len(list_acc_component), np.mean(list_acc_component), np.std(list_acc_component))
    print(len(list_acc_pixel), np.mean(list_acc_pixel), np.std(list_acc_pixel))


if __name__ == "__main__":
    main()
