import os
import glob
import random
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image


def collect_boy_shots():
    root_dir = "D:/Data/GeekToys/coloring_data"
    output_dir = "D:/Data/GeekToys/coloring_data/boy_collection"
    list_path = os.path.join(output_dir, "boy_collection.txt")

    boy_cuts = []
    with open(list_path) as file:
        for line in file:
            boy_cuts.append(line.strip())

    for set_name in ["complete_data", "different_data"]:
        cuts = natsorted(glob.glob(os.path.join(root_dir, set_name, "*")))
        cuts = [cut for cut in cuts if os.path.basename(cut) in boy_cuts]

        for cut in cuts:
            cut_name = os.path.basename(cut)
            os.makedirs(os.path.join(output_dir, cut_name), exist_ok=True)

            paths = natsorted(glob.glob(os.path.join(cut, "color", "*.tga")))
            for path in paths:
                name = os.path.splitext(os.path.basename(path))[0]
                image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, cut_name, "%s.png" % name), image)
    return


def make_geek_conversion():
    output_dir = "D:/Data/GeekToys/pose_data/geek_conversion_dataset"
    """
    root_dir = "D:/Data/GeekToys/coloring_data"

    for set_name in ["complete_data", "different_data"]:
        paths = natsorted(glob.glob(os.path.join(root_dir, set_name, "*", "color", "*.tga")))
        for path in paths:
            cut_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
            name = os.path.splitext(os.path.basename(path))[0]
            image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "trainA", "%s_%s.png" % (cut_name, name)), image)
    """

    root_dir = "D:/Data/GeekToys/pose_data/fashion_dataset/train"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    length = len(glob.glob(os.path.join(output_dir, "trainB", "*.png")))
    max_count = 2000

    for path in paths:
        if length < max_count and random.random() < 0.04:
            name = os.path.splitext(os.path.basename(path))[0]
            image = cv2.imread(path)
            path = os.path.join(output_dir, "trainB", "%s.png" % name)
            if not os.path.exists(path):
                cv2.imwrite(path, image)
                length += 1
    return


def remove_pickle():
    root_dir = "D:/Data/GeekToys/coloring_data/full_data"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.pkl")))
    print(len(paths))

    for path in paths:
        os.remove(path)
    return


def read_image(path):
    if path.endswith(".tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def get_sample_item():
    root_dir = "D:/Data/GeekToys/pose_data/geek_source"
    output_dir = "D:/Data/GeekToys/pose_data/pifu_sample"
    size = (512, 512)

    paths = natsorted(glob.glob(os.path.join(root_dir, "*.tga")))
    paths += natsorted(glob.glob(os.path.join(root_dir, "*.png")))
    paths += natsorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    print(len(paths))

    for path in paths[:14]:
        name = os.path.splitext(os.path.basename(path))[0]
        image = read_image(path)
        image[np.where((image == [246, 246, 246]).all(axis=-1))] = [255, 255, 255]
        image[np.where((image == [254, 254, 254]).all(axis=-1))] = [255, 255, 255]

        image_sum = np.sum(255 - image, axis=-1)
        sum_y = np.sum(image_sum, axis=-1)
        ys = np.where(sum_y > 0)[0]
        top, bottom = min(ys), max(ys)
        sum_x = np.sum(image_sum, axis=0)
        xs = np.where(sum_x > 0)[0]
        left, right = min(xs), max(xs)

        left, right = max(left - 200, 0), min(right + 200, image.shape[1])
        top, bottom = max(top - 100, 0), min(bottom + 100, image.shape[0])
        image = image[top:bottom, left:right]

        if image.shape[0] < image.shape[1]:
            margin = abs(image.shape[1] - image.shape[0]) // 2
            image = image[margin:-margin]
        elif image.shape[0] > image.shape[1]:
            margin = abs(image.shape[1] - image.shape[0]) // 2
            white_space = np.full([image.shape[0], margin, 3], 255)
            image = np.concatenate([white_space, image, white_space], axis=1)

        image = np.ascontiguousarray(image).astype(np.uint8)
        image_sum = np.sum(image, axis=-1)
        mask = (image_sum < 765).astype(np.uint8) * 255

        image[np.where((image == [255, 255, 255]).all(axis=-1))] = [0, 0, 0]
        image = cv2.resize(image, size, cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "%s.png" % name), image)
        mask = cv2.resize(mask, size, cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "%s_mask.png" % name), mask)
    return


def get_multi_view_item():
    root_dir = "D:/Documents/Cinnamon/painting/PIFu/sample_images/goku_01"
    output_dir = "D:/Documents/Cinnamon/painting/PIFu/sample_images/goku_02_out"
    output_dir = "D:/Data/GeekToys/pose_data/boy_little"
    size = (256, 256)

    paths = natsorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    print(len(paths))

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = read_image(path)

        if image.shape[0] < image.shape[1]:
            margin = abs(image.shape[1] - image.shape[0]) // 2
            image = image[margin:-margin]
        elif image.shape[0] > image.shape[1]:
            margin = abs(image.shape[1] - image.shape[0]) // 2
            white_space = np.full([image.shape[0], margin, 3], 255)
            image = np.concatenate([white_space, image, white_space], axis=1)

        image = np.ascontiguousarray(image).astype(np.uint8)
        image_sum = np.sum(image, axis=-1)
        mask = (image_sum < 720).astype(np.uint8) * 255

        image[np.where(image_sum >= 720)] = [255, 255, 255]
        image = cv2.resize(image, size, cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "%s.png" % name), image)
        mask = cv2.resize(mask, size, cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "%s_mask.png" % name), mask)
    return


if __name__ == "__main__":
    get_multi_view_item()
