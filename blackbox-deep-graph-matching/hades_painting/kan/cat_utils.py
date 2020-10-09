import os
import shutil
from multiprocessing import Pool, cpu_count

import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from PIL import Image


def cv2_imgshow(image):
    plt.imshow(image)
    plt.show()


def run_multi_process(f, args, n_cpu=-1):
    if n_cpu == -1: n_cpu = cpu_count()

    print(n_cpu)
    p = Pool(n_cpu)
    output = p.map(f, args)

    return output


def find_bound(color_image, sketch):
    sum_invert = np.sum(color_image, axis=-1)
    sum_invert = np.sum(sum_invert, axis=-1)
    coords = np.where(sum_invert > 0)[0]
    ref_bound = [coords[0], coords[-1]]

    sum_sketch = np.sum(255 - sketch, axis=-1)
    sum_sketch = np.sum(sum_sketch, axis=-1)
    coords = np.where(sum_sketch > 0)[0]
    target_bound = [coords[0], coords[-1]]

    i = 0
    for i in range(0, len(coords)):
        if coords[i] >= ref_bound[0]:
            target_bound[0] = coords[i]
            break

    j = 0
    for j in range(len(coords) - 1, -1, -1):
        if coords[j] <= ref_bound[1]:
            target_bound[1] = coords[j]
            break

    sketch_bound = [target_bound[0], target_bound[1]]

    for ii in range(i - 1, -1, -1):
        if coords[ii] == sketch_bound[0] - 1:
            sketch_bound[0] = coords[ii]
        else:
            break

    for jj in range(j + 1, len(coords)):
        if coords[jj] == sketch_bound[1] + 1:
            sketch_bound[1] = coords[jj]
        else:
            break

    return ref_bound, target_bound, sketch_bound


def crop_image(img_lst, upper_bound, lower_bound):
    """Crops all the images in the image list.

    Args:
        img_lst (numpy array list): list of image arrays to be cropped
        upper_bound (int): the starting row of the cropped image
        lower_bound (int): the end row of the cropped image

    Returns:
        numpy array list: list of all the cropped images
    """
    crop_list = []
    for img in img_lst:
        crop_part = img[upper_bound:lower_bound].copy()
        crop_list.append(crop_part)
    return crop_list


def draw_bound_line(sketch, color_image, ref_bound, target_bound, sketch_bound,
                    n=5, k=0.3, q=0.4, max_height=200, edge=50, margin=5):
    if sketch_bound[1] - target_bound[1] > max_height:
        return sketch

    bound = [min(target_bound[0], sketch_bound[0]), max(target_bound[1], sketch_bound[1])]
    for index, line in enumerate(bound):
        if margin <= line <= sketch.shape[0] - margin:
            continue

        ref_line = ref_bound[index]
        start, end = max(ref_line - n, 0), min(ref_line + n, sketch.shape[0])
        limit = line + 1

        region = sketch[line:limit]
        region = (np.sum(region, axis=-1) / (255 * 3)).astype(np.uint8)

        labels = skimage.measure.label(region, connectivity=1, background=0)
        segments = skimage.measure.regionprops(labels, intensity_image=region)

        for segment in segments:
            area, box = segment.area, segment.bbox
            if box[1] < edge or box[3] > sketch.shape[1] - edge:
                if abs(box[3] - box[1]) > sketch.shape[1] * q:
                    continue

            ref_region = color_image[start:end, box[1]:box[3]]
            ref_region = (np.sum(ref_region, axis=0, keepdims=True) / ref_region.shape[0]).astype(np.uint8)
            ref_region = np.sum(np.full_like(ref_region, 255) - ref_region, axis=-1)
            count_color = np.count_nonzero(ref_region)
            if count_color < area * k:
                continue
            limit = line + 1
            sketch[line:limit, box[1]:box[3], :] = 0

    return sketch


def convert_channels(image, mode):
    assert mode in ["3to1", "1to3"]
    if mode == "3to1" and image.ndim == 3:
        image = image[..., 0]
    elif mode == "1to3" and image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def huge_preprocess_images(sketch, ref_sketch, color_image):
    assert sketch.shape[0] == color_image.shape[0] and sketch.shape[1] == color_image.shape[1], "Incompatible size"

    sketch, ref_sketch = convert_channels(sketch, "1to3"), convert_channels(ref_sketch, "1to3")
    invert_color_image = 255 - color_image
    ref_bound, target_bound, sketch_bound = find_bound(invert_color_image, sketch)

    sketch = draw_bound_line(sketch, color_image, ref_bound, target_bound, sketch_bound)
    ref_sketch = draw_bound_line(ref_sketch, color_image, ref_bound, target_bound, sketch_bound)

    sketch, ref_sketch = convert_channels(sketch, "3to1"), convert_channels(ref_sketch, "3to1")
    upper_bound = min(ref_bound[0], target_bound[0])
    lower_bound = max(ref_bound[1], target_bound[1], sketch_bound[1])

    return [sketch, ref_sketch, color_image], (upper_bound, lower_bound)


def find_bound_horizontal(color_image, sketch):
    sum_invert = np.sum(color_image, axis=-1)
    sum_invert = np.sum(sum_invert, axis=0)
    coords = np.where(sum_invert > 0)[0]
    ref_bound = [coords[0], coords[-1]]

    sum_sketch = np.sum(255 - sketch, axis=-1)
    sum_sketch = np.sum(sum_sketch, axis=0)
    coords = np.where(sum_sketch > 0)[0]
    target_bound = [coords[0], coords[-1]]

    for i in range(0, len(coords)):
        if coords[i] >= ref_bound[0]:
            target_bound[0] = coords[i]
            break

    for i in range(len(coords) - 1, -1, -1):
        if coords[i] <= ref_bound[1]:
            target_bound[1] = coords[i]
            break

    return target_bound


def get_horizontal_bound(sketch, color_image):
    assert sketch.shape[0] == color_image.shape[0] and sketch.shape[1] == color_image.shape[1], "Incompatible size"
    sketch = convert_channels(sketch, "1to3")

    invert_color_image = 255 - color_image
    horizontal_bound = find_bound_horizontal(invert_color_image, sketch)
    return horizontal_bound


def pad_img(colored_sketch, bound, original_shape):
    """Pads the sketch so that its shape equals the original shape.

    Args:
        colored_sketch (numpy array): image array (RGB)
        bound (tuple (int, int)): (upper_bound, lower_bound)
        original_shape(tuple): (height, width, color_channels)

    Returns:
        numpy array: padded image (RGB)
    """
    upper_bound, lower_bound = bound
    new_sketch = np.ones(original_shape, np.uint8) * 255
    new_sketch[upper_bound:lower_bound] = colored_sketch
    return new_sketch


def check_filenames(img_paths, sketch_paths):
    """Raise AssertionError if any sketch path does not have the same basename as its corresponding image path.

    Args:
        img_paths (list): list of image paths
        sketch_paths (list): list of sketch paths
    """
    for i, img_path in enumerate(img_paths):
        img_fn = os.path.basename(img_path)
        sketch_fn = os.path.basename(sketch_paths[i])
        assert img_fn == sketch_fn, 'Sketch and image folders contain differently named pair.'


def make_new_folder(folder_fn):
    """Create a new folder. If the folder already exists, delete it and create a new one."""
    if os.path.isdir(folder_fn):
        shutil.rmtree(folder_fn)
    os.makedirs(folder_fn)


def imread(img_path, grayscale=False):
    """Load the image from img_path.

    Args:
        img_path (str): path to the image file
        grayscale (bool): True if the image should be in grayscale, False otherwise

    Returns:
        numpy array: the image array
    """
    if grayscale:
        return np.asarray(Image.open(img_path).convert('L'))
    img = np.asarray(Image.open(img_path))
    return img[:, :, :3]


def imsave(result, output_dir, img_name):
    """Save the image.

    Args:
        result (numpy array): image array to be saved
        output_dir (str): path to output folder
        img_name (str): basename of the image (e.g. output_1.png)
    """
    Image.fromarray(result).save('%s/%s' % (output_dir, img_name))


def check_dimensions(arr_list):
    """Check if all image arrays have the identical heights and widths.

    Args:
        arr_list (numpy array list): list of image arrays

    Returns:
        bool: True if all arrays have identical heights and widths, False otherwise
    """
    shape = arr_list[0].shape
    for arr in arr_list[1:]:
        if arr.shape[0] != shape[0] or arr.shape[1] != shape[1]:
            return False
        shape = arr.shape
    return True
