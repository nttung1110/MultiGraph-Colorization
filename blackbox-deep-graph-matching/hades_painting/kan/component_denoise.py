import numpy as np
import cv2
import skimage.measure as measure
from PIL import Image


def compute_iou(a, b):
    inter_box = [
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    ]
    inter_area = max(0, inter_box[2] - inter_box[0] + 1) * max(0, inter_box[3] - inter_box[1] + 1)
    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    inter = inter_area / a_area
    iou = inter_area / float(a_area + b_area - inter_area)
    return inter, iou


def overlap_with_foreground(region, foreground):
    inter, iou = compute_iou(region["bbox"], foreground["bbox"])
    if inter < 0.8 and iou < 0.5:
        return False

    y_min, x_min, y_max, x_max = region["bbox"]
    sub_fore = foreground["image"][y_min:y_max, x_min:x_max]
    sub_image = region["image"] + sub_fore
    inter = np.count_nonzero(sub_image == 2) / len(region["coords"])
    return inter > 0.6


class ComponentDenoise:
    @staticmethod
    def extract_background(sketch):
        if sketch.ndim == 2:
            sketch = np.stack([sketch] * 3, axis=-1)

        # Convert rgb image to (h, w, 1)
        sketch = cv2.copyMakeBorder(sketch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        b, g, r = cv2.split(sketch)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)
        processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
        uniques = np.unique(processed_image)

        result = None
        white_value = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [255]]

        for unique in uniques:
            # Get coords by color
            if unique not in white_value:
                continue

            cols, rows = np.where(processed_image == unique)
            image_tmp = np.zeros_like(processed_image)
            image_tmp[cols, rows] = 255
            # Get components
            labels = measure.label(image_tmp, connectivity=1, background=0)
            regions = measure.regionprops(labels, intensity_image=processed_image)
            result = regions[0]

        h, w = sketch.shape[:2]
        bg_image = np.ones((h, w), dtype=np.uint8)
        coords = result["coords"]
        bg_image[coords[:, 0], coords[:, 1]] = 0
        bg_image = bg_image[1:-1, 1:-1]
        return bg_image

    @staticmethod
    def get_foreground_regions(image, min_fore_area=30):
        # Get reference foreground
        labels = measure.label(image, connectivity=1, background=0)
        regions = measure.regionprops(labels)

        foreground = []
        for region in regions:
            if region.area < min_fore_area:
                continue
            mask = np.zeros_like(image)
            mask[region.coords[:, 0], region.coords[:, 1]] = 1
            component = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": mask,
                "label": len(foreground) + 1,
                "coords": region.coords,
                "bbox": region.bbox,
            }
            foreground.append(component)
        return foreground

    @staticmethod
    def denoise_with_reference(image, color_reference):
        out_image = image.copy()
        # Get the background of sketch (component with largest area)
        bg_image1 = ComponentDenoise.extract_background(color_reference)
        bg_image2 = ComponentDenoise.extract_background(image)
        foreground_regions = ComponentDenoise.get_foreground_regions(bg_image1)

        # Remove all components whose boxes are not overlap on the bbox of character in reference
        # Get components foreground of the target
        labels = measure.label(bg_image2, connectivity=1, background=0)
        regions = measure.regionprops(labels)

        for region in regions:
            if not any([overlap_with_foreground(region, f) for f in foreground_regions]):
                coords = region["coords"]
                bg_image2[coords[:, 0], coords[:, 1]] = 0

        coords = np.where(bg_image2 == 0)
        out_image[coords[0], coords[1]] = 255
        return out_image


def read_image(path):
    if path.endswith("tga"):
        image = np.array(Image.open(path).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)
    return image


def close_sketch(sketch):
    from kan.denoise import pre_denoise
    from kan.closing import ClosingModel

    closing_model = ClosingModel()
    sketch = np.ascontiguousarray(sketch)
    color = 0

    sketch = pre_denoise(sketch, np.stack([sketch] * 3, axis=-1))[0]
    pair_points = closing_model.process(sketch)[0]
    for p1, p2 in pair_points:
        cv2.line(sketch, (p1[1], p1[0]), (p2[1], p2[0]), color=color, thickness=1)
    return sketch


def main():
    import os
    import glob
    from natsort import natsorted

    root_dir = "D:/Data/GeekToys/coloring_data/server_test"
    output_dir = "D:/Data/GeekToys/output/rules"

    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    print(len(character_dirs))

    for character_dir in character_dirs:
        if "hor02_182" not in character_dir:
            continue

        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        reference_index = 0
        reference_path = paths[reference_index]

        character_name = os.path.basename(character_dir)
        if not os.path.exists(os.path.join(output_dir, character_name)):
            os.makedirs(os.path.join(output_dir, character_name))

        for path in paths:
            print(path)
            image_name = os.path.splitext(os.path.basename(path))[0]
            sketch_path = os.path.join(os.path.dirname(path), "..", "sketch", "%s.tga" % image_name)

            sketch = read_image(sketch_path)
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            sketch = cv2.threshold(sketch, 240, 255, cv2.THRESH_BINARY)[1]
            sketch = close_sketch(sketch)

            reference = read_image(reference_path)
            output = ComponentDenoise.denoise_with_reference(sketch, reference)
            cv2.imwrite(os.path.join(output_dir, character_name, "%s.png" % image_name), output)
    return


if __name__ == "__main__":
    main()
