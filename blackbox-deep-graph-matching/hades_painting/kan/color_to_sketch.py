import os
import glob
import numpy as np
import cv2
import matplotlib.pylab as plt
import skimage
import skimage.morphology
import skimage.measure
import skimage.segmentation
from PIL import Image


def imread(filename, flags=cv2.IMREAD_COLOR, data_type=np.uint8):
    try:
        n = np.fromfile(filename, data_type)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
    return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
    return False


def main():
    root_dir = "D:/Data/GeekToys/coloring_data/server_test"
    count_cut = 0
    count_img = 0

    for (dir_path, dir_names, file_names) in os.walk(root_dir):
        dir_name = dir_path.replace("\\", "/").split("/")
        if len(dir_name) != 6 or "hor01" not in dir_name[5]:
            continue
        print("Cut " + str(count_cut) + ": " + dir_path)

        if not os.path.exists(os.path.join(root_dir, dir_name[5], "sketch_v3")):
            os.makedirs(os.path.join(root_dir, dir_name[5], "sketch_v3"))

        files = glob.glob(os.path.join(dir_path, "color", "*.tga"))
        print(dir_path)
        for file in files:
            print(file)
            count_img += 1
            name = os.path.basename(file)
            short_name = os.path.splitext(name)[0]

            sketch = Image.open(file)
            image = np.asarray(sketch)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_lb = skimage.measure.label(image)
            result = skimage.segmentation.find_boundaries(image_lb, connectivity=8)
            kernel = np.ones((1, 2), np.uint8)
            result = skimage.morphology.binary_dilation(result, kernel)
            result = (1 - result) * 255
            plt.imsave(
                os.path.join(root_dir, dir_name[5], "sketch_v3", "%s.png" % short_name), result,
                cmap=plt.cm.gray)
        count_cut += 1

    print("TOTAL IMAGE: " + str(count_img))
    print("TOTAL CUT: " + str(count_cut))


if __name__ == "__main__":
    main()
