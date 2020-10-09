import os
import glob
import numpy as np
import cv2
from natsort import natsorted
from math import copysign, log10
from PIL import Image
from rules.component_wrapper import ComponentWrapper


def get_data():
    root_dir = "D:/Data/GeekToys/coloring_data/complete_data"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "sketch_v3", "*.png")))
    print(len(paths))

    size = (768, 512)
    component_wrapper = ComponentWrapper()
    data = []

    for sketch_path in paths:
        color_path = sketch_path.replace("sketch_v3", "color").replace("png", "tga")

        sketch_image = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.cvtColor(np.array(Image.open(color_path).convert("RGB")), cv2.COLOR_RGB2BGR)

        color_image = np.where(
            np.stack([sketch_image == 0] * 3, axis=-1), np.zeros_like(color_image),
            color_image)
        color_image = cv2.resize(color_image, size, interpolation=cv2.INTER_NEAREST)

        mask, components = component_wrapper.process(color_image)
        for component in components:
            image = component["image"]
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            moments = cv2.moments(image)
            moments = cv2.HuMoments(moments)[:, 0]
            for i in range(0, 7):
                if moments[i] == 0:
                    continue
                moments[i] = -1 * copysign(1.0, moments[i]) * log10(abs(moments[i]))

            moments = moments.tolist()
            moments.append(component["area"])
            data.append(moments)

    print(len(data))
    data = np.array(data)
    np.save("D:/Data/GeekToys/coloring_data/training_data.npy", data)


def compute_statistics():
    path = "D:/Data/GeekToys/coloring_data/training_data.npy"
    data = np.load(path)
    print(data.shape)

    area = 50000.0
    mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
    std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
    print(len(mean), len(std))

    for index in range(0, 8):
        row = data[:, index]
        print(row.min(), row.max(), row.mean(), row.std())
        print((row.min() - mean[index]) / std[index])
        print((row.max() - mean[index]) / std[index])
        print()
    return


if __name__ == "__main__":
    compute_statistics()
