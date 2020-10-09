# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np
import random
import cv2
from PIL import Image
from skimage.measure import regionprops

import matplotlib.pyplot as plt
def imgshow(im):
    plt.imshow(im)
    plt.show()

def random_cord():
    # initialize ...
    c_src = [
        [random.uniform(0, 0.2), random.uniform(0, 0.2)],
        [random.uniform(0.8, 1), random.uniform(0, 0.2)],
        [random.uniform(0.8, 1), random.uniform(0.8, 1)],
        [random.uniform(0, 0.2), random.uniform(0.8, 1)],
    ]

    c_dst = [
        [random.uniform(0, 0.2), random.uniform(0, 0.2)],
        [random.uniform(0.8, 1), random.uniform(0, 0.2)],
        [random.uniform(0.8, 1), random.uniform(0.8, 1)],
        [random.uniform(0, 0.2), random.uniform(0.8, 1)],
    ]

    #
    _src = [random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)]

    k = random.uniform(1.1, 1.3)
    _dst = [_src[0] * k, _src[1] * k]

    c_src += [_src]
    c_dst += [_dst]

    #
    _src = [random.uniform(0.5, 0.8), random.uniform(0.5, 0.8)]

    k = random.uniform(0.8, 1.)
    _dst = [_src[0] * k, _src[1] * k]

    c_src += [_src]
    c_dst += [_dst]

    return np.array(c_src), np.array(c_dst)

class TPS:
    @staticmethod
    def fit(c, lambd=0., reduced=False):
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32) * lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n + 3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n + 3, n + 3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v)  # p has structure w,a
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r ** 2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + b


def uniform_grid(shape):
    '''Uniform grid coordinates.

    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H, W = shape[:2]
    c = np.empty((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c


def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst

    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))

    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)

def batch_tps_grid(batch_theta, batch_c_dst, dshape):
    B = batch_theta.shape[0]

    batch_grid = []
    for b in range(B):
        grid = tps_grid(batch_theta[b], batch_c_dst[b], dshape)
        batch_grid += [grid]

    return np.stack(batch_grid, axis=0)


def tps_grid(theta, c_dst, dshape):
    ugrid = uniform_grid(dshape)

    reduced = c_dst.shape[0] + 2 == theta.shape[0]

    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid

    return grid


def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.

    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my

class TPSWrapper:
    def __init__(self):
        self.matrix_dct = {}

    def map_to_image(self, mapx, mapy, image):
        if len(image.shape) == 3:
            border_value = (255,255,255)
            return cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)

        elif len(image.shape) == 2:
            return cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        else:
            raise Exception ('unknown n_channel, expected 1 & 3, got %d ...' % len(image.shape))

    def gen(self, dshape, key_name):
        c_src, c_dst = random_cord()

        theta = tps_theta_from_points(c_src, c_dst, reduced=True)
        grid = tps_grid(theta, c_dst, dshape)
        mapx, mapy = tps_grid_to_remap(grid, dshape)

        self.matrix_dct[key_name] = (mapx, mapy)

    def augment(self, image, key_name):
        mapx, mapy = self.matrix_dct[key_name]

        return self.map_to_image(mapx, mapy, image)

    # def augment(self, image, dshape=None):
    #     dshape = dshape or tuple(image.shape[:2])
    #     c_src, c_dst = random_cord()
    #     mapx, mapy = self.warp_image_cv(image, c_src, c_dst, dshape=dshape)
    #
    #     return self.map_to_image(mapx, mapy, image), mapx, mapy

    def warp_image_cv(self, img, c_src, c_dst, dshape=None):
        dshape = dshape or img.shape
        theta = tps_theta_from_points(c_src, c_dst, reduced=True)

        grid = tps_grid(theta, c_dst, dshape)

        mapx, mapy = tps_grid_to_remap(grid, img.shape)
        return mapx, mapy

def extract_regions(mask):
    lbl2comps = {}

    for region in regionprops(mask):
        lbl2comps[region['label']] = region

    return lbl2comps

if __name__ == '__main__':
    image_path = "/home/kan/Desktop/Cinnamon/tyler/hades_painting_version_github/dummy_data/hor01_032_k_A/color/A0001.tga"
    np_image = np.asarray(Image.open(image_path).convert('RGB'))
    resized_image = cv2.resize(np_image, dsize=(512, 768), interpolation=cv2.INTER_NEAREST)

    output_image, mapx, mapy = TPSWrapper().augment(resized_image)
    debug_image = np.concatenate([resized_image, output_image], axis=1)
    imgshow(debug_image)

    from rules.component_wrapper import resize_mask, ComponentWrapper
    component_wrapper = ComponentWrapper(min_area=10, min_size=1)

    mask, components = component_wrapper.extract_on_color_image(np_image)
    mask = resize_mask(mask, components, (512, 768)).astype(np.int32)
    cors_mask = TPSWrapper().map_to_image(mapx, mapy, mask)

    debug_mask = np.concatenate([mask, cors_mask], axis=1)
    imgshow(debug_mask)

    comps = extract_regions(mask)
    cors_comps = extract_regions(cors_mask)

    for label_id, comp in comps.items():
        if comp['area'] < 50: continue

        cors_comp = cors_comps[label_id]

        image = cv2.resize(comp['image'].astype(np.uint8) * 255, (56,56))
        cors_image = cv2.resize(cors_comp['image'].astype(np.uint8) * 255, (56,56))

        imgshow(image)
        imgshow(cors_image)

        print ('image shape:', image.shape)
        print ('cors_image shape:', cors_image.shape)


