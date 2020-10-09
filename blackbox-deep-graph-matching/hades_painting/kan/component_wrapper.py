import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure

from kan.close_object_v2 import may_close, do_exist_path_btw_points
from kan.closing import ClosingModel


def imgshow(img):
    plt.imshow(img)
    plt.show()


class ComponentWrapper:
    def __init__(self, binary_threshold=1, kernel_size=3):
        self.binary_threshold = binary_threshold
        self.background_value = 255
        self.minimum_n_pixels = 1

        self.kernel_size = kernel_size
        self.params_close_boundary_dct = {
            'small': {'max_traveled_pixel': 1, 'max_pair_distance': 13, 'keypoint_to_boundary_distance': 7},
            'medium': {'max_traveled_pixel': 10, 'max_pair_distance': 50, 'keypoint_to_boundary_distance': 15}
        }

    @staticmethod
    def to_image(comp1, comp2):
        list_of_coords = [comp1['coords'], comp2['coords']]

        combined_coords = np.concatenate(list_of_coords, axis=0)
        min_y, max_y = np.min(combined_coords[:, 0]), np.max(combined_coords[:, 0])
        min_x, max_x = np.min(combined_coords[:, 1]), np.max(combined_coords[:, 1])

        coords1 = comp1['coords'] - np.array([min_y, min_x]).reshape((1, 2))
        coords2 = comp2['coords'] - np.array([min_y, min_x]).reshape((1, 2))

        new_h, new_w = max_y - min_y + 1, max_x - min_x + 1

        # image 1
        tmp1 = np.zeros(shape=(new_h, new_w), dtype=np.uint8)
        tmp1[coords1[:, 0], coords1[:, 1]] = 255

        # image 2
        tmp2 = np.zeros(shape=(new_h, new_w), dtype=np.uint8)
        tmp2[coords2[:, 0], coords2[:, 1]] = 255

        return tmp1, tmp2

    @staticmethod
    def may_split_component(comp, h, w, unset_points=[], full_points=[]):
        # parameter for closing object
        _params = {'max_traveled_pixel': 10, 'max_pair_distance': 50, 'keypoint_to_boundary_distance': 15}

        im = np.zeros(shape=(h, w), dtype=np.uint8)
        im[comp['coords'][:, 0], comp['coords'][:, 1]] = 255

        """
        Debug ...
        """
        tmp_im = im.copy()
        chosen_rows = []
        chosen_cols = []

        for (r, c) in unset_points:
            neighbor_mat = tmp_im[max(r - 2, 0):min(r + 2, h), max(c - 2, 0):min(c + 2, w)].reshape((-1,))
            n_positive_pixel = np.count_nonzero(neighbor_mat)

            # must in the edge
            if n_positive_pixel in [0, len(neighbor_mat)]: continue

            chosen_rows += [r]
            chosen_cols += [c]

        # create a new pair
        coords1 = np.array([[r, c] for (r, c) in zip(chosen_rows, chosen_cols)])
        coords2 = np.array([[r, c] for (r, c) in full_points])

        if len(coords1) == 0: return None, []

        distance = np.expand_dims(coords1, axis=1) - np.expand_dims(coords2, axis=0)
        distance = np.linalg.norm(distance, ord=2, axis=-1)

        final_p1p2 = []

        for _id1 in range(len(coords1)):
            distance1 = distance[_id1]  # nsamples
            coord1 = coords1[_id1]

            _argmin_ids = np.argsort(distance1)
            for _argmin_id in _argmin_ids:
                coord2 = coords2[_argmin_id]

                if distance[_id1, _argmin_id] == 0: continue
                if distance[_id1, _argmin_id] >= 50: break

                sub_im = 255 - im
                sub_im[coord1[0], coord1[1]] = 255
                sub_im[coord2[0], coord2[1]] = 255

                exist_path = do_exist_path_btw_points(point1=(coord1[0], coord1[1]), point2=(coord2[0], coord2[1]),
                                                      cv2_im=sub_im, padding=10)

                if not exist_path:
                    # check if this line can split to two new components
                    _tmp_im = im.copy()
                    cv2.line(_tmp_im, (coord1[1], coord1[0]), (coord2[1], coord2[0]), color=0, thickness=1)
                    _labels = measure.label(_tmp_im, neighbors=4, background=0)

                    _regions = [r for r in measure.regionprops(_labels) if r.area > 230]
                    if len(_regions) >= 2:
                        final_p1p2 += [((coord1[1], coord1[0]), (coord2[1], coord2[0]))]
                        cv2.line(tmp_im, (coord1[1], coord1[0]), (coord2[1], coord2[0]), color=40, thickness=3)

                    break

        for (src_c, src_r), (tgt_c, tgt_r) in final_p1p2:
            cv2.line(im, (src_c, src_r), (tgt_c, tgt_r), color=0, thickness=1)

        labels = measure.label(im, neighbors=4, background=0)
        index = 0
        components = {}

        for region in measure.regionprops(labels):
            centroid = region.centroid
            area = region.area

            components[index] = {
                'centroid': np.array(centroid),
                'area': area,
                'image': region.image.astype(np.uint8) * 255,
                'label': region.label,
                'coords': region.coords,
                'bbox': region['bbox']
            }
            index += 1

        return components, final_p1p2

    def process(self, cv2_image, is_gray=False, process=False):
        """
        :param cv2_image: np.ndarray, should be of size (h,w) if is_gray else (h,w,3)
        :param is_gray: True/False
        :param process: True/False
        :return:
        """
        unset_points = []

        pair_points = []
        full_points = []
        new_closing = True

        if is_gray:
            _params = self.params_close_boundary_dct['small']

            if new_closing:
                pair_points, _, _, _ = ClosingModel().process(cv2_image)
                cv2_image = cv2_image.copy()
                for (src_r, src_c), (tgt_r, tgt_c) in pair_points:
                    cv2.line(cv2_image, (src_c, src_r), (tgt_c, tgt_r), color=0, thickness=1)
            else:
                _params['grayscale_im'] = cv2_image
                pair_points, unset_points, full_points = may_close(**_params)
                for (src_r, src_c), (tgt_r, tgt_c) in pair_points:
                    cv2.line(cv2_image, (src_c, src_r), (tgt_c, tgt_r), color=0, thickness=1)

            # convert {black_line/white_background} to opposite
            cv2_image = 255 - cv2_image

            # to binary
            binarized_image = self.__binarize(cv2_image)

            # pre processing
            if process:
                kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
                processed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)
            else:
                processed_image = binarized_image

            mask, components = self.__get_component(processed_image, background_value=self.background_value)

        else:
            # convert each value of 3-dimensional vector to unique value.
            b, g, r = cv2.split(cv2_image)
            processed_image = b + 1000 * (g + 1) + 1000 * 1000 * (r + 1)

            mask, components = self.__get_component(processed_image, background_value=16016015)
            # update the background detection, which component is background
            # heuristic, the background is the ones which color is white and number of pixel > 10000

            for _id, component in components.items():
                sample_coord = component['coords'][0]  # (r,c)
                color = cv2_image[sample_coord[0], sample_coord[1]]

                if tuple(color) == tuple([255, 255, 255]) and component['area'] > 10000:
                    component['background_predict'] = True
                else:
                    component['background_predict'] = False

        return mask, components, unset_points, full_points, pair_points

    def __get_component(self, binarized_image, background_value):
        labels = measure.label(binarized_image, neighbors=4, background=background_value)

        mask = labels
        index = 0
        components = {}

        for region in measure.regionprops(labels):
            centroid = region.centroid
            area = region.area

            if area > self.minimum_n_pixels:
                components[index] = {
                    'centroid': np.array(centroid),
                    'area': area,
                    'image': region.image.astype(np.uint8) * 255,
                    'label': region.label,
                    'coords': region.coords,
                    'bbox': region['bbox']
                }
                index += 1

        return mask, components

    def __binarize(self, cv2_image):
        _, binarized_image = cv2.threshold(cv2_image, self.binary_threshold, 255, cv2.THRESH_BINARY)
        return binarized_image
