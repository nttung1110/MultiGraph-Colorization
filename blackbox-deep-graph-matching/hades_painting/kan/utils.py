import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.draw import line
from skimage.morphology import thin
from skimage import measure as measure
from kan.intersect import doIntersect, Point


def imgshow(im):
    plt.imshow(im)
    plt.show()


def get_confidence_score(point2component, pair_points, regions):
    dct = {}

    for p1, p2 in pair_points: # (r,c)
        comp_ids_1 = point2component[p1]
        comp_ids_2 = point2component[p2]

        common_ids = set(map(int, comp_ids_1)).intersection(set(map(int, comp_ids_2)))
        common_ids = list(common_ids)
        if common_ids:
            for common_id in common_ids:
                common_id = common_id - 1

                if p1 in [(1176, 1014)]:
                    print('found')

                region = regions[common_id]['image'].astype(np.uint8) * 255
                _y1, _x1, _, _ = regions[common_id]['bbox']
                cv2.line(region, (p1[1] - _x1, p1[0] - _y1), (p2[1] - _x1, p2[0] - _y1), color=0, thickness=1)

                max_v = np.max(measure.label(region, neighbors=4, background=0))
                if max_v > 1:
                    dct[(p1, p2)] = 1
                else:
                    dct[(p1, p2)] = 0 if (p1, p2) not in dct else dct[(p1, p2)]

    return dct


def filter_by_components(pair_points, thinned_image):
    results = []

    # for original image
    _labels  = measure.label(thinned_image, neighbors=4, background=0)
    _regions = measure.regionprops(_labels)

    #
    for (s_r, s_c), (t_r, t_c) in pair_points:

        if (s_r, s_c) in [(992, 1293)]:
            print('found')

        _tmp = thinned_image.copy()
        cv2.line(_tmp, (s_c, s_r), (t_c, t_r), color=0, thickness=1)

        _tmp_regions = measure.regionprops(measure.label(_tmp, neighbors=4, background=0))
        if len(_tmp_regions) > len(_regions):
            results += [((s_r, s_c), (t_r, t_c))]

    return results


def do_thin(cv_im, do_preprocess=False):
    """
    Thinning the given image to 1-pixel boundary
    :param cv_im: np.ndarray, should be binarized to better performance
    :return:
    """
    cv_im = cv2.threshold(cv_im, 254, 255, cv2.THRESH_BINARY)[1]

    if do_preprocess:
        cv_im = cv2.GaussianBlur(cv_im, (7, 7), 0)
        cv_im = cv2.threshold(cv_im, 254, 255, cv2.THRESH_BINARY)[1]

    cv_im[cv_im == 255] = 1
    cv_im = 1 - cv_im

    thinned_im = thin(cv_im).astype(np.uint8) * 255
    return thinned_im


def _count_active_neighbor(row, col, matrix):
    """
    Count the active (the boundary pixel, has value > 0) neighbor pixels of the given pixel.
    :param row: int
    :param col: int
    :param matrix: np.ndarray
    :return:
    """
    h, w = matrix.shape

    # get the neighbor ids
    min_row = max(row - 1, 0)
    max_row = min(row + 1 + 1, h)

    min_col = max(col - 1, 0)
    max_col = min(col + 1 + 1, w)

    coords = []
    for _row in range(min_row, max_row):
        for _col in range(min_col, max_col):
            if (_row, _col) != (row, col) and matrix[_row, _col] > 0:
                coords += [(_row, _col)]

    return coords, matrix[min_row:max_row, min_col:max_col]


def to_neighbor_matrix(cv2_im):
    """
    :param cv2_im: 255 is boundary, 0 is background. cv2_im is a binary matrix
    :return: a matrix whose element represent the number of active pixels (pixel > 0).
    """
    rows, cols = np.where(cv2_im != 0)
    result = np.zeros_like(cv2_im, dtype=np.uint8)
    for row, col in zip(rows, cols):
        neighbors, _matrix = _count_active_neighbor(row, col, matrix=cv2_im)
        result[row, col] = len(neighbors)

        if False:
            print("row:%d-col:%d has %d neighbors..." % (row, col, len(neighbors)))

    return result


def calc_gradient_2(M):
    kernel_size = 5

    # x
    kernel_x = np.array([
        [-1] * kernel_size + [0] + [1] * kernel_size,
    ] * (2 * kernel_size + 1))
    kernel_x[kernel_size] *= 2

    # y
    kernel_y = kernel_x.T

    direct_x = cv2.filter2D(M, -1, kernel_x)
    direct_y = cv2.filter2D(M, -1, kernel_y)

    theta = np.arctan2(direct_y, direct_x)
    return theta


def calc_gradient(M):
    # return calc_gradient_2(M)
    """
    Calculating the angle of point ...
    :param M:
    :return:
    """
    sobel_x = cv2.Sobel(M, cv2.CV_64F, 1, 0, ksize=11)
    sobel_y = cv2.Sobel(M, cv2.CV_64F, 0, 1, ksize=11)

    theta = np.arctan2(sobel_y, sobel_x)
    return theta


def find_direction_v2(img, rows, cols, D, max_traveled_pixel=5):
    def _find(row, col, img, max_traveled_pixel):
        stack = [(row, col)]
        traveled, directions = [], []
        tmp_max_traveled_pixel = max_traveled_pixel
        while (len(stack) > 0 and tmp_max_traveled_pixel > 0):
            cur_row, cur_col = stack.pop(0)
            traveled += [(cur_row, cur_col)]

            if (cur_row, cur_col) in [(784, 1127)]:
                print('found')

            neighbor_coords = get_neighbor_ids(cur_row, cur_col, img)
            neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

            n_neighbor = len(neighbor_coords)
            if n_neighbor == 1:
                stack += neighbor_coords
                tmp_max_traveled_pixel -= 1

                _direction = (cur_col - neighbor_coords[0][1], cur_row - neighbor_coords[0][0])
                if np.abs(_direction[0]) == 1 and np.abs(_direction[1]) == 1:
                    directions += [_direction] * 2
                else:
                    directions += [_direction]

        _vs, _fs = np.unique(directions, return_counts=True, axis=0)
        _max_f_id = np.argmax(_fs)

        if _fs[_max_f_id] == 1:
            return directions[0]
        else:
            return _vs[_max_f_id]

    result_dct = {}
    for i, j in zip(rows, cols):
        _next = _find(i, j, img, max_traveled_pixel)
        result_dct[(i, j)] = _next

    return result_dct


def find_direction(img, rows, cols, D, n_neighbor = 5):
    """
    Mapping angle to the next neighbor points ...
    """
    h, w = img.shape[:2]

    def angle2direction(a):
        #
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
        angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])

        #
        _tmp = np.abs(angles - a)
        _min_id = np.argmin(_tmp)

        #
        if _min_id == len(angles) - 1: _min_id = 2

        _next = directions[_min_id]

        return _next

    angle = D * 180. / np.pi
    angle[angle < 0] += 360

    #
    result_dct = {}
    for i, j in zip(rows, cols):
        if (i,j) in [(101,570)]:
            print ('found')

        nb_cords = get_neighbor_ids(i, j, img, only_active_pixel=True, max_neighbor=n_neighbor)

        _nexts = [angle2direction(angle[_i,_j]) for _i,_j in nb_cords if 0 <= _i < h and 0 <= _j < w ]
        _vs, _fs = np.unique(_nexts, return_counts=True, axis=0)
        _max_f = 1 #np.argmax(_fs)

        _next = angle2direction(angle[i,j]) if _max_f == 1 else _vs[np.argmax(_fs)]
        result_dct[(i,j)] = tuple(_next)

    return result_dct


def points_to_components(points, thinned_image):
    def _point_to_component(point, tgt_label, org_shape,window_size = 5):
        h, w = org_shape
        _row, _col = point

        min_y = max(_row - window_size, 0)
        max_y = min(_row + window_size, h)

        min_x = max(_col - window_size, 0)
        max_x = min(_col + window_size, w)

        _sub_im = tgt_label[min_y: max_y, min_x: max_x]
        _values, _counts = np.unique(_sub_im, return_counts=True)

        _most_vals = list(_values[np.argsort(_counts)][::-1])

        if 0 in _most_vals: _most_vals.remove(0)

        return _most_vals

    labels = measure.label(thinned_image, neighbors=4, background=0)
    point2comp = {}
    for point in points:
        comp_ids = _point_to_component(point, labels, thinned_image.shape[:2], window_size=5)
        point2comp[point] = list(map(str,comp_ids))

    return point2comp, measure.regionprops(labels)


def get_neighbor_ids(_row, _col, _cv2_im, only_active_pixel=True, max_neighbor=1):
    """

    :param _row: ez
    :param _col: ez
    :param _cv2_im: ez
    :param only_active_pixel: if True, get only active pixel (pixel > 0) else get full
    :return:
    """
    h, w = _cv2_im.shape

    min_row = max(_row - max_neighbor, 0)
    max_row = min(_row + max_neighbor + 1, h)

    min_col = max(_col - max_neighbor, 0)
    max_col = min(_col + max_neighbor + 1, w)

    coords = []
    for __row in range(min_row, max_row):
        for __col in range(min_col, max_col):

            if (__row, __col) != (_row, _col):
                if not only_active_pixel:
                    coords += [(__row, __col)]
                else:
                    if _cv2_im[__row, __col] > 0:
                        coords += [(__row, __col)]

    return coords


def is_intersection_point(row, col, cv2_im, max_traveled_pixel=10):
    """
    One assumption is that the un-clo10ed boundary is just a mistake from the client.
    So that it can not be the intersection point of two lines.

    > Check if as the pixel (row, col) is the intersection point of two lines.
    :param row:
    :param col:
    :param cv2_im:1
    :return:
    """

    stack = [(row, col)]
    traveled = []
    tmp_max_traveled_pixel = max_traveled_pixel
    while (len(stack) > 0 and tmp_max_traveled_pixel > 0):
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, cv2_im)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

        n_neighbor = len(neighbor_coords)
        if False:
            print("row: %d,col: %d has %d neighbor, %d" % (cur_row, cur_col, n_neighbor, tmp_max_traveled_pixel))

        if n_neighbor < 1:
            pass
        elif n_neighbor == 1:
            stack += neighbor_coords
            tmp_max_traveled_pixel -= 1
        else:
            return True

    return False


def choose_pair_by_distance(rows, cols, max_distance, return_matrix = False):
    """
    Choose the pair for each key_point (r,c) by compare the distance between them.
    :param rows: list of row
    :param cols: list of column
    :param max_distance: max distance to be considered as pair or not
    :return:
    """
    coords = np.array([[row, col] for (row, col) in zip(rows, cols)], dtype=np.int32)  # (n_samples,2)

    # sorted by norm2
    distance = np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)  # (n_samples, n_samples, 2)
    distance = np.linalg.norm(distance, ord=2, axis=-1).T  # (n_samples, n_samples)
    distance[np.arange(len(rows)), np.arange(len(rows))] = np.inf

    # get the min distance
    min_ids = np.argsort(distance, axis=-1)# np.argmin(distance, axis=-1) # n_samples,

    # build pair
    pair = []
    topk = 2
    for k, vs in enumerate(min_ids):
        _tmp = [(k,v) for v in vs[:topk] if distance[k,v] <= max_distance]

        pair += _tmp

    return pair

    #
    # pair = {k:v for k,v in enumerate(min_ids) if distance[k,v] <= max_distance}
    # if return_matrix == False:
    #     return pair
    # else:
    #     return pair, distance


def normalize_sub_im(point1, point2, cv2_im, padding, org_shape):
    h, w = org_shape

    point1 = np.array(point1) # r, c
    point2 = np.array(point2) # r, c

    # get min, max
    min_h, min_w = min([point1[0], point2[0]]), min([point1[1], point2[1]])
    max_h, max_w = max([point1[0], point2[0]]), max([point1[1], point2[1]])

    # add padding
    min_h, min_w = max(min_h - padding, 0), max(min_w - padding, 0)
    max_h, max_w = min(max_h + padding + 1, h), min(max_w + padding + 1, w)

    # apply normalize
    point1 -= [min_h, min_w]
    point2 -= [min_h, min_w]
    sub_cv2_im = cv2_im[min_h:max_h, min_w:max_w]

    return point1, point2, sub_cv2_im


def do_exist_path_btw_points(point1, point2, cv2_im, padding=2):
    """
    Check if exist path from point1 to point2.
    :param point1: (r1,c1)
    :param point2: (r2,c2)
    :param cv2_im:
    :return:
    """
    point1, point2, cv2_im = normalize_sub_im(point1, point2, cv2_im, padding, cv2_im.shape)

    # running algorithm
    stack = [point1]
    traveled = []
    while (len(stack) > 0):
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, cv2_im)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

        for (_row, _col) in neighbor_coords:
            if (_row, _col) == tuple(point2):
                return True
            else:
                stack += [(_row, _col)]

    return False


def connect_keypoint_to_boundary(point1, cv2_im, max_distance=6, direction_dct = None):
    h, w = cv2_im.shape

    n_iter = max_distance
    r, c = point1

    if (r,c) in [(901, 1413)]:
        print('found')

    direction = direction_dct[(r,c)][::-1]

    next_rs, next_cs = [r + _ * direction[0] for _ in range(1, n_iter + 1)], [c + _ * direction[1] for _ in range(1, n_iter + 1)]
    next_pxls = [cv2_im[_r, _c] if (0 <= _r < h and 0 <= _c < w) else 0 for _r, _c in zip(next_rs, next_cs)]

    if 255 not in next_pxls: return -1, -1
    else:
        _id = next_pxls.index(255)

        return next_rs[_id], next_cs[_id]


def can_connect_two_points(point1, point2, cv2_im):
    def is_in_neighbor(r, c, org_r, orc_c, nb_k):
        if 0 <= np.abs(r - org_r) <= nb_k and 0 <= np.abs(c - orc_c) <= nb_k:
            return True
        return False

    line_rs, line_cs = line(*point1, *point2)

    traveled = []
    for r, c in zip(line_rs, line_cs):
        nb_points = get_neighbor_ids(r, c, cv2_im, only_active_pixel=True)

        if is_in_neighbor(r, c, point1[0], point1[1], 1): continue
        if is_in_neighbor(r, c, point2[0], point2[1], 1): continue

        traveled += [(r,c)]
        nb_points = [point for point in nb_points if point not in traveled]
        if len(nb_points) > 1: return False

    return True


def match_direction_v2(point1, point2, direction_dct, org_shape):
    p1_r, p1_c = point1
    p2_r, p2_c = point2
    h, w = org_shape

    d1 = direction_dct[(p1_r, p1_c)][::-1]  # (y,x)
    d2 = direction_dct[(p2_r, p2_c)][::-1]  # (y,x)

    iter = 100
    e_p1_r = min(h - 1, p1_r + iter * d1[0])
    e_p1_c = min(w - 1, p1_c + iter * d1[1])

    e_p2_r = min(h - 1, p2_r + iter * d2[0])
    e_p2_c = min(w - 1, p2_c + iter * d2[1])

    n_max_neighbor = 5
    ofs_x = [(0, _x) for _x in range(-n_max_neighbor, n_max_neighbor + 1)]
    ofs_y = [(_y, 0) for _y in range(-n_max_neighbor, n_max_neighbor + 1)]
    ofs = ofs_x + ofs_y

    for ofs_r_1, ofs_c_1 in ofs:
        for ofs_r_2, ofs_c_2 in ofs:
            _p1_c = np.clip(p1_c + ofs_c_1, 0, w - 1)
            _p1_r = np.clip(p1_r + ofs_r_1, 0, h - 1)
            _e_p1_c = np.clip(e_p1_c + ofs_c_1, 0, w - 1)
            _e_p1_r = np.clip(e_p1_r + ofs_r_1, 0, h - 1)

            _p2_c = np.clip(p2_c + ofs_c_2, 0, w - 1)
            _p2_r = np.clip(p2_r + ofs_r_2, 0, h - 1)
            _e_p2_c = np.clip(e_p2_c + ofs_c_2, 0, w - 1)
            _e_p2_r = np.clip(e_p2_r + ofs_r_2, 0, h - 1)

            is_intersect = doIntersect(Point(_p1_c, _p1_r), Point(_e_p1_c, _e_p1_r),
                                       Point(_p2_c, _p2_r), Point(_e_p2_c, _e_p2_r))
            if is_intersect == True:
                return True

    return False


def match_direction(point1, point2, direction_dct, org_shape):
    p1_r, p1_c = point1
    p2_r, p2_c = point2
    h, w = org_shape

    d1 = direction_dct[(p1_r, p1_c)][::-1]  # (y,x)
    d2 = direction_dct[(p2_r, p2_c)][::-1]  # (y,x)

    iter = 100
    e_p1_r = min(h - 1, p1_r + iter * d1[0])
    e_p1_c = min(w - 1, p1_c + iter * d1[1])

    e_p2_r = min(h - 1, p2_r + iter * d2[0])
    e_p2_c = min(w - 1, p2_c + iter * d2[1])

    n_max_neighbor = 5
    ofs_x = [(0, _x) for _x in range(-n_max_neighbor, n_max_neighbor + 2)]
    ofs_y = [(_y, 0) for _y in range(-n_max_neighbor, n_max_neighbor + 2)]
    ofs = ofs_x + ofs_y

    for ofs_r_1, ofs_c_1 in ofs:
        for ofs_r_2, ofs_c_2 in ofs:
            is_intersect = doIntersect(Point(p1_c + ofs_c_1, p1_r + ofs_r_1), Point(e_p1_c + ofs_c_1, e_p1_r + ofs_r_1),
                                       Point(p2_c + ofs_c_2, p2_r + ofs_r_2), Point(e_p2_c + ofs_c_2, e_p2_r + ofs_r_2))
            if is_intersect == True:
                return True

    return False


def rescale_coordinates(point, up_bound):
    row, col = point
    row += up_bound
    return row, col


def convert_output_format(pair_points, up_bound=0):
    """Converts the format output pairs to that of evaluation.

    :param pair_points:
    :return:
    """
    result = dict()
    for pair in pair_points:
        start, end = pair
        start = rescale_coordinates(start, up_bound)
        end = rescale_coordinates(end, up_bound)

        key = '%d_%d' % (start[0], start[1])
        value = '%d_%d' % (end[0], end[1])
        result[key] = value
    return result
