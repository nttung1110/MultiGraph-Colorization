import numpy as np
import cv2
from skimage.draw import line
from skimage.morphology import thin


def may_close(gray_image, max_traveled_pixel=10, max_pair_distance=50, keypoint_to_boundary_distance=15):
    """
    Step2: thinning, make the boundary width to 1-pixel
    """
    thinned_im = do_thin(gray_image)
    pair_points = []

    """
    Step3: the key-points is defined as the pixel which has the most number of neighbor background-pixels.
    """
    neighbor_matrix = to_neighbor_matrix(thinned_im)
    rows, cols = np.where(neighbor_matrix == 1)

    """
    Step4: Post-process the key-point, with our heuristic is that the key-points can not be the intersect btw two lines
    """
    chosen_rows, chosen_cols = [], []
    for row, col in zip(rows, cols):
        if is_intersection_point(row, col, thinned_im, max_traveled_pixel):
            continue

        chosen_rows += [row]
        chosen_cols += [col]

    """
    Step5: Construct the pair based on the distance btw key-points.
    >> Heuristic, the pair must satisfy following conditions:
        1. Not existing path btw 2 key-points (checking happen in the small window size, not full image)
        2. The line connecting two key-points is not allowed to intersect with other boundary line.
        3. 
    """
    pair = choose_pair_by_distance(chosen_rows, chosen_cols, max_distance=max_pair_distance)
    traveled = []

    for src_id, tgt_id in pair.items():
        src_r, src_c = chosen_rows[src_id], chosen_cols[src_id]
        tgt_r, tgt_c = chosen_rows[tgt_id], chosen_cols[tgt_id]

        p1 = (src_r, src_c)
        p2 = (tgt_r, tgt_c)

        if do_exist_path_between_points(point1=p1, point2=p2, image=thinned_im.copy(), padding=2):
            continue

        if not can_connect_two_boundary_points(point1=p1, point2=p2, image=thinned_im.copy()):
            continue

        traveled += [p1, p2]
        pair_points += [(p1, p2)]

    """
    Step6: Connect the remaining key-points to the nearest boundary (if satisfy the given max distance)
    """
    unset_points = []
    for row, col in zip(chosen_rows, chosen_cols):
        if (row, col) in traveled: continue

        dest_row, dest_col = connect_keypoint_to_boundary(point1=(row, col), image=thinned_im.copy(),
                                                          max_distance=keypoint_to_boundary_distance)
        if dest_row != -1:
            pair_points += [((row, col), (dest_row, dest_col))]
        else:
            unset_points += [(row, col)]

    return pair_points, unset_points, [(row, col) for (row, col) in zip(chosen_rows, chosen_cols)]


def do_thin(image, do_preprocess=False):
    image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)[1]

    if do_preprocess:
        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)[1]

    image[image == 255] = 1
    image = 1 - image

    thinned_im = thin(image).astype(np.uint8) * 255
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
    for r in range(min_row, max_row):
        for c in range(min_col, max_col):
            if (r, c) != (row, col) and matrix[r, c] > 0:
                coords += [(r, c)]

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

    return result


def can_connect_two_boundary_points(point1, point2, image):
    """
    Check if can connect 2 boundary pixels, (without facing any issues of intersect with other
    boundary line).

    :param point1: (r1,c1)
    :param point2: (r1,c1)
    :param image:
    :return:
    """

    # get the points from the line from p1 to p2
    line_rs, line_cs = line(*point1, *point2)
    line_rs = line_rs[1:-1]
    line_cs = line_cs[1:-1]

    # dilate with kernel(1,2)
    dilated_image = cv2.dilate(image, kernel=np.ones(shape=(1, 2), dtype=np.uint8), iterations=1)

    # check if the pixels in line is the boundary pixel or not
    for row, col in zip(line_rs, line_cs):

        # don't get the neighbors of the point1.
        if 0 <= abs(row - point1[0]) <= 1 and 0 <= abs(col - point1[1]) <= 1:
            continue

        if dilated_image[row, col] > 0:
            return do_exist_path_between_points(point1=(row, col), point2=point2, image=dilated_image.copy())

    return True


def do_exist_path_between_points(point1, point2, image, padding=2):
    """
    Check if exist path from point1 to point2.
    :param point1: (r1,c1)
    :param point2: (r2,c2)
    :param image:
    :param padding:
    :return:
    """
    h, w = image.shape

    # pre-processing
    point1 = np.array(point1)
    point2 = np.array(point2)

    min_h, min_w = min([point1[0], point2[0]]), min([point1[1], point2[1]])
    max_h, max_w = max([point1[0], point2[0]]), max([point1[1], point2[1]])

    # add padding
    min_h, min_w = max(min_h - padding, 0), max(min_w - padding, 0)
    max_h, max_w = min(max_h + padding + 1, h), min(max_w + padding + 1, w)

    # apply processing
    point1 -= [min_h, min_w]
    point2 -= [min_h, min_w]
    image = image[min_h:max_h, min_w:max_w]

    # running algorithm
    stack = [point1]
    traveled = []
    while len(stack) > 0:
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, image)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

        for (r, c) in neighbor_coords:
            if (r, c) == tuple(point2):
                return True
            else:
                stack += [(r, c)]

    return False


def calc_dist_btw_points(point1, point2):
    """
    Euclid distance btw two points.
    :param point1: (r1,c1)
    :param point2: (r2,c2)
    :return:
    """
    return np.linalg.norm(point2 - point1, ord=2)


def get_neighbor_ids(_row, _col, _cv2_im, only_active_pixel=True):
    """

    :param _row: ez
    :param _col: ez
    :param _cv2_im: ez
    :param only_active_pixel: if True, get only active pixel (pixel > 0) else get full
    :return:
    """
    h, w = _cv2_im.shape

    min_row = max(_row - 1, 0)
    max_row = min(_row + 1 + 1, h)

    min_col = max(_col - 1, 0)
    max_col = min(_col + 1 + 1, w)

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


def choose_pair_by_distance(rows, cols, max_distance, return_matrix=False):
    """
    Choose the pair for each key_point (r,c) by compare the distance between them.
    :param rows: list of row
    :param cols: list of column
    :param max_distance: max distance to be considered as pair or not
    :param return_matrix:
    :return:
    """
    coords = np.array([[row, col] for (row, col) in zip(rows, cols)], dtype=np.int32)  # (n_samples,2)

    # sorted by norm2
    distance = np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)  # (n_samples, n_samples, 2)
    distance = np.linalg.norm(distance, ord=2, axis=-1).T  # (n_samples, n_samples)
    distance[np.arange(len(rows)), np.arange(len(rows))] = np.inf

    # get the min distance
    min_ids = np.argmin(distance, axis=-1) # n_samples,

    pair = {k: v for k, v in enumerate(min_ids) if distance[k, v] <= max_distance}
    if not return_matrix:
        return pair
    return pair, distance


def connect_keypoint_to_boundary(point1, image, max_distance=6):
    """
    connect the keypoint to the nearest boundary (but still in a specified distance)
    :param point1: (r1,c1)
    :param image:
    :param max_distance:
    :return:
    """
    h, w = image.shape

    min_h, min_w = max(point1[0] - max_distance, 0), max(point1[1] - max_distance, 0)
    max_h, max_w = min(point1[0] + max_distance + 1, h), min(point1[1] + max_distance + 1, w)

    new_point1 = (point1[0] - min_h, point1[1] - min_w)

    kernel_mat = image[min_h:max_h, min_w:max_w]

    bound_rs, bound_cs = np.where(kernel_mat > 0)

    min_dist = np.inf
    final_r, final_c = -1, -1
    for row, col in zip(bound_rs, bound_cs):
        if (row, col) == tuple(new_point1):
            continue

        if do_exist_path_between_points(point1=(row, col), point2=new_point1, image=kernel_mat.copy(), padding=4):
            continue

        _dist = np.sqrt(row ** 2 + col ** 2)
        if _dist < min_dist:
            final_r, final_c = point1[0] + row - new_point1[0], point1[1] + col - new_point1[1]
            min_dist = _dist

    return final_r, final_c


def is_intersection_point(row, col, cv2_im, max_travel_pixel=10):
    """
    One assumption is that the un-clo10ed boundary is just a mistake from the client.
    So that it can not be the intersection point of two lines.

    > Check if as the pixel (row, col) is the intersection point of two lines.
    :param row:
    :param col:
    :param cv2_im:1
    :param max_travel_pixel:
    :return:
    """

    stack = [(row, col)]
    traveled = []
    tmp_max_traveled_pixel = max_travel_pixel
    while len(stack) > 0 and tmp_max_traveled_pixel > 0:
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, cv2_im)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]
        n_neighbor = len(neighbor_coords)

        if n_neighbor < 1:
            pass
        elif n_neighbor == 1:
            stack += neighbor_coords
            tmp_max_traveled_pixel -= 1
        else:
            return True

    return False
