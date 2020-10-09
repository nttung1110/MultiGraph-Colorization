from PIL import Image
from glob import glob
import numpy as np
import cv2


def process_sum(sketch, thres, axis):
    """Sum the sketch array along the columns or rows. Keep only those rows whose sum is smaller than <thres>.

    Args:
        sketch (2D ndarray): grayscale sketch
        thres (int): threshold for the sum
        axis (int): 0 for rows or 1 for columns

    Returns:
        Sum array along the specified axis.

    """
    arr_sum = np.sum(sketch, axis=axis, keepdims=True)
    temp = np.zeros_like(arr_sum)
    temp[arr_sum < thres] = 1
    return temp


def denoise_borders(sketch, rgb_sketch, top_bot_thres=150000, left_right_thres=80000):
    sketch = 255 - sketch
    kernel = np.ones((5, 5), np.uint8)
    preprocess_sketch = cv2.dilate(sketch, kernel, iterations=1)

    # Remove top and bottom borders' noise (axis=1).
    tb_denoise = process_sum(preprocess_sketch, top_bot_thres, axis=1)
    denoise = sketch * tb_denoise

    # Remove left and right borders' noise (axis=0).
    lr_denoise = process_sum(denoise, left_right_thres, axis=0)
    denoise = denoise * lr_denoise
    denoise = denoise.astype(np.uint8)

    # Remove leftover noise.
    kernel_2 = np.ones((5, 5), np.uint8)
    bin_denoise = np.zeros_like(denoise)

    # Binarize
    bin_denoise[denoise > 127] = 1
    bin_denoise[denoise <= 127] = 0

    bin_denoise = cv2.erode(bin_denoise, kernel_2, iterations=1)

    denoise[bin_denoise > 0] = 0
    denoise = 255 - denoise

    rgb_denoise = rgb_sketch.copy()
    for i in range(3):
        rgb_denoise_layer = rgb_denoise[:, :, i]
        rgb_denoise_layer[denoise == 255] = 255
        
    return denoise, rgb_denoise


def denoise_dot(sketch, rgb_sketch, threshold=0.2, size_dot=2):
    img = sketch
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        x, y, w, h = cv2.boundingRect(cnt)

        is_noise = np.sum(img[y:y+h, x:x+w]) < int(255*threshold*h*w)

        if is_noise or (w < size_dot or h < size_dot or radius < size_dot):
            img[y:y+h, x:x+w] = 255

    rgb_denoise = rgb_sketch.copy()
    for i in range(3):
        rgb_denoise_layer = rgb_denoise[:, :, i]
        rgb_denoise_layer[img == 255] = 255
        
    return img, rgb_denoise


def denoise_black_region(sketch, rgb_sketch, kernel_size=101, threshold=50):
    img = sketch
    kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size*kernel_size)
    dst = cv2.filter2D(img, -1, kernel)
    filter_mask = dst < (kernel_size*threshold*255)/(kernel_size*kernel_size)
    img[filter_mask] = 255

    rgb_denoise = rgb_sketch.copy()
    for i in range(3):
        rgb_denoise_layer = rgb_denoise[:, :, i]
        rgb_denoise_layer[img == 255] = 255
        
    return img, rgb_denoise


def pre_denoise(sketch, rgb_sketch):
    img, rgb_img = denoise_borders(sketch, rgb_sketch)
    img, rgb_img = denoise_black_region(img, rgb_img)
    img, rgb_img = denoise_dot(img, rgb_img)
    return img, rgb_sketch


def main():
    import cv2
    import os

    root_dir = 'app/ai_core/data/cat/sketch'
    save_result = "."
    if not os.path.exists(save_result):
        os.makedirs(save_result)

    for i, img_path in enumerate(glob('%s/*' % root_dir)[0:20]):
        print(img_path)
        rgb_sketch = Image.open(img_path)
        sketch = rgb_sketch.convert('L')
        rgb_sketch = np.array(rgb_sketch)
        sketch = np.array(sketch)
        denoise, rgb_denoise = pre_denoise(sketch, rgb_sketch)
        out_path = '%s/%d.png' % (save_result, i)
        divide = np.zeros((sketch.shape[0], 100, 3))
        divide[:, :, :] = 127
        res = np.concatenate([rgb_sketch, divide, rgb_denoise], axis=1).astype(np.uint8)
        cv2.imwrite(out_path, res)
    return


if __name__ == '__main__':
    main()
