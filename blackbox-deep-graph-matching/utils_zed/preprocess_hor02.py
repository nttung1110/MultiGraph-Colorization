from skimage import measure
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
'''
This script is used for converting color image to corresponding image without 
changing component information too much
'''


def get_sketch(image):
    '''
        Input params:
            image: color image to convert(should be in .png format)
        Returned params:
            sketch, color: sketch along with its corresponding color image after being converted
    '''
    # Convert rgb image to (h, w, 1)
    b, r, g = cv2.split(image)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    sketch = np.zeros_like(processed_image)
    color = image.copy()

    for unique in uniques:
        if unique == 1444815 or unique == 90300:
            rows, cols = np.where(processed_image == unique)
            sketch_1 = np.zeros_like(processed_image)
            sketch_1[rows, cols] = 255

        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region['area'] <= 10:
                continue

            image_tmp_ = np.zeros_like(processed_image)
            coord = region['coords']
            image_tmp_[coord[:, 0], coord[:, 1]] = 255

            contours = measure.find_contours(image_tmp_, 0.8)
            for _, contour in enumerate(contours):
                contour = np.array(contour, dtype=np.int)
                sketch[contour[:, 0], contour[:, 1]] = 255
                color[contour[:, 0], contour[:, 1], :] = 0
    try:
        sketch = np.array(sketch + sketch_1, dtype=np.uint8)
    except:
        sketch = np.array(sketch, dtype=np.uint8)
    # print(sketch.shape)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    sketch = 255 - cv2.threshold(sketch, 0, 255, cv2.THRESH_BINARY)[1]
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return sketch, color

def mkdirs(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)



def process_hor02(data_path_in, data_path_out):
    for folder_name in os.listdir(data_path_in):
        sub_folder_in_color = os.path.join(data_path_in, folder_name, "color")
        if os.path.isdir(sub_folder_in_color)==False :
            continue 

        print("Process ", folder_name)
        sub_folder_out = os.path.join(data_path_out, folder_name)

        mkdirs(sub_folder_out)

        sub_folder_out_color = os.path.join(sub_folder_out, "color")
        sub_folder_out_sketch = os.path.join(sub_folder_out, "sketch")

        mkdirs(sub_folder_out_color)
        mkdirs(sub_folder_out_sketch)

        for file_name in os.listdir(sub_folder_in_color):
            color_image = np.array(Image.open(os.path.join(sub_folder_in_color, file_name)), dtype=np.uint8)
            out_sketch, out_color = get_sketch(color_image)

            # saving 
            cv2.imwrite(os.path.join(sub_folder_out_color, file_name[:-4]+".png"), out_color)
            cv2.imwrite(os.path.join(sub_folder_out_sketch, file_name[:-4]+".png"), out_sketch)

        

if __name__ == "__main__":
    data_path_in = "/mnt/ai_filestore/home/zed/multi-graph-matching/Geek_data/HOR02_Full_Formated"
    data_path_out = "/mnt/ai_filestore/home/zed/multi-graph-matching/Geek_data/processed_HOR02"

    mkdirs(data_path_out)

    process_hor02(data_path_in, data_path_out)

# if __name__ == "__main__":
#     cuts = sorted(glob.glob(os.path.join('/home/dotieuthien/Downloads/dataset_report/HOR02_noise_report', '*')))
#     for cut in cuts:
#         if not os.path.exists('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut)):
#             os.makedirs('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut))
#             os.makedirs('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut) + '/color')
#             os.makedirs('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut) + '/sketch')
#         images = sorted(glob.glob(os.path.join(cut, 'color', '*')))
#         for path in images:
#             img_name = os.path.basename(path)[:-4]
#             img = np.array(Image.open(path), dtype=np.uint8)
#             sketch, color = get_sketch(img)
#             cv2.imwrite('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut) + '/sketch/' + img_name + '.png', sketch)
#             cv2.imwrite('/home/dotieuthien/Documents/Geektoys/convex/validate_dataset/valid/' + os.path.basename(cut) + '/color/' + img_name + '.png', color)
