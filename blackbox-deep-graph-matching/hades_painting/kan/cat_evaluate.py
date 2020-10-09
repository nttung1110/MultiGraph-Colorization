import os
import glob
import numpy as np
import cv2
import skimage.measure
from natsort import natsorted
from PIL import Image
from rules.component_wrapper import ComponentWrapper


def read_image(path):
    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)
    return image


def evaluate_per_pixel(result, label):
    """Calculate the percentage of pixels that are correctly colorized.
    Args:
        result (numpy array): colored sketch (RGB)
        label (numpy array): label (RGB)
    Returns:
        float: accuracy of the colorization process
    """
    height, width, depth = result.shape
    diff = result - label
    reduced_diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]
    n_accurate_pixels = height * width - np.count_nonzero(reduced_diff)
    total_pixels = height * width

    accuracy = n_accurate_pixels * 1.0 / total_pixels
    return accuracy


def calculate_iou(comp1, comp2, img_shape, mode=1):
    """
    :param comp1: first component, type dict.
    :param comp2: second component, type dict.
    :param img_shape: (height,width) of the original image.
    :param mode: If
        1: value = iou / np.sqrt(len(1) * len(2))
        2: value = iou / len(2)
    :return:
    """
    h, w = img_shape
    coords1 = comp1["coords"]
    coords2 = comp2["coords"]

    coords1_flatten = coords1[:, 0] * w + coords1[:, 1]
    coords2_flatten = coords2[:, 0] * w + coords2[:, 1]
    intersect_points = np.intersect1d(coords1_flatten, coords2_flatten)
    intersect_points = np.array(intersect_points)

    if mode == 1:
        iou_ratio = intersect_points.shape[0] / np.sqrt(coords2.shape[0] * coords1.shape[0])
    elif mode == 2:
        iou_ratio = intersect_points.shape[0] / coords2_flatten.shape[0]
    else:
        iou_ratio = intersect_points.shape[0]
    return iou_ratio


def evaluate_per_component(result, label, percent_thres=0.9, area_thres=15):
    """Calculate the percentage of components that are correctly colorized.
    Args:
        result (numpy array): colored sketch (RGB)
        label (numpy array): label (RGB)
        percent_thres (float): if the percentage of correctly colored pixels of each components is larger than this
                               threshold then that component is classified as correctly colorized.
        area_thres (int): only consider the components whose numbers of pixels are larger than this threshold.
    Returns:
        int: the number of components in the image
        int: the number of correctly colorized components
        float: accuracy of the colorization process
    """
    components_properties = ComponentWrapper(min_area=15, min_size=3).process(
        label, None, ComponentWrapper.EXTRACT_COLOR)
    components = components_properties[1]
    total_components = 0
    total_correct_components = 0

    for label_component in components:
        # Get the set of coordinates of all pixels in a component.
        component_pixels = label_component["coords"]
        rs = component_pixels[:, 0]  # list of all row coordinates
        cs = component_pixels[:, 1]  # list of all column coordinates
        # If the area of a component is less than the threshold, skip this component.
        if len(rs) < area_thres:
            continue
        total_components += 1

        # Get the mask for the component being considered.
        label_copy = np.zeros_like(label)
        label_copy[rs, cs, :] = label[rs, cs, :].copy()  # the mask

        # Get all the color values from the mask.
        color = tuple(label_copy[rs[0], cs[0], :])
        if color == (0, 0, 0):
            continue

        result_copy = result.copy()
        result_copy[result != color] = 0
        result_copy[result == color] = 1
        mask = result_copy[:, :, 0] + result_copy[:, :, 1] + result_copy[:, :, 2]
        mask[mask != 3] = 0

        mask_labels = skimage.measure.label(mask, connectivity=1)
        result_area = -1
        max_iou = 0
        for region in skimage.measure.regionprops(mask_labels):
            result_component = {"coords": region.coords}
            iou = calculate_iou(label_component, result_component, mask.shape, mode=0)
            if iou > max_iou:
                max_iou = iou
                result_area = region.area

        # If there's no region that overlaps with the considered component, skip it.
        if result_area == -1:
            continue
        ratio = label_component["area"] * 1.0 / result_area
        if ratio > 1:
            ratio = 1 / ratio

        if ratio > percent_thres:
            total_correct_components += 1

    accuracy = total_correct_components * 1.0 / total_components
    return accuracy


def write_the_title_row(save_path):
    from openpyxl import Workbook

    titles = ["Movie Folder Name", "Cut Name", "Reference Name", "Sketch Name", "Result File Name", "Time Taken",
              "Per Component", "Per Pixel", "User Experience Evaluation"]
    wb = Workbook()
    ws = wb.active
    ws.append(titles)
    wb.save(save_path)


def write_to_excel_sheet(result, save_path):
    from xlsxwriter import Workbook

    wb = Workbook(save_path)
    sheet = wb.add_worksheet("Report")
    titles = ["Movie Folder Name", "Cut Name", "Reference Name", "Sketch Name", "Result File Name", "Time Taken",
              "Per Component", "Per Pixel", "User Experience Evaluation"]
    for j, title in enumerate(titles):
        sheet.write(0, j, title)

    i = 0
    for img_name, report in result.items():
        i += 1
        sketch_name_list, ref_sketch_path, accuracy_by_pixel, accuracy_by_component, time = report
        sheet.write(i, 0, sketch_name_list[0])
        sheet.write(i, 1, sketch_name_list[1])
        sheet.write(i, 2, os.path.basename(ref_sketch_path))
        sheet.write(i, 3, img_name.split("_")[-1])
        sheet.write(i, 4, img_name.split("_")[-1].replace(".tga", ".png"))
        sheet.write(i, 5, time)
        sheet.write(i, 6, round(accuracy_by_component, 5))
        sheet.write(i, 7, round(accuracy_by_pixel, 5))
    wb.close()


def main():
    root_dir = "D:/Data/GeekToys/coloring_data/dataset_report/HOR02_report"
    output_dir = "D:/Data/GeekToys/output/report_data/hyper/"

    report_path = "D:/Data/GeekToys/output/report_data/hyper/report.xlsx"
    write_the_title_row(report_path)
    accuracy_report = dict()

    cut_dirs = natsorted(glob.glob(os.path.join(output_dir, "*")))
    for cut_dir in cut_dirs:
        print(cut_dir)
        cut_name = os.path.basename(cut_dir)
        paths = natsorted(glob.glob(os.path.join(cut_dir, "*.png")))
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

        color_paths = natsorted(glob.glob(os.path.join(root_dir, cut_name, "color", "*.tga")))
        color_names = [os.path.splitext(os.path.basename(p))[0] for p in color_paths]
        ref_name = [n for n in color_names if n not in names][0] + ".tga"

        for path in paths:
            image_name = os.path.splitext(os.path.basename(path))[0]
            if image_name == "reference":
                continue

            image_name = "%s.tga" % image_name
            full_name = "%s_%s" % (cut_name, image_name)
            name_list = ["HOR02_report", cut_name]
            color_path = os.path.join(root_dir, cut_name, "color", image_name)

            output_image = read_image(path)
            true_image = read_image(color_path)
            assert output_image.shape[0] == true_image.shape[0]

            acc_pixel = evaluate_per_pixel(output_image, true_image)
            acc_component = evaluate_per_component(output_image, true_image)
            accuracy_report[full_name] = (name_list, ref_name, acc_pixel, acc_component, 1.0)
            write_to_excel_sheet(accuracy_report, report_path)
            print(path, acc_component, acc_pixel)
    return


if __name__ == "__main__":
    main()
