import os
import glob
import shutil
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from loader.config_utils import unzip


def filter_data():
    input_dir = "D:/Data/GeekToys/coloring_data/geektoy_data"
    output_dir = "D:/Data/GeekToys/coloring_data/filter_data"

    wrong = [
        "hor01_018_021_k_A", "hor01_041_k_A", "hor01_043_k_A",
        "hor01_067_071_k_A", "hor01_080_088_090_k_A", "hor01_120_k_A",
        "hor01_123_140_k_A", "hor01_141_122_k_A",
        "hor01_034_k_A",
        "hor01_022_k_B", "hor01_036_k_B", "hor01_041_k_B", "hor01_042_k_B", "hor01_043_k_B",
        "hor01_044_046_k_B", "hor01_064_k_B", "hor01_047_k_B", "hor01_048_k_B", "hor01_056_k_B",
        "hor01_057_k_B", "hor01_059_k_B", "hor01_063_065_k_B", "hor01_066_k_B", "hor01_070_k_B",
        "hor01_074_k_B", "hor01_075_k_B", "hor01_076_078_k_B", "hor01_083_k_B", "hor01_113_103_k_B",
        "hor01_120_k_B", "hor01_123_140_k_B", "hor01_133_k_B", "hor01_134_k_B", "hor01_136_k_B",
        "hor01_141_122_k_B", "hor01_142_k_B", "hor01_143_k_B",
    ]
    right = [
        "hor01_112_118_135_169_k_C", "hor01_134_k_C",
        "hor01_055_k_R_B", "hor01_023_026_k_A",
        "hor01_101_104_106_108_k_B", "hor01_101_104_106_108_k_C", "hor01_101_104_106_108_k_D",
    ]

    characters = natsorted(glob.glob(os.path.join(input_dir, "*")))
    for character in characters:
        character_name = os.path.basename(character)

        get_character = False
        if character_name in right:
            get_character = True
        if character_name.endswith("k_A") and character_name not in wrong:
            get_character = True
        if character_name.endswith("k_B") and character_name not in wrong:
            get_character = True
        if all([c.isdigit() for c in character_name[-3:]]):
            get_character = True
        if not get_character:
            continue

        shutil.copytree(character, os.path.join(output_dir, character_name))

    return


def get_character_path():
    import numpy as np
    import cv2

    input_dir = "D:/Data/GeekToys/coloring_data/geektoy_data"
    path = "D:/Data/GeekToys/coloring_data/simple_data/E/sketch_v3/A0004.png"

    data_paths = natsorted(glob.glob(os.path.join(input_dir, "*", "sketch_v3", "A0004.png")))
    image = cv2.imread(path)

    for data_path in data_paths:
        data_image = cv2.imread(data_path)

        if np.all(image == data_image):
            print(data_path)
    return


def same_character(character_dir, ref_dir):
    if os.path.basename(character_dir) == os.path.basename(ref_dir):
        return True

    image_path = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))[0]
    image_name = os.path.basename(image_path)
    ref_path = os.path.join(ref_dir, "color", image_name)

    if not os.path.exists(ref_path):
        return False

    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    ref_image = cv2.cvtColor(np.array(Image.open(ref_path).convert("RGB")), cv2.COLOR_RGB2BGR)

    if (image.shape == ref_image.shape) and np.all(image == ref_image):
        return True
    return False


def get_different_data():
    input_dir = "D:/Data/GeekToys/coloring_data/geektoy_data"
    ref_dir = "D:/Data/GeekToys/coloring_data/complete_data"
    output_dir = "D:/Data/GeekToys/coloring_data/different_data"

    character_dirs = natsorted(glob.glob(os.path.join(input_dir, "*")))
    ref_dirs = natsorted(glob.glob(os.path.join(ref_dir, "*")))

    wrong = [
        "hor01_018_021_k_A", "hor01_041_k_A", "hor01_043_k_A",
        "hor01_067_071_k_A", "hor01_080_088_090_k_A", "hor01_120_k_A",
        "hor01_123_140_k_A", "hor01_141_122_k_A",
        "hor01_034_k_A",
        "hor01_022_k_B", "hor01_036_k_B", "hor01_041_k_B", "hor01_042_k_B", "hor01_043_k_B",
        "hor01_044_046_k_B", "hor01_064_k_B", "hor01_047_k_B", "hor01_048_k_B", "hor01_056_k_B",
        "hor01_057_k_B", "hor01_059_k_B", "hor01_063_065_k_B", "hor01_066_k_B", "hor01_070_k_B",
        "hor01_074_k_B", "hor01_075_k_B", "hor01_076_078_k_B", "hor01_083_k_B", "hor01_113_103_k_B",
        "hor01_120_k_B", "hor01_123_140_k_B", "hor01_133_k_B", "hor01_134_k_B", "hor01_136_k_B",
        "hor01_141_122_k_B", "hor01_142_k_B", "hor01_143_k_B",
    ]

    for character_dir in character_dirs:
        character_name = os.path.basename(character_dir)
        if character_name in wrong:
            continue
        if any([word in character_name for word in ["go", "moto"]]):
            continue
        print(character_dir)

        compare_ref = [same_character(character_dir, ref_dir) for ref_dir in ref_dirs]
        if not any(compare_ref):
            shutil.copytree(character_dir, os.path.join(output_dir, character_name))
    return


def get_server_hard_data():
    sketch_root_dir = "D:/Data/GeekToys/coloring_data/Tyler_HOR02_cat_collection"
    color_root_dir = "D:/Data/GeekToys/coloring_data/HOR02_deta_1108/paint"
    output_dir = "D:/Data/GeekToys/coloring_data/server_super_hard"

    sketch_dirs = natsorted(glob.glob(os.path.join(sketch_root_dir, "*", "*")))
    for sketch_dir in sketch_dirs:
        print(sketch_dir)
        # sketch
        name = os.path.basename(os.path.dirname(sketch_dir)).replace("t", "")
        part = os.path.basename(sketch_dir)
        full_name = "%s_%s" % (name, part)

        if not os.path.exists(os.path.join(output_dir, full_name)):
            os.makedirs(os.path.join(output_dir, full_name, "sketch"))
            os.makedirs(os.path.join(output_dir, full_name, "color"))

        sketch_paths = natsorted(glob.glob(os.path.join(sketch_dir, "*.tga")))
        for path in sketch_paths:
            shutil.copy(path, os.path.join(output_dir, full_name, "sketch", os.path.basename(path)))

        # color
        zip_path = os.path.join(color_root_dir, "%s.zip" % name)
        zip_dir = os.path.join(color_root_dir, name, part)

        if not os.path.exists(zip_dir):
            unzip(zip_path, color_root_dir)

        color_paths = natsorted(glob.glob(os.path.join(zip_dir, "*.tga")))
        for path in color_paths:
            shutil.copy(path, os.path.join(output_dir, full_name, "color", os.path.basename(path)))
    return


def get_good_sketch_data():
    data_dir = "D:/Data/GeekToys/coloring_data/full_data"
    root_dir = "D:/Data/GeekToys/coloring_data/good_hor02"
    output_dir = "D:/Data/GeekToys/output/good_hor02"

    cut_dirs = natsorted(glob.glob(os.path.join(output_dir, "*")))
    for cut_dir in cut_dirs:
        cut_name = os.path.basename(cut_dir)

        root_cut_dir = os.path.join(root_dir, cut_name)
        if not os.path.exists(root_cut_dir):
            os.makedirs(os.path.join(root_cut_dir, "color"))
            os.makedirs(os.path.join(root_cut_dir, "sketch"))

        paths = natsorted(glob.glob(os.path.join(cut_dir, "*.png")))
        for path in paths:
            image_name = os.path.splitext(os.path.basename(path))[0]
            image_name = "%s.tga" % image_name

            color_path = os.path.join(data_dir, cut_name, "color", image_name)
            sketch_path = os.path.join(data_dir, cut_name, "sketch", image_name)

            shutil.copy(color_path, os.path.join(root_cut_dir, "color", image_name))
            shutil.copy(sketch_path, os.path.join(root_cut_dir, "sketch", image_name))
    return


def filter_new_data():
    root_dir = "D:/Data/GeekToys/coloring_data/pd_data/p1"
    output_dir = "D:/Data/GeekToys/coloring_data/pd_data/pd1"
    sample_dir = "D:/Data/GeekToys/coloring_data/pd_data/sample"

    all_cuts = natsorted(glob.glob(os.path.join(root_dir, "*")))
    all_cuts = [cut for cut in all_cuts if ".zip" not in cut]

    bad_names = [
        "PD09_048_k_a", "PD09_068_k_a", "PD09_074_k_a", "PD09_098_k_a", "PD09_127_k_a",
        "PD09_143_k_a", "PD10_003_a", "PD10_013_a", "PD10_025_a", "PD10_064_071_a", "PD10_081_a",
        "PD10_089_a", "PD10_107_114_a", "PD10_109_a", "PD10_112_a", "PD10_116_a", "PD10_221_a",
        "PD10_245_a"]

    for cut_dir in all_cuts:
        cut_name = os.path.basename(cut_dir)
        cut_parts = natsorted(glob.glob(os.path.join(cut_dir, "*")))

        for part_dir in cut_parts:
            part_name = os.path.basename(part_dir)
            output_dir_name = "%s_%s" % (cut_name, part_name)
            if part_name != "a" or output_dir_name in bad_names:
                continue

            paths = natsorted(glob.glob(os.path.join(part_dir, "*.tga")))
            if len(paths) < 3 or len(paths) > 15:
                continue

            os.makedirs(os.path.join(output_dir, output_dir_name, "color"), exist_ok=True)
            sample_path = paths[0]
            shutil.copy(sample_path, os.path.join(sample_dir, "%s.tga" % output_dir_name))
            continue

            for path in paths:
                name = os.path.basename(path)
                shutil.copy(path, os.path.join(output_dir, output_dir_name, "color", name))
    return


def unzip_data():
    import zipfile

    root_dir = "D:/Data/GeekToys/coloring_data/pd_data/p2"
    all_zip = natsorted(glob.glob(os.path.join(root_dir, "*.zip")))

    for zip_path in all_zip:
        zip_name = os.path.basename(zip_path)
        try:
            unzip(zip_path, root_dir, root_only=True)
        except zipfile.BadZipFile:
            print(zip_name)
            continue
    return


if __name__ == "__main__":
    filter_new_data()
