
import os 
import cv2 
import shutil 
from PIL import Image
def mkdirs(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def prepare_image(path_image_all, path_image_out):
    '''
        Preparing image including both color and sketch(extracted from color) for
        standardized training data
    '''
    num_folder = 0
    num_image = 0

    for folder_dir in natsorted(os.listdir(path_image_all)):
        if not os.path.isdir(join_two(path_image_all, folder_dir)):
            continue 

        if folder_dir in drop_folder:
            continue

        print("Process folder:", folder_dir)
        color_path_in = join_three(path_image_all, folder_dir, "color")

        color_path_out = join_three(path_image_out, "color", folder_dir)
        sketch_path_out = join_three(path_image_out, "sketch", folder_dir)

        mkdirs(color_path_out)
        mkdirs(sketch_path_out)

        allowed_ext = [".tga", ".png"]
        for each_image in os.listdir(color_path_in):
            ext = each_image[-4:]
            if ext not in allowed_ext:
                continue 
            full_path_img = join_two(color_path_in, each_image)
            color_img = read_img(full_path_img)
            out_sketch, out_color = get_sketch(color_img)
            
            shutil.copy(full_path_img, join_two(color_path_out, each_image[:-4]+".png"))
            cv2.imwrite(join_two(sketch_path_out, each_image[:-4]+".png"), out_sketch)

            num_image += 1
        
        num_folder += 1

    print("Finish processing " + str(num_folder) +" folders")
    print("Finish processing " + str(num_image) +" images")

def process_annot():



if __name__ == "__main__":
    folder_name = "hor02_055"
    in_data_path = "../../stuff/Geek_data/processed_HOR02/"+folder_name+"/color"
    out_data_path = "../inference/"+folder_name

    