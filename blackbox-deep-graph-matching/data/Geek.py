import pickle
import random
import xml.etree.ElementTree as ET
import cv2
import json
import os
from pathlib import Path
from natsort import natsorted


import numpy as np
from PIL import Image


from utils.config import cfg
from utils_zed.read_matching_info import build_pairs_path, read_matching_info
from utils.utils import lexico_iter

anno_path = cfg.Geek.KPT_ANNO_DIR
img_path = cfg.Geek.ROOT_DIR + "image"
ori_anno_path = cfg.Geek.ROOT_DIR + "Annotations"
set_path = cfg.Geek.SET_SPLIT
cache_path = cfg.CACHE_PATH

KPT_NAMES = {
    # "character": [
    #     "l eye_1",
    #     "mouth_1",
    #     "r eye_1",
    #     "nose_1",
    #     "chin_1"
    # ]
    "character": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "130",
        "131",
        "132",
        "133",
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "140",
        "141",
        "142",
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "233",
        "234",
        "235",
        "236",
        "237",
        "238",
        "239",
        "240",
        "241",
        "242",
        "243",
        "244",
        "245",
        "246",
        "247",
        "248",
        "249",
        "250",
        "251",
        "252",
        "253",
        "254",
        "255",
        "256",
        "257",
        "258",
        "259",
        "260",
        "261",
        "262",
        "263",
        "264",
        "265",
        "266",
        "267",
        "268",
        "269",
        "270",
        "271",
        "272",
        "273",
        "274",
        "275",
        "276",
        "277",
        "278",
        "279",
        "280",
        "281",
        "282",
        "283",
        "284",
        "285",
        "286",
        "287",
        "288",
        "289",
        "290",
        "291",
        "292",
        "293",
        "294",
        "295",
        "296",
        "297",
        "298",
        "299",
        "300",
        "301",
        "302",
        "303",
        "304",
        "305",
        "306",
        "307",
        "308",
        "309",
        "310",
        "311",
        "312",
        "313",
        "314",
        "315",
        "316",
        "317",
        "318",
        "319",
        "320",
        "321",
        "322",
        "323",
        "324",
        "325",
        "326",
        "327",
        "328",
        "329",
        "330",
        "331",
        "332",
        "333",
        "334",
        "335",
        "336",
        "337",
        "338",
        "339",
        "340",
        "341",
        "342",
        "343",
        "344",
        "345",
        "346",
        "347",
        "348",
        "349",
        "350",
        "351",
        "352",
        "353",
        "354",
        "355",
        "356",
        "357",
        "358",
        "359",
        "360",
        "361",
        "362",
        "363",
        "364",
        "365",
        "366",
        "367",
        "368",
        "369",
        "370",
        "371",
        "372",
        "373",
        "374",
        "375",
        "376",
        "377",
        "378",
        "379",
        "380",
        "381",
        "382",
        "383",
        "384",
        "385",
        "386",
        "387",
        "388",
        "389",
        "390",
        "391",
        "392",
        "393",
        "394",
        "395",
        "396",
        "397",
        "398",
        "399",
        "400",
        "401",
        "402",
        "403",
        "404",
        "405",
        "406",
        "407",
        "408",
        "409",
        "410",
        "411",
        "412",
        "413",
        "414",
        "415",
        "416",
        "417",
        "418",
        "419",
        "420",
        "421",
        "422",
        "423",
        "424",
        "425",
        "426",
        "427",
        "428",
        "429",
        "430",
        "431",
        "432",
        "433",
        "434",
        "435",
        "436",
        "437",
        "438",
        "439",
        "440",
        "441",
        "442",
        "443",
        "444",
        "445",
        "446",
        "447",
        "448",
        "449",
        "450",
        "451",
        "452",
        "453",
        "454",
        "455",
        "456",
        "457",
        "458",
        "459",
        "460",
        "461",
        "462",
        "463",
        "464",
        "465",
        "466",
        "467",
        "468",
        "469",
        "470",
        "471",
        "472",
        "473",
        "474",
        "475",
        "476",
        "477",
        "478",
        "479",
        "480",
        "481",
        "482",
        "483",
        "484",
        "485",
        "486",
        "487",
        "488",
        "489",
        "490",
        "491",
        "492",
        "493",
        "494",
        "495",
        "496",
        "497",
        "498",
        "499",
        "500",
    ]
}


class Geek:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.classes = cfg.Geek.CLASSES
        self.folder_list = self.get_list_folder()
        
        self.kpt_len = [len(KPT_NAMES[_]) for _ in cfg.Geek.CLASSES]

        self.classes_kpts = {cls: len(KPT_NAMES[cls]) for cls in self.classes}

        self.anno_path = Path(anno_path)
        self.img_path = Path(img_path)
        # self.ori_anno_path = Path(ori_anno_path)
        self.obj_resize = obj_resize
        self.sets = sets

        assert sets in ["train", "test"], "No match found for dataset {}".format(sets)
        cache_name = "geek_" + sets + ".pkl"
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        # if self.cache_file.exists():
        #     with self.cache_file.open(mode="rb") as f:
        #         self.xml_list = pickle.load(f)
        #     print("xml list loaded from {}".format(self.cache_file))
        # else:
        print("Caching xml list to {}...".format(self.cache_file))
        self.cache_path.mkdir(exist_ok=True, parents=True)
        with np.load(set_path, allow_pickle=True) as f:
            self.xml_list = f[sets]

        # print(self.xml_list)
        before_filter = sum([len(k) for k in self.xml_list])
        # self.filter_list()
        after_filter = sum([len(k) for k in self.xml_list])
        # self.xml_list = []
        with self.cache_file.open(mode="wb") as f:
            pickle.dump(self.xml_list, f)
        print("Filtered {} images to {}. Annotation saved.".format(before_filter, after_filter))

    def get_list_folder(self):
        color_image_folder = os.path.join(img_path, "color")
        list_folder = []
        for folder_name in natsorted(os.listdir(color_image_folder)):
            list_folder.append(folder_name)

        return list_folder

    def get_k_samples(self, idx, k, mode, cls=None, folder=None, shuffle=True, num_iterations=200):
        """
        Randomly get a sample of k objects from Geek dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param num_iterations: maximum number of iterations for sampling a datapoint
        :return: (k samples of data, k \choose 2 groundtruth permutation matrices)
        """
        if idx is not None:
            raise NotImplementedError("No indexed sampling implemented for PVOC.")
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)

        if folder is None:
            folder_id = random.randrange(0, len(self.folder_list))
            folder_name = self.folder_list[folder_id]
        elif type(folder) == str:
            folder_name = folder
            folder_id = self.folder_list.index(folder)

        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_list = []
        
        for xml_name in random.sample(list(self.xml_list[folder_id]), k): #replace cls by folder
            anno_dict = self.__get_anno_dict_v1(xml_name, cls, folder_name) #replace cls by folder
            if shuffle:

                random.shuffle(anno_dict["keypoints"])
                # shuffle all keypoint
            anno_list.append(anno_dict)

        # if shuffle:
        #     for anno_dict in anno_list:
        #         random.shuffle(anno_dict["keypoints"])

        # build permutation matrices

        perm_mat_list = [
            np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32) for anno_pair in lexico_iter(anno_list)
        ]
        for n, (s1, s2) in enumerate(lexico_iter(anno_list)):
            image_name_1 = s1["image_name"]
            image_name_2 = s2["image_name"]

            pair_file_name = build_pairs_path(image_name_1, image_name_2)
            # path to info of pair matching
            full_path = os.path.join(cfg.Geek.ROOT_DIR, "matching_info", folder_name, pair_file_name+".json")
            with open(full_path) as f:
                pair_match_info = json.load(f)
            
            #exchange s1, s2 so that s1 should be left and s2 should be right
            name_img_left = pair_match_info["imagePathLeft"].split("/")[-1][-4:]
            name_img_right = pair_match_info["imagePathRight"].split("/")[-1][-4:]

            if image_name_1 == name_img_right:
                # exchange
                tmp = s1 
                s1 = s2
                s2 = tmp 
                image_name_1 = s1["image_name"]
                image_name_2 = s2["image_name"]
                perm_mat_list[n] = np.transpose(perm_mat_list[n])

            pair_match_info = pair_match_info["pairs"]
            for i, keypoint in enumerate(s1["keypoints"]):
                for j, _keypoint in enumerate(s2["keypoints"]):
                    # if keypoint["name"] == _keypoint["name"]:
                    #     perm_mat_list[n][i, j] = 1
                    key_of_pair = keypoint["name"]+"_"+_keypoint["name"]
                    
                    

                    # check if key of pair exist in annotation file in any order
                    if key_of_pair in pair_match_info:
                        perm_mat_list[n][i, j] = 1


        return anno_list, perm_mat_list


    def __get_anno_dict(self, xml_name, cls):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), "{} does not exist.".format(xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        img_name = root.find("./image").text + ".png"
        img_file = self.img_path / img_name
        bounds = root.find("./visible_bounds").attrib

        h = float(bounds["height"])
        w = float(bounds["width"])
        # geek
        image_name = root.find("./image").text
        type_image = root.find("./type").text

        xmin = float(bounds["xmin"])
        ymin = float(bounds["ymin"])

        with Image.open(str(img_file)) as img:

            if type_image == "color":
                tyler_input_img = img
                tyler_input_img = tyler_input_img.convert("RGB")
                tyler_input_img = cv2.cvtColor(np.array(tyler_input_img), cv2.COLOR_RGB2BGR)

            elif type_image == "sketch":
                tyler_input_img = np.array(img)

            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))
            # print(obj.shape)
            # obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

        keypoint_list = []
        for keypoint in root.findall("./keypoints/keypoint"):
            attr = keypoint.attrib
            attr["x"] = (float(attr["x"]) - xmin) * self.obj_resize[0] / w
            attr["y"] = (float(attr["y"]) - ymin) * self.obj_resize[1] / h
            if -1e-5 < attr["x"] < self.obj_resize[0] + 1e-5 and -1e-5 < attr["y"] < self.obj_resize[1] + 1e-5:
                keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict["image"] = obj
        anno_dict["tyler_image"] = tyler_input_img
        anno_dict["keypoints"] = keypoint_list
        anno_dict["bounds"] = xmin, ymin, w, h
        anno_dict["ori_sizes"] = ori_sizes
        anno_dict["cls"] = self.classes[cls]
        
        # Geek
        anno_dict["image_name"] = image_name
        anno_dict["type_name"] = type_image

        return anno_dict

    def __get_anno_dict_v1(self, xml_name, cls, folder_name):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), "{} does not exist.".format(xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        
        bounds = root.find("./visible_bounds").attrib

        h = float(bounds["height"])
        w = float(bounds["width"])
        # geek
        image_name = root.find("./image").text
        png_img = image_name + ".png"
        type_image = root.find("./type").text
        img_file = self.img_path / type_image / folder_name / png_img
        

        xmin = float(bounds["xmin"])
        ymin = float(bounds["ymin"])

        with Image.open(str(img_file)) as img:

            if type_image == "color":
                tyler_input_img = img
                tyler_input_img = tyler_input_img.convert("RGB")
                tyler_input_img = cv2.cvtColor(np.array(tyler_input_img), cv2.COLOR_RGB2BGR)

            elif type_image == "sketch":
                tyler_input_img = np.array(img)

            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))
            # print(obj.shape)
            # obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

        keypoint_list = []
        for keypoint in root.findall("./keypoints/keypoint"):
            attr = keypoint.attrib
            attr["x"] = (float(attr["x"]) - xmin) * self.obj_resize[0] / w
            attr["y"] = (float(attr["y"]) - ymin) * self.obj_resize[1] / h
            if -1e-5 < attr["x"] < self.obj_resize[0] + 1e-5 and -1e-5 < attr["y"] < self.obj_resize[1] + 1e-5:
                keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict["image"] = obj
        anno_dict["tyler_image"] = tyler_input_img
        anno_dict["keypoints"] = keypoint_list
        anno_dict["bounds"] = xmin, ymin, w, h
        anno_dict["ori_sizes"] = ori_sizes
        anno_dict["cls"] = self.classes[cls]
        
        # Geek
        anno_dict["image_name"] = image_name
        anno_dict["type_name"] = type_image
        anno_dict["folder_name"] = folder_name

        return anno_dict