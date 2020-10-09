import configparser
import zipfile


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config_value(func, section, option, default_value, is_check_required=False):
    try:
        value = func(section, option)
    except configparser.NoOptionError:
        value = default_value

    if is_check_required and value is None:
        raise ValueError("PLEASE FILL THIS FIELD: section - {}, option - {}".format(section, option))
    return value


def convert_to_object(dict_config):
    config = Config(**dict_config)
    return config


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    dict_config = {}

    common_section = "COMMON"
    dict_config["experiment_name"] = get_config_value(config.get, common_section, "experiment_name", None)
    dict_config["gpu"] = get_config_value(config.getint, common_section, "gpu", 0)
    dict_config["spot_checkpoint"] = get_config_value(config.get, common_section, "spot_checkpoint", None)
    dict_config["output_dir"] = get_config_value(config.get, common_section, "output_dir", None)
    dict_config["device"] = get_config_value(config.get, common_section, "device", None)
    dict_config["progress_iter"] = get_config_value(config.getint, common_section, "progress_iter", 100)

    data_section = "DATA"
    dict_config["data_zip_file"] = get_config_value(config.get, data_section, "data_zip_file", None)
    dict_config["data_folder"] = get_config_value(config.get, data_section, "data_folder", None)
    dict_config["image_size"] = get_config_value(config.get, data_section, "image_size", 512)

    train_section = "TRAIN"
    dict_config["learning_rate"] = get_config_value(config.getfloat, train_section, "learning_rate", 0.0002)
    dict_config["batch_size"] = get_config_value(config.getint, train_section, "batch_size", 1)
    dict_config["num_epochs"] = get_config_value(config.getint, train_section, "num_epochs", 25)
    dict_config["step_size"] = get_config_value(config.get, train_section, "step_size", 12)
    dict_config["print_freq"] = get_config_value(config.get, train_section, "print_freq", 100)
    dict_config["save_freq"] = get_config_value(config.get, train_section, "save_freq", 2)

    return dict_config


def unzip(zip_file, to_folder, root_only=False):
    zip_ref = zipfile.ZipFile(zip_file, "r")

    if root_only:
        info_list = zip_ref.infolist()
        count = len([x for x in info_list if x.filename.count("/") == 1])
        if count > 1:
            return

    zip_ref.extractall(to_folder)
    zip_ref.close()
