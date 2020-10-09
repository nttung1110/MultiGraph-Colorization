import sys
import os
import random
import argparse
from loader.config_utils import load_config, convert_to_object, unzip


def setup_environment(config):
    if os.path.isfile("requirements.txt"):
        os.system("pip install -r requirements.txt")

    import numpy as np
    import imgaug as ia
    import torch
    import torch.backends.cudnn as cudnn

    # Seed setting
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    ia.seed(1)

    # GPU setting
    if torch.cuda.is_available():
        if config["gpu"] is not None:
            device = torch.device("cuda:" + str(config["gpu"]))
            config["device"] = device
            cudnn.benchmark = True
            cudnn.deterministic = True
        else:
            device = torch.device("cpu")
            config["device"] = device
    else:
        device = torch.device("cpu")
        config["device"] = device

    # set name experiment
    if not config["experiment_name"]:
        config["experiment_name"] = "experiment"

    if "data_zip_file" in config and config["data_zip_file"]:
        zip_file = config["data_zip_file"]
        folder = os.path.dirname(zip_file)
        unzip(zip_file, folder)

    os.makedirs(config["spot_checkpoint"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--config")
    args = parser.parse_args(sys.argv[1:])

    config = load_config(args.config)
    setup_environment(config)

    from training.train import main
    main(convert_to_object(config))


if __name__ == "__main__":
    main()
