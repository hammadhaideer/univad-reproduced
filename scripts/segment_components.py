import argparse
import os
import glob

import yaml

from models.component_segmentaion import grounding_segmentation


def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def segment_dataset(categories, data_root, mask_root, ext="*.png", configs_root="./configs/class_histogram"):
    for category in categories:
        for split in ["test", "train"]:
            image_paths = sorted(glob.glob(
                os.path.join(data_root, category, split, "*", ext)
            ))
            config = read_config(os.path.join(configs_root, f"{category}.yaml"))
            out_dir = os.path.join(mask_root, category)
            os.makedirs(out_dir, exist_ok=True)
            grounding_segmentation(image_paths, out_dir, config["grounding_config"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Segment Components", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--mask_path", type=str, default="./masks")
    parser.add_argument("--configs_root", type=str, default="./configs/class_histogram")
    args = parser.parse_args()

    data_path = args.data_path
    mask_path = args.mask_path
    configs_root = args.configs_root

    mvtec_categories = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    ]
    segment_dataset(
        mvtec_categories,
        data_root=os.path.join(data_path, "mvtec"),
        mask_root=os.path.join(mask_path, "mvtec"),
        ext="*.png",
        configs_root=configs_root,
    )

    visa_categories = [
        "candle", "capsules", "chewinggum", "cashew", "fryum", "pipe_fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
    ]
    segment_dataset(
        visa_categories,
        data_root=os.path.join(data_path, "VisA_pytorch", "1cls"),
        mask_root=os.path.join(mask_path, "VisA_pytorch", "1cls"),
        ext="*.JPG",
        configs_root=configs_root,
    )

    loco_categories = [
        "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors",
    ]
    segment_dataset(
        loco_categories,
        data_root=os.path.join(data_path, "mvtec_loco_caption"),
        mask_root=os.path.join(mask_path, "mvtec_loco_caption"),
        ext="*.png",
        configs_root=configs_root,
    )

    medical_categories = ["LiverCT", "BrainMRI", "RESC", "HIS", "ChestXray"]
    segment_dataset(
        medical_categories,
        data_root=data_path,
        mask_root=mask_path,
        ext="*.png",
        configs_root=configs_root,
    )

    segment_dataset(
        ["OCT17"],
        data_root=data_path,
        mask_root=mask_path,
        ext="*.jpeg",
        configs_root=configs_root,
    )