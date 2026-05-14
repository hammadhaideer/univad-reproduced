import argparse
import logging
import os
import sys
import threading
import math

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
from prefetch_generator import BackgroundGenerator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from UniVAD import UniVAD
from mvtec import MVTecDataset
from visa import VisaDataset
from mvtec_loco import MVTecLocoDataset
from brainmri import BrainMRIDataset
from his import HISDataset
from resc import RESCDataset
from liverct import LiverCTDataset
from chestxray import ChestXrayDataset
from oct17 import OCT17Dataset

results_lock = threading.Lock()


def resize_tokens(x):
    B, N, C = x.shape
    h = int(math.sqrt(N))
    return x.view(B, h, h, C)


def cal_score(obj, results, table_ls, auroc_sp_ls, auroc_px_ls):
    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []

    for i in range(len(results["cls_names"])):
        if results["cls_names"][i] == obj:
            gt_px.append(results["imgs_masks"][i].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][i])
            gt_sp.append(results["gt_sp"][i])
            pr_sp.append(results["pr_sp"][i])

    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)

    if len(np.unique(gt_sp)) < 2 or len(np.unique(gt_px.ravel())) < 2:
        logging.getLogger("test").warning(
            "Skipping %s — single class in labels", obj
        )
        return

    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())

    row = [
        obj,
        str(np.round(auroc_sp * 100, decimals=1)),
        str(np.round(auroc_px * 100, decimals=1)),
    ]

    with results_lock:
        table_ls.append(row)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("UniVAD Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--class_name", type=str, default="None")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    dataset_name = args.dataset
    device = args.device
    k_shot = args.k_shot
    image_size = args.image_size
    data_path = args.data_path

    save_path = os.path.join(args.save_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, "log.txt")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("test")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    for arg in vars(args):
        logger.info("%s: %s", arg, getattr(args, arg))

    UniVAD_model = UniVAD(image_size=image_size).to(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    dataset_roots = {
        "mvtec":      os.path.join(data_path, "mvtec"),
        "visa":       os.path.join(data_path, "VisA_pytorch", "1cls"),
        "mvtec_loco": os.path.join(data_path, "mvtec_loco_caption"),
        "brainmri":   os.path.join(data_path, "BrainMRI"),
        "his":        os.path.join(data_path, "HIS"),
        "resc":       os.path.join(data_path, "RESC"),
        "chestxray":  os.path.join(data_path, "ChestXray"),
        "oct17":      os.path.join(data_path, "OCT17"),
        "liverct":    os.path.join(data_path, "LiverCT"),
    }

    dataset_classes = {
        "mvtec":      MVTecDataset,
        "visa":       VisaDataset,
        "mvtec_loco": MVTecLocoDataset,
        "brainmri":   BrainMRIDataset,
        "his":        HISDataset,
        "resc":       RESCDataset,
        "chestxray":  ChestXrayDataset,
        "oct17":      OCT17Dataset,
        "liverct":    LiverCTDataset,
    }

    if dataset_name not in dataset_classes:
        raise NotImplementedError(f"Dataset '{dataset_name}' not supported.")

    test_data = dataset_classes[dataset_name](
        root=dataset_roots[dataset_name],
        transform=transform,
        target_transform=transform,
        aug_rate=-1,
        mode="test",
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    results = {
        "cls_names": [],
        "imgs_masks": [],
        "anomaly_maps": [],
        "gt_sp": [],
        "pr_sp": [],
    }

    cls_last = None

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    for items in tqdm(test_dataloader):
        image = items["img"].to(device)
        image_pil = items["img_pil"]
        image_path = items["img_path"][0]

        if args.class_name != "None" and args.class_name not in image_path:
            continue

        cls_name = items["cls_name"][0]
        results["cls_names"].append(cls_name)

        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5] = 1
        gt_mask[gt_mask <= 0.5] = 0
        results["imgs_masks"].append(gt_mask)
        results["gt_sp"].append(items["anomaly"].item())

        if cls_name != cls_last:
            if dataset_name == "mvtec":
                normal_image_paths = [
                    os.path.join(
                        dataset_roots["mvtec"],
                        cls_name.replace(" ", "_"),
                        "train", "good",
                        str(i).zfill(3) + ".png",
                    )
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "mvtec_loco":
                normal_image_paths = [
                    os.path.join(
                        dataset_roots["mvtec_loco"],
                        cls_name.replace(" ", "_"),
                        "train", "good",
                        str(i).zfill(3) + ".png",
                    )
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "visa":
                visa_root = dataset_roots["visa"]
                cls_key = cls_name.replace(" ", "_")
                short_pad = ["capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
                zfill_n = 3 if cls_key in short_pad else 4
                normal_image_paths = [
                    os.path.join(
                        visa_root, cls_key, "train", "good",
                        str(i).zfill(zfill_n) + ".JPG",
                    )
                    for i in range(args.round, args.round + k_shot)
                ]
            else:
                ref_dir = os.path.join(
                    dataset_roots[dataset_name],
                    cls_name.replace(" ", "_"),
                    "train", "good",
                )
                files = sorted(os.listdir(ref_dir))[:k_shot]
                normal_image_paths = [os.path.join(ref_dir, f) for f in files]

            normal_images = torch.cat(
                [
                    image_transform(Image.open(x).convert("RGB")).unsqueeze(0)
                    for x in normal_image_paths
                ],
                dim=0,
            ).to(device)

            setup_data = {
                "few_shot_samples": normal_images,
                "dataset_category": cls_name.replace(" ", "_"),
                "image_path": normal_image_paths,
            }
            UniVAD_model.setup(setup_data)
            cls_last = cls_name

        with torch.no_grad():
            pred_value = UniVAD_model(image, image_path, image_pil)
            anomaly_score = pred_value["pred_score"]
            anomaly_map = pred_value["pred_mask"]
            results["anomaly_maps"].append(anomaly_map.detach().cpu().numpy())
            results["pr_sp"].append(anomaly_score.item())

    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []

    threads = []
    for obj in tqdm(obj_list):
        t = threading.Thread(
            target=cal_score,
            args=(obj, results, table_ls, auroc_sp_ls, auroc_px_ls),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    table_ls.append([
        "mean",
        str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
        str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
    ])

    result_table = tabulate(
        table_ls,
        headers=["objects", "auroc_sp", "auroc_px"],
        tablefmt="pipe",
    )
    logger.info("\n%s", result_table)
