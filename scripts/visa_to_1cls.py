"""VisA raw -> 1cls converter (mirrors amazon-science/spot-diff prepare_data.py)."""
import os, shutil, csv, argparse

p = argparse.ArgumentParser()
p.add_argument("--data-folder", required=True)
p.add_argument("--save-folder", required=True)
p.add_argument("--split-file", required=True)
args = p.parse_args()

save_root = os.path.join(args.save_folder, "1cls")
n_img, n_mask = 0, 0

with open(args.split_file, "r", newline="") as f:
    for row in csv.DictReader(f):
        obj = row["object"].strip()
        split = row["split"].strip()
        label = row["label"].strip()
        img_rel = row["image"].strip().replace("/", os.sep)
        mask_rel = row.get("mask", "").strip().replace("/", os.sep)

        sub = "good" if label == "normal" else "bad"
        dst_split = "test" if label == "anomaly" else split
        dst_dir = os.path.join(save_root, obj, dst_split, sub)
        os.makedirs(dst_dir, exist_ok=True)

        src = os.path.join(args.data_folder, img_rel)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(img_rel)))
        n_img += 1

        if label == "anomaly" and mask_rel:
            gt_dir = os.path.join(save_root, obj, "ground_truth", "bad")
            os.makedirs(gt_dir, exist_ok=True)
            msrc = os.path.join(args.data_folder, mask_rel)
            shutil.copy2(msrc, os.path.join(gt_dir, os.path.basename(mask_rel)))
            n_mask += 1

        if n_img % 1000 == 0:
            print(f"  {n_img} images, {n_mask} masks...")

print(f"\nDone. Total: {n_img} images, {n_mask} masks copied to {save_root}")