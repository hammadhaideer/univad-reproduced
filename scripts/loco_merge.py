"""MVTec LOCO raw -> Caption: build ground_truth_merge_mask via OR-merge of component masks."""
import os, argparse
from PIL import Image
import numpy as np

CLASSES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
DEFECTS = ['logical_anomalies', 'structural_anomalies']

def merge_dir(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    if not files:
        return None
    out = None
    for f in files:
        m = np.array(Image.open(os.path.join(folder, f)).convert('L'))
        m = (m > 0).astype(np.uint8) * 255
        out = m if out is None else np.maximum(out, m)
    return Image.fromarray(out, mode='L')

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, help='mvtec_loco_caption root')
args = p.parse_args()

total = 0
for cls in CLASSES:
    for defect in DEFECTS:
        gt_dir = os.path.join(args.root, cls, 'ground_truth', defect)
        out_dir = os.path.join(args.root, cls, 'ground_truth_merge_mask', f'{defect}_merge_mask')
        if not os.path.isdir(gt_dir):
            print(f"[skip] {gt_dir} not found")
            continue
        os.makedirs(out_dir, exist_ok=True)
        subs = sorted([d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))])
        for sub in subs:
            merged = merge_dir(os.path.join(gt_dir, sub))
            if merged is None:
                print(f"[warn] no masks in {gt_dir}/{sub}")
                continue
            merged.save(os.path.join(out_dir, f"{sub}.png"))
            total += 1
        print(f"  {cls}/{defect}: merged {len(subs)} masks")

print(f"\nDone. {total} merged masks total.")