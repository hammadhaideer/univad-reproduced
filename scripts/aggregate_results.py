import json
import argparse
from pathlib import Path

PAPER_NUMBERS = {
    "mvtec":      {"image_auroc": 97.8, "pixel_auroc": 96.5},
    "visa":       {"image_auroc": 93.5, "pixel_auroc": 98.2},
    "mvtec_loco": {"image_auroc": 71.0, "pixel_auroc": 75.1},
    "brainmri":   {"image_auroc": 80.2, "pixel_auroc": 96.8},
    "liverct":    {"image_auroc": 70.0, "pixel_auroc": 96.3},
    "retinalOCT": {"image_auroc": 85.5, "pixel_auroc": 94.9},
}


def load_results(results_dir: str) -> dict:
    results = {}
    for path in Path(results_dir).glob("*.json"):
        with open(path) as f:
            results[path.stem] = json.load(f)
    return results


def print_table(results: dict):
    header = f"{'Dataset':<15} {'Img-AUROC (Paper)':<20} {'Img-AUROC (Ours)':<20} {'Δ':<8} {'Px-AUROC (Paper)':<20} {'Px-AUROC (Ours)':<20} {'Δ':<8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for dataset, paper in PAPER_NUMBERS.items():
        if dataset in results:
            ours_img = results[dataset].get("image_auroc", None)
            ours_pix = results[dataset].get("pixel_auroc", None)
            delta_img = f"{ours_img - paper['image_auroc']:+.1f}" if ours_img else "TBD"
            delta_pix = f"{ours_pix - paper['pixel_auroc']:+.1f}" if ours_pix else "TBD"
            img_str = f"{ours_img:.1f}" if ours_img else "TBD"
            pix_str = f"{ours_pix:.1f}" if ours_pix else "TBD"
        else:
            img_str = pix_str = delta_img = delta_pix = "TBD"

        print(f"{dataset:<15} {paper['image_auroc']:<20} {img_str:<20} {delta_img:<8} {paper['pixel_auroc']:<20} {pix_str:<20} {delta_pix:<8}")

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print_table(results)


if __name__ == "__main__":
    main()