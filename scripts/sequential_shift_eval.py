import json
import argparse
from pathlib import Path


PROTOCOLS = {
    "S1_category": [
        {"dataset": "mvtec", "category": "bottle"},
        {"dataset": "mvtec", "category": "cable"},
        {"dataset": "mvtec", "category": "capsule"},
        {"dataset": "mvtec", "category": "carpet"},
        {"dataset": "mvtec", "category": "grid"},
    ],
    "S2_dataset": [
        {"dataset": "mvtec",      "category": "all"},
        {"dataset": "visa",       "category": "all"},
        {"dataset": "mvtec_loco", "category": "all"},
        {"dataset": "brainmri",   "category": "all"},
    ],
    "S3_modality": [
        {"dataset": "brainmri",   "category": "all"},
        {"dataset": "liverct",    "category": "all"},
        {"dataset": "retinalOCT", "category": "all"},
    ],
}


def compute_bwt(results: list[float]) -> float:
    """Backward Transfer: mean AUROC drop on earlier domains."""
    if len(results) < 2:
        return 0.0
    drops = [results[i] - results[-1] for i in range(len(results) - 1)]
    return sum(drops) / len(drops)


def compute_af(peak: list[float], final: list[float]) -> float:
    """Average Forgetting: mean drop from peak to final performance."""
    drops = [p - f for p, f in zip(peak, final)]
    return sum(drops) / len(drops)


def run_protocol(protocol_name: str, steps: list[dict], args) -> dict:
    print(f"\n{'='*60}")
    print(f"Protocol: {protocol_name}")
    print(f"{'='*60}")

    auroc_history = []
    peak_auroc = []

    for step_idx, step in enumerate(steps):
        dataset = step["dataset"]
        category = step["category"]
        print(f"\nStep {step_idx + 1}: {dataset} / {category}")
        print("  [UniVAD frozen — no adaptation]")

        # TODO: replace with actual UniVAD inference call
        # auroc = run_univad_inference(dataset, category, args)
        auroc = None  # placeholder

        if auroc is not None:
            auroc_history.append(auroc)
            peak_auroc.append(auroc)
            print(f"  Image-AUROC: {auroc:.1f}")

    bwt = compute_bwt(auroc_history) if auroc_history else None
    af  = compute_af(peak_auroc, auroc_history) if auroc_history else None

    return {
        "protocol": protocol_name,
        "steps": steps,
        "auroc_per_step": auroc_history,
        "bwt": bwt,
        "af": af,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sequential shift evaluation for frozen UniVAD"
    )
    parser.add_argument(
        "--protocol",
        choices=["S1_category", "S2_dataset", "S3_modality", "all"],
        default="all",
    )
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--output", default="results/sequential_shift_results.json")
    args = parser.parse_args()

    protocols_to_run = (
        PROTOCOLS if args.protocol == "all"
        else {args.protocol: PROTOCOLS[args.protocol]}
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, steps in protocols_to_run.items():
        all_results[name] = run_protocol(name, steps, args)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()