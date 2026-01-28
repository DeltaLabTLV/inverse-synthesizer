# main.py
# One entrypoint to run:
#  - pretrain transformer + autoencoder
#  - train transformer (decoder-only then full) + autoencoder (no mask)
#  - test transformer + autoencoder
#
# Usage examples:
#   python main.py --config configs/experiments.yaml --run all
#   python main.py --config configs/experiments.yaml --run pretrain
#   python main.py --config configs/experiments.yaml --run train
#   python main.py --config configs/experiments.yaml --run test
#   python main.py --config configs/experiments.yaml --run stage --stage pretrain_transformer
#
# Notes:
# - This script assumes you are using the YAML structure I gave earlier:
#     experiment: <name>
#     experiments: { <name>: {...} }
# - It merges base cfg + selected experiment preset and then calls train_unified.run(cfg).

import os
import json
import argparse
from copy import deepcopy
from typing import Any, Dict, List

import yaml

from train_unified import run as run_stage


# -----------------------------
# Config helpers
# -----------------------------
def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_cfg_for_experiment(cfg_all: Dict[str, Any], exp_name: str) -> Dict[str, Any]:
    if "experiments" not in cfg_all or exp_name not in cfg_all["experiments"]:
        raise ValueError(f"Experiment '{exp_name}' not found under cfg['experiments']")

    base = deepcopy(cfg_all)
    exp = deepcopy(cfg_all["experiments"][exp_name])

    # Merge preset into base
    merged = deep_update(base, exp)

    # Clean up to avoid accidental nesting usage
    merged.pop("experiments", None)
    merged["experiment_name"] = merged.get("experiment_name", exp_name)

    return merged


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def dump_effective_config(cfg: Dict[str, Any], out_dir: str, name: str):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"effective_{name}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[main] wrote effective config: {path}")


# -----------------------------
# Pipeline definitions
# -----------------------------
PIPELINES: Dict[str, List[str]] = {
    # Pretrain both
    "pretrain": [
        "pretrain_autoencoder_mae",
        "pretrain_transformer_contrastive",
    ],
    # Train/fine-tune both
    "train": [
        "train_transformer_decoder",
        "train_full_transformer",
        "train_autoencoder_nomask_finetune",
    ],
    # Test both
    "test": [
        "test_autoencoder",
        "test_transformer",
    ],
    # Everything
    "all": [
        "pretrain_autoencoder_mae",
        "pretrain_transformer_contrastive",
        "train_transformer_decoder",
        "train_full_transformer",
        "train_autoencoder_nomask_finetune",
        "test_autoencoder",
        "test_transformer",
    ],
}


def map_experiment_to_stage(exp_name: str) -> str:
    """
    train_unified.py expects cfg["stage"] values:
      - pretrain_autoencoder
      - pretrain_transformer
      - train_transformer_decoder
      - train_full_transformer
      - train_autoencoder_nomask
      - test_autoencoder
      - test_transformer

    Our YAML experiment names differ slightly; map them here.
    """
    mapping = {
        "pretrain_autoencoder_mae": "pretrain_autoencoder",
        "pretrain_transformer_contrastive": "pretrain_transformer",
        "train_transformer_decoder": "train_transformer_decoder",
        "train_full_transformer": "train_full_transformer",
        "train_autoencoder_nomask_finetune": "train_autoencoder_nomask",
        "test_autoencoder": "test_autoencoder",
        "test_transformer": "test_transformer",
    }
    if exp_name not in mapping:
        raise ValueError(f"No stage mapping for experiment: {exp_name}")
    return mapping[exp_name]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to configs/experiments.yaml")
    ap.add_argument("--run", type=str, default="all", choices=["all", "pretrain", "train", "test", "stage"])
    ap.add_argument("--stage", type=str, default=None, help="When --run stage, one of experiment names in YAML")
    ap.add_argument("--dump_effective", action="store_true", help="Save effective JSON config per stage into out_dir")
    args = ap.parse_args()

    cfg_all = load_yaml(args.config)

    if args.run == "stage":
        if not args.stage:
            raise ValueError("--stage is required when --run stage")
        exp_list = [args.stage]
    else:
        exp_list = PIPELINES[args.run]

    for exp_name in exp_list:
        # Build merged config from YAML preset
        cfg = build_cfg_for_experiment(cfg_all, exp_name)

        # Convert preset to the trainer's stage name
        cfg["stage"] = map_experiment_to_stage(exp_name)

        # Make sure experiment_name exists
        cfg["experiment_name"] = cfg.get("experiment_name", exp_name)

        # Optionally dump the merged JSON config so you can reproduce runs easily
        if args.dump_effective:
            out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
            dump_effective_config(cfg, out_dir, exp_name)

        print(f"\n==============================")
        print(f"[main] Running: {exp_name}  -> stage={cfg['stage']}")
        print(f"==============================\n")

        run_stage(cfg)

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
