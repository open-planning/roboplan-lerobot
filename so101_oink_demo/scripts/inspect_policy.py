"""Print the architecture and parameter counts of a trained LeRobot policy.

Loads any checkpoint (diffusion, ACT, ...) the same way eval_policy.py does and
prints the nested torch.nn.Module tree plus a per-block parameter summary.

Example:
    pixi run so101-inspect --policy-path outputs/train/so101_diffusion/checkpoints/last/pretrained_model
    pixi run so101-inspect --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --depth 2
"""

import argparse

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def n_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def fmt(n: int) -> str:
    return f"{n / 1e6:10.2f} M"


def print_summary(module: torch.nn.Module, depth: int, prefix: str = "", level: int = 0) -> None:
    for name, child in module.named_children():
        total = n_params(child)
        if total == 0:
            continue
        trainable = n_params(child, trainable_only=True)
        indent = "  " * level
        label = f"{indent}{name} ({type(child).__name__})"
        print(f"{label:60s} {fmt(total)}  trainable {fmt(trainable)}")
        if level + 1 < depth:
            print_summary(child, depth, prefix + name + ".", level + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="checkpoint pretrained_model dir")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="how many module levels to show in the summary",
    )
    parser.add_argument("--no-tree", action="store_true", help="skip the full print(model) module tree")
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path)
    policy.eval()

    if not args.no_tree:
        print("=" * 100)
        print(f"MODULE TREE  ({cfg.type})")
        print("=" * 100)
        print(policy)

    print("=" * 100)
    print(f"PARAMETER SUMMARY  ({cfg.type}, depth={args.depth})")
    print("=" * 100)
    print_summary(policy, args.depth)
    total = n_params(policy)
    trainable = n_params(policy, trainable_only=True)
    print("-" * 100)
    print(f"{'TOTAL':60s} {fmt(total)}  trainable {fmt(trainable)}")
    print(f"{'inputs':10s} " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in cfg.input_features.items()))
    print(f"{'outputs':10s} " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in cfg.output_features.items()))


if __name__ == "__main__":
    main()
