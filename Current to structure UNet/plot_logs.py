# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from utils.plot_loss import plot_iteration_val_accuracy, plot_training_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Plot current-to-structure training logs")
    parser.add_argument("--kind", choices=("epoch", "iteration"), required=True)
    parser.add_argument("--csv", required=True, help="Input training-log CSV path")
    parser.add_argument("--output", required=True, help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kind == "epoch":
        plot_training_loss(args.csv, args.output)
    else:
        plot_iteration_val_accuracy(args.csv, args.output)


if __name__ == "__main__":
    main()
