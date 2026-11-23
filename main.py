"""Command-line entry point for the City Hotel (H2) ML project."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="City Hotel (H2) booking demand ML project")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("analyze", help="Dataset information and EDA")

    for name in ("train", "tune", "evaluate"):
        p = sub.add_parser(name, help=f"{name.capitalize()} models")
        p.add_argument("--task", choices=["regression", "classification"], required=True)
        p.add_argument("--model", default="all")

    p = sub.add_parser("cluster", help="Run clustering analysis")
    p.add_argument("--algorithm", default="kmeans")

    p = sub.add_parser("predict", help="Predict using saved artifacts")
    p.add_argument("--task", choices=["regression", "classification"], required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"'{args.command}' is not implemented yet.")


if __name__ == "__main__":
    main()
