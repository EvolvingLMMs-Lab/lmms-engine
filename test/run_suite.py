import argparse
from pathlib import Path

from utils import run_unittest_files

current_dir = Path(__file__).parent.resolve()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        choices=["all", "dataset", "models", "utils", "train"],
        default="all",
        help="The test suite you want to test, default to test all files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    if args.test == "all":
        files = [p for p in current_dir.glob("**/test_*.py")]
    else:
        files = [p for p in current_dir.glob(f"{args.test}/test_*.py")]

    exit_code = run_unittest_files(files)
    exit(exit_code)
