#!/usr/bin/env python3
"""
CICD launcher for training tests.
This script runs the training tests using Python unittest framework.
"""

import argparse
import os
import sys
import unittest
from pathlib import Path


def run_training_tests(test_pattern="test_*.py", verbose=False, failfast=False):
    """
    Run training tests using Python unittest.

    Args:
        test_pattern: Pattern to match test files
        verbose: Whether to run tests in verbose mode
        failfast: Whether to stop on first failure
    """
    # Get the directory containing this script
    test_dir = Path(__file__).parent

    # Discover and run tests
    loader = unittest.TestLoader()

    # For the new folder structure, we need to discover tests recursively
    # Start from the current directory and search all subdirectories
    suite = loader.discover(
        str(test_dir), pattern=test_pattern, top_level_dir=str(test_dir)
    )

    # Create test runner
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2, failfast=failfast)
    else:
        runner = unittest.TextTestRunner(verbosity=1, failfast=failfast)

    # Run tests
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run training tests for CICD")
    parser.add_argument(
        "--test-pattern",
        default="test_*.py",
        help="Pattern to match test files (default: test_*.py)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count for testing")

    args = parser.parse_args()

    # Set GPU count environment variable if specified
    if args.gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(args.gpu_count)
        )
        print(f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Run tests
    print(f"Running training tests with pattern: {args.test_pattern}")
    success = run_training_tests(
        test_pattern=args.test_pattern, verbose=args.verbose, failfast=args.failfast
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
