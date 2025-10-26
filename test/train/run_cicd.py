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


def run_training_tests(test_pattern="test_*.py", verbose=False, failfast=False, model_name=None):
    """
    Run training tests using Python unittest.

    Args:
        test_pattern: Pattern to match test files
        verbose: Whether to run tests in verbose mode
        failfast: Whether to stop on first failure
        model_name: Optional model name to run tests only for that model. If None, runs tests for all models.
    """
    # Get the directory containing this script
    test_dir = Path(__file__).parent

    # If a specific model is requested, search only in that model's directory
    if model_name:
        model_dir = test_dir / model_name
        if not model_dir.exists():
            print(f"Error: Model directory '{model_name}' not found in {test_dir}")
            print(
                f"Available models: {', '.join([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])}"
            )
            return False
        search_dir = model_dir
        print(f"Running tests for model: {model_name}")
    else:
        search_dir = test_dir
        print("Running tests for all models")

    # Discover and run tests
    loader = unittest.TestLoader()

    # For the new folder structure, we need to discover tests recursively
    # Start from the current directory and search all subdirectories
    suite = loader.discover(str(search_dir), pattern=test_pattern, top_level_dir=str(test_dir))

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count for testing")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name to run tests only for that model (e.g., qwen2_5, qwen2_5_vl, qwen3_vl). Default: None (tests all models)",
    )

    args = parser.parse_args()

    # Set GPU count environment variable if specified
    if args.gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpu_count))
        print(f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Run tests
    print(f"Running training tests with pattern: {args.test_pattern}")
    success = run_training_tests(
        test_pattern=args.test_pattern,
        verbose=args.verbose,
        failfast=args.failfast,
        model_name=args.model_name,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
