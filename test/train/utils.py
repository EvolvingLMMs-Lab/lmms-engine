import os
import shutil
import subprocess
import tempfile
import time
from functools import wraps


# Copied from https://github.com/axolotl-ai-cloud/axolotl/blob/main/tests/e2e/utils.py
def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Pass the temporary directory to the test function
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            # Clean up the directory after the test
            shutil.rmtree(temp_dir)

    return wrapper


def get_available_gpus():
    """Get the number of available GPUs."""
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def launch_torchrun_training(
    script_path,
    output_dir,
    nproc_per_node=None,
    nnodes=1,
    node_rank=0,
    master_addr="127.0.0.1",
    master_port="8000",
    timeout=300,
):
    """
    Launch training using torchrun with subprocess and real-time streaming output.

    Args:
        script_path: Path to the training script
        output_dir: Output directory for training
        nproc_per_node: Number of processes per node (defaults to available GPUs)
        nnodes: Number of nodes
        node_rank: Rank of this node
        master_addr: Master address
        master_port: Master port
        timeout: Timeout in seconds for the training process

    Returns:
        StreamingResult: Result object with returncode, stdout, and stderr attributes
    """
    if nproc_per_node is None:
        nproc_per_node = get_available_gpus()
        if nproc_per_node == 0:
            nproc_per_node = 1

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path,
        "--output_dir",
        output_dir,
        "--nproc_per_node",
        str(nproc_per_node),
        "--nnodes",
        str(nnodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
    ]

    # Launch the training process with streaming output
    print(f"Launching torchrun with command: {' '.join(cmd)}")

    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            cwd=os.getcwd(),
        )

        # Stream output in real-time
        output_lines = []
        start_time = time.time()

        while True:
            # Check if process is still running
            if process.poll() is not None:
                break

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                process.wait()
                print(f"Training process timed out after {timeout} seconds")
                return None

            # Read output line by line
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                print(f"[TRAINING] {line}")
                output_lines.append(line)
            else:
                # No output available, sleep briefly
                time.sleep(0.1)

        # Wait for process to complete and get return code
        return_code = process.wait()

        # Create a CompletedProcess-like result
        class StreamingResult:
            def __init__(self, returncode, stdout, stderr=None):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        return StreamingResult(return_code, "\n".join(output_lines))

    except Exception as e:
        print(f"Error launching training: {e}")
        return None


def with_multi_gpu_training(test_func):
    """
    Decorator for tests that need multi-GPU training.
    Automatically detects available GPUs and launches training with appropriate configuration.
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Get available GPUs
        nproc_per_node = get_available_gpus()
        if nproc_per_node == 0:
            print("No GPUs available, falling back to single process")
            nproc_per_node = 1

        # Pass GPU configuration to the test function
        test_func(*args, nproc_per_node=nproc_per_node, **kwargs)

    return wrapper
