"Copied from sglang https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_utils.py"
import os
import subprocess
import time
from typing import List


def run_unittest_files(files: List[str]):
    tic = time.time()
    success = True

    for filename in files:

        def run_one_file(filename):
            filename = os.path.join(os.getcwd(), filename)
            print(f"\n\nRun:\npython3 {filename}\n\n", flush=True)
            process = subprocess.Popen(["python3", filename], stdout=None, stderr=None, env=os.environ)
            process.wait()
            return process.returncode

        try:
            ret_code = run_one_file(filename)
            assert ret_code == 0
        except Exception as e:
            time.sleep(5)
            print(
                f"\nError {e} when running {filename}\n",
                flush=True,
            )
            success = False
            break

    if success:
        print(f"Success. Time elapsed: {time.time() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.time() - tic:.2f}s", flush=True)

    return 0 if success else -1
