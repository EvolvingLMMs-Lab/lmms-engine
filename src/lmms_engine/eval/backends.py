import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class EvalServerBackend:
    """
    Backend for submitting evaluation jobs to an LMMS-Eval server asynchronously.

    This backend allows non-blocking evaluation submission to an eval server,
    with results retrieved in the background and logged when ready.

    Usage:
        backend = EvalServerBackend(url="http://localhost:8000", poll_interval=20.0)
        backend.submit_eval("/path/to/checkpoint", step=100)
        # Later in training loop
        for eval_step, metrics in backend.check_and_get_completed():
            tracking.log(metrics, step=eval_step)
    """

    def __init__(
        self,
        url: str,
        poll_interval: float = 20.0,
        eval_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the eval server backend.

        Args:
            url: Base URL of the eval server (e.g., "http://localhost:8000")
            poll_interval: Seconds between status checks (default: 20.0)
            eval_config: Eval configuration (model, tasks, model_args, etc.)
        """
        self.url = url.rstrip("/")
        self.poll_interval = poll_interval
        self.eval_config = eval_config or {}
        self.checkpoint_key = self.eval_config.get("checkpoint_key", "model")
        try:
            from lmms_eval.entrypoints import EvalClient

            self.client = EvalClient(base_url=self.url)
            logger.info(f"EvalServerBackend initialized with server: {self.url}")
        except ImportError:
            logger.error("Failed to import EvalClient from lmms_eval. Install lmms_eval to use eval server backend.")
            raise

        self.pending_evals: Dict[str, int] = {}
        self.results_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._poll_results, daemon=True)
        self.worker_thread.start()
        logger.info("EvalServerBackend worker thread started")

    def submit_eval(
        self,
        checkpoint_dir: str,
        step: int,
        eval_output_dir: str,
        checkpoint_type: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Submit an evaluation job to the server (non-blocking).

        Args:
            checkpoint_dir: Path to the checkpoint directory to evaluate
            step: Training step number (for logging results later)

        Returns:
            Job ID if submission was successful, None otherwise
        """
        model = self.eval_config.get("model")
        tasks = self.eval_config.get("tasks", [])
        model_args = self.eval_config.get("model_args", {}).copy()

        if not model or not tasks:
            logger.error(f"Missing model or tasks in eval_config: {self.eval_config}")
            return None

        model_args[self.checkpoint_key] = checkpoint_dir

        lmms_engine_kwargs = None
        if checkpoint_type is not None:
            lmms_engine_kwargs = {
                "model_path": checkpoint_dir,
                "checkpoint_type": checkpoint_type,
                "output_path": output_path,
            }

        try:
            job_response = self.client.evaluate(
                model=model,
                tasks=tasks,
                model_args=model_args,
                num_fewshot=self.eval_config.get("num_fewshot"),
                batch_size=self.eval_config.get("batch_size"),
                device=self.eval_config.get("device"),
                limit=self.eval_config.get("limit"),
                gen_kwargs=self.eval_config.get("gen_kwargs"),
                log_samples=self.eval_config.get("log_samples", True),
                predict_only=self.eval_config.get("predict_only", False),
                num_gpus=self.eval_config.get("num_gpus", 1),
                output_dir=eval_output_dir,
                lmms_engine_kwargs=lmms_engine_kwargs,
            )

            job_id = job_response.get("job_id")
            if job_id:
                self.pending_evals[job_id] = step
                logger.info(f"Eval job {job_id[:8]}... submitted for step {step}")
                return job_id
            else:
                logger.error(f"Job submission failed: {job_response}")
                return None

        except Exception as e:
            logger.error(f"Failed to submit eval job: {e}")
            return None

    def check_and_get_completed(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Get completed evaluation results (non-blocking).

        Returns:
            List of (step, metrics_dict) tuples for completed evaluations
        """
        completed = []
        while not self.results_queue.empty():
            try:
                completed.append(self.results_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def _poll_results(self):
        """
        Background thread that polls for job status and processes completed jobs.

        This runs in a daemon thread and automatically terminates when the process exits.
        """
        while True:
            time.sleep(self.poll_interval)

            if not self.pending_evals:
                continue

            jobs_to_remove = []
            for job_id in list(self.pending_evals.keys()):
                try:
                    job = self.client.get_job(job_id)
                    status = job.get("status")

                    if status == "completed":
                        step = self.pending_evals.pop(job_id)
                        result_data = job.get("result", {})
                        metrics = self._parse_eval_results(result_data)

                        if metrics:
                            self.results_queue.put((step, metrics))
                            logger.info(f"Eval job {job_id[:8]}... completed for step {step}")
                        else:
                            logger.warning(f"Eval job {job_id[:8]}... completed but no metrics extracted")

                    elif status == "failed":
                        error = job.get("error", "Unknown error")
                        step = self.pending_evals.pop(job_id)
                        logger.error(f"Eval job {job_id[:8]}... failed for step {step}: {error}")

                    elif status in ("queued", "running"):
                        pass

                    else:
                        step = self.pending_evals.pop(job_id)
                        logger.warning(f"Eval job {job_id[:8]}... has unexpected status: {status}")

                except Exception as e:
                    logger.error(f"Error checking job {job_id[:8]}...: {e}")
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                self.pending_evals.pop(job_id, None)

    def _parse_eval_results(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse evaluation results from eval server response.

        The result_data format (from _parse_output_directory):
        {
            "model_name": {
                "results": "/path/to/.../timestamp_results.json",
                "samples": ["/path/to/.../timestamp_samples_*.jsonl"]
            }
        }

        Args:
            result_data: Parsed result data from eval server

        Returns:
            Dict of metrics in tracking format: {metric_key: metric_value}
        """
        metrics = {}

        if not result_data:
            return metrics

        for model_name, model_data in result_data.items():
            results_file = model_data.get("results")
            if not results_file:
                continue

            try:
                with open(results_file, "r") as f:
                    results_json = json.load(f)

                task_results = results_json.get("results", {})
                for task_name, task_metrics in task_results.items():
                    for metric_key, metric_value in task_metrics.items():
                        if "_stderr," in metric_key or "_pass_at_k," in metric_key:
                            continue

                        metric_name = metric_key.split(",")[0]
                        tracking_key = f"eval/{task_name}_{metric_name}"

                        if isinstance(metric_value, str) and metric_value == "N/A":
                            metric_value = float("nan")
                        elif isinstance(metric_value, list) and len(metric_value) == 0:
                            continue

                        try:
                            metrics[tracking_key] = float(metric_value)
                        except (ValueError, TypeError):
                            pass

            except Exception as e:
                logger.error(f"Error parsing results file {results_file}: {e}")

        return metrics

    def close(self):
        """
        Clean up resources (mainly for testing purposes).

        Note: The daemon thread will automatically terminate when the process exits.
        """
        if hasattr(self, "client"):
            self.client.close()
        logger.info("EvalServerBackend closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
