"""
EditReward model loading module for server.

Environment variables:
    EDITREWARD_MODEL_PATH: Path to base model (required)
    EDITREWARD_CHECKPOINT_PATH: Path to checkpoint (required)
"""
import os
import warnings

warnings.filterwarnings("ignore")

from lmms_engine.utils.rewards.editreward_service.editreward_scorer import (
    EditRewardScorer,
)


def load_editreward(
    model_name_or_path: str = None,
    checkpoint_path: str = None,
    device: str = "cuda",
):
    """
    Load EditReward model and return inference function.

    Args:
        model_name_or_path: Base model path (default from EDITREWARD_MODEL_PATH env)
        checkpoint_path: Checkpoint path (default from EDITREWARD_CHECKPOINT_PATH env)
        device: Device to use (default: cuda)

    Returns:
        inference function that takes (prompts, source_images, edited_images)

    Raises:
        ValueError: If required paths are not provided
    """
    model_path = model_name_or_path or os.environ.get("EDITREWARD_MODEL_PATH")
    ckpt_path = checkpoint_path or os.environ.get("EDITREWARD_CHECKPOINT_PATH")

    if not model_path:
        raise ValueError("model_name_or_path is required. Set EDITREWARD_MODEL_PATH env or pass argument.")
    if not ckpt_path:
        raise ValueError("checkpoint_path is required. Set EDITREWARD_CHECKPOINT_PATH env or pass argument.")

    print(f"Loading EditReward model from {model_path}")
    print(f"Loading checkpoint from {ckpt_path}")
    print(f"Using device: {device}")

    scorer = EditRewardScorer(
        checkpoint_path=ckpt_path,
        model_name_or_path=model_path,
        device=device,
        reward_dim="overall_detail",
        rm_head_type="ranknet_multi_head",
        pooling_strategy="mean",
        use_special_tokens=True,
        output_dim=2,
        rm_head_kwargs=None,
    )

    print("EditReward model loaded successfully!")

    def inference_fn(prompts, source_images, edited_images):
        """Run EditReward inference."""
        scores = scorer.score(prompts, source_images, edited_images)
        if hasattr(scores, "cpu"):
            scores = scores.cpu()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return scores

    return inference_fn
