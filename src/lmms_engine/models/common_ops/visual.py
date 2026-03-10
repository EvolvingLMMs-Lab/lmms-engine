def parse_visual_output(output):
    """Parse visual encoder output (Qwen2.5 VL / Omni)."""
    if isinstance(output, tuple):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def parse_visual_output_with_deepstack(output):
    """Parse visual encoder output with deepstack features (Qwen3 VL)."""
    if isinstance(output, tuple):
        return output
    if hasattr(output, "pooler_output") and hasattr(output, "deepstack_features"):
        return output.pooler_output, output.deepstack_features
    if hasattr(output, "pooler_output"):
        return output.pooler_output, None
    return output, None
