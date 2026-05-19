#!/usr/bin/env python3
"""Convert LLaVA-Video-178K JSON data to lmms-engine parquet format.

Reads the original LLaVA-Video-178K JSON files (ShareGPT format with
``from``/``value`` keys and ``<image>`` as video placeholder) and converts
them to the OpenAI-style message format expected by lmms-engine.

Output parquet schema:
    - ``id``: str  — sample identifier
    - ``messages``: str (JSON)  — OpenAI-style chat messages

Video paths in the output ``video_url`` are relative to the
``data_folder`` specified in the training config.

Usage:
    python convert_llava_video_to_parquet.py \
        --input_dir /data/v-kaichen/azure_blob/data/LLaVA-Video-178K \
        --output_dir ./data/LLaVA-Video-178K \
        --splits 0_30_s \
        --max_per_split 0

    ``--splits`` selects which duration buckets to include.
    ``--max_per_split`` limits rows per source JSON (0 = no limit).
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────

# System prompt for normal video QA — tells the model to watch the full
# video in silence and only respond after the video ends and the user
# asks a question.
NORMAL_VIDEO_QA_SYSTEM_PROMPT = (
    "You are a helpful video assistant. Watch the video carefully and "
    "remain silent while it plays. Once the video ends and the user asks "
    "a question, provide a clear and accurate answer."
)


def convert_sharegpt_to_openai(entry: dict) -> Dict:
    """Convert a single ShareGPT-format entry to OpenAI message format.

    The original format:
        {"id": "...", "conversations": [{"from": "human", "value": "..."},
         {"from": "gpt", "value": "..."}], "video": "path/to/video.mp4"}

    The ``<image>`` token in the human turn is used as a video placeholder.
    We convert it to a ``video_url`` content item.

    A system message is prepended to instruct the model to stay silent
    during video playback and answer only after the question is asked.
    """
    conversations = entry["conversations"]
    video_path = entry.get("video", None)
    entry_id = entry.get("id", "unknown")

    messages: List[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": NORMAL_VIDEO_QA_SYSTEM_PROMPT}],
        }
    ]
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        value = turn["value"]

        content: List[dict] = []
        if role == "user" and "<image>" in value:
            # Replace <image> with video_url content item
            # Strip the <image> and any surrounding whitespace/newlines
            text = value.replace("<image>", "").strip()
            text = text.lstrip("\n").strip()
            if video_path is not None:
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": video_path},
                    }
                )
            if text:
                content.append({"type": "text", "text": text})
        else:
            content.append({"type": "text", "text": value})

        messages.append({"role": role, "content": content})

    return {"id": str(entry_id), "messages": json.dumps(messages)}


def process_json_file(json_path: str, max_rows: int = 0) -> List[dict]:
    """Load a single LLaVA-Video JSON file and convert all entries."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if max_rows > 0:
        data = data[:max_rows]

    rows = []
    for entry in data:
        try:
            row = convert_sharegpt_to_openai(entry)
            rows.append(row)
        except Exception as e:
            print(f"  Skipping entry {entry.get('id', '?')}: {e}")
    return rows


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA-Video-178K to lmms-engine parquet")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/v-kaichen/azure_blob/data/LLaVA-Video-178K",
        help="Root directory of LLaVA-Video-178K data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/LLaVA-Video-178K",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["0_30_s"],
        help="Duration bucket prefixes to include (e.g. 0_30_s 30_60_s 1_2_m 2_3_m)",
    )
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=0,
        help="Max rows per source JSON file (0 = no limit)",
    )
    parser.add_argument(
        "--include_types",
        type=str,
        nargs="+",
        default=["cap", "oe", "mc"],
        help="Data types to include: cap (caption), oe (open-ended), mc (multi-choice)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSON files matching the requested splits
    all_json_files = []
    for split_prefix in args.splits:
        pattern = os.path.join(args.input_dir, f"{split_prefix}*", "*_processed.json")
        matched = sorted(glob.glob(pattern))
        all_json_files.extend(matched)

    if not all_json_files:
        print(f"No JSON files found for splits: {args.splits}")
        print(f"Searched pattern: {args.input_dir}/<split>*/*_processed.json")
        return

    # Filter by include_types
    filtered_files = []
    for f in all_json_files:
        basename = os.path.basename(f)
        for t in args.include_types:
            # Match patterns like *_cap_*, *_oe_*, *_mc_*
            if f"_{t}_" in basename or f"_{t}." in basename:
                filtered_files.append(f)
                break

    print(f"Found {len(filtered_files)} JSON files to convert:")
    for f in filtered_files:
        print(f"  {os.path.relpath(f, args.input_dir)}")

    # Process all files
    all_rows = []
    for json_path in filtered_files:
        rel_path = os.path.relpath(json_path, args.input_dir)
        print(f"\nProcessing: {rel_path}")
        rows = process_json_file(json_path, max_rows=args.max_per_split)
        print(f"  -> {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows converted.")
        return

    # Write to parquet
    splits_tag = "_".join(args.splits)
    types_tag = "_".join(args.include_types)
    output_path = os.path.join(args.output_dir, f"llava_video_{splits_tag}_{types_tag}.parquet")

    table = pa.table(
        {
            "id": pa.array([r["id"] for r in all_rows], type=pa.string()),
            "messages": pa.array([r["messages"] for r in all_rows], type=pa.string()),
        }
    )

    pq.write_table(table, output_path)
    print(f"\nWrote {len(all_rows)} rows to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Print a sample row for verification
    sample = json.loads(all_rows[0]["messages"])
    print(f"\nSample row (id={all_rows[0]['id']}):")
    print(json.dumps(sample, indent=2)[:1000])


if __name__ == "__main__":
    main()
