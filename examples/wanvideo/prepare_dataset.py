#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 WanVideo team. All rights reserved.
"""
Dataset preparation script for WanVideo training.

This script helps prepare video datasets for training WanVideo models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_metadata_json(
    video_dir: str,
    output_path: str,
    video_extensions: List[str] = [".mp4", ".avi", ".mov", ".webm"],
    caption_file_ext: str = ".txt",
    metadata_format: str = "json",
) -> None:
    """
    Create metadata file for video dataset.
    
    Args:
        video_dir: Directory containing video files
        output_path: Path to save metadata file
        video_extensions: List of video file extensions to include
        caption_file_ext: Extension for caption files
        metadata_format: Format of metadata file (json or jsonl)
    """
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    # Scan for video files
    for ext in video_extensions:
        for video_file in video_dir.glob(f"**/*{ext}"):
            # Look for corresponding caption file
            caption_file = video_file.with_suffix(caption_file_ext)
            
            entry = {
                "video_path": str(video_file.relative_to(video_dir)),
                "video_id": video_file.stem,
            }
            
            # Add caption if file exists
            if caption_file.exists():
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                entry["caption"] = caption
            else:
                entry["caption"] = ""  # Empty caption if no file
                logger.warning(f"No caption file found for {video_file}")
            
            # Add optional metadata
            entry["duration"] = None  # Can be filled with actual duration
            entry["fps"] = None  # Can be filled with actual FPS
            entry["resolution"] = None  # Can be filled with actual resolution
            
            metadata.append(entry)
    
    # Save metadata
    if metadata_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    elif metadata_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Created metadata file with {len(metadata)} entries: {output_path}")


def create_i2v_metadata(
    video_dir: str,
    image_dir: str,
    output_path: str,
    video_extensions: List[str] = [".mp4", ".avi", ".mov", ".webm"],
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"],
) -> None:
    """
    Create metadata file for Image-to-Video dataset.
    
    Args:
        video_dir: Directory containing video files
        image_dir: Directory containing first frame images
        output_path: Path to save metadata file
        video_extensions: List of video file extensions
        image_extensions: List of image file extensions
    """
    video_dir = Path(video_dir)
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    # Create mapping of video stems to video files
    video_files = {}
    for ext in video_extensions:
        for video_file in video_dir.glob(f"**/*{ext}"):
            video_files[video_file.stem] = video_file
    
    # Match images with videos
    for ext in image_extensions:
        for image_file in image_dir.glob(f"**/*{ext}"):
            stem = image_file.stem
            
            # Look for corresponding video
            if stem in video_files:
                video_file = video_files[stem]
                
                entry = {
                    "video_path": str(video_file.relative_to(video_dir)),
                    "image_path": str(image_file.relative_to(image_dir)),
                    "video_id": stem,
                    "caption": "",  # Can be filled with captions
                }
                
                metadata.append(entry)
            else:
                logger.warning(f"No matching video found for image: {image_file}")
    
    # Save metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created I2V metadata file with {len(metadata)} entries: {output_path}")


def validate_dataset(
    metadata_path: str,
    video_root: str = None,
    check_files: bool = True,
) -> None:
    """
    Validate dataset metadata and optionally check file existence.
    
    Args:
        metadata_path: Path to metadata file
        video_root: Root directory for video files
        check_files: Whether to check if files actually exist
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        if metadata_path.endswith(".jsonl"):
            metadata = [json.loads(line) for line in f]
        else:
            metadata = json.load(f)
    
    logger.info(f"Loaded {len(metadata)} entries from {metadata_path}")
    
    # Check required fields
    required_fields = ["video_path", "caption"]
    missing_fields = set()
    
    for i, entry in enumerate(metadata):
        for field in required_fields:
            if field not in entry:
                missing_fields.add(field)
                logger.error(f"Entry {i} missing required field: {field}")
    
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return
    
    # Check file existence if requested
    if check_files and video_root:
        video_root = Path(video_root)
        missing_files = []
        
        for entry in metadata:
            video_path = video_root / entry["video_path"]
            if not video_path.exists():
                missing_files.append(str(video_path))
        
        if missing_files:
            logger.error(f"Found {len(missing_files)} missing video files")
            for f in missing_files[:10]:  # Show first 10
                logger.error(f"  Missing: {f}")
        else:
            logger.info("All video files exist")
    
    logger.info("Dataset validation complete")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for WanVideo training"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create metadata command
    create_parser = subparsers.add_parser(
        "create", help="Create metadata file from video directory"
    )
    create_parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing videos"
    )
    create_parser.add_argument(
        "--output", type=str, required=True, help="Output metadata file path"
    )
    create_parser.add_argument(
        "--format", type=str, default="json", choices=["json", "jsonl"],
        help="Metadata file format"
    )
    
    # Create I2V metadata command
    i2v_parser = subparsers.add_parser(
        "create_i2v", help="Create I2V metadata file"
    )
    i2v_parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing videos"
    )
    i2v_parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    i2v_parser.add_argument(
        "--output", type=str, required=True, help="Output metadata file path"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset metadata"
    )
    validate_parser.add_argument(
        "--metadata", type=str, required=True, help="Metadata file path"
    )
    validate_parser.add_argument(
        "--video_root", type=str, help="Root directory for video files"
    )
    validate_parser.add_argument(
        "--check_files", action="store_true", help="Check if files exist"
    )
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_metadata_json(
            args.video_dir,
            args.output,
            metadata_format=args.format,
        )
    elif args.command == "create_i2v":
        create_i2v_metadata(
            args.video_dir,
            args.image_dir,
            args.output,
        )
    elif args.command == "validate":
        validate_dataset(
            args.metadata,
            args.video_root,
            args.check_files,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
