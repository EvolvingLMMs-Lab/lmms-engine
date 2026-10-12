"""Extract a small OpenMMReasoner subset without changing the source dataset."""

import argparse
import shutil
import zipfile
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/OpenMMReasoner-SFT-874K"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", default="OpenVLThinker-sft-iter3.parquet")
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")
    source = pq.ParquetFile(args.data_root / "parquet" / args.shard)
    table = next(source.iter_batches(batch_size=args.samples))
    paths = {
        content["image_url"]["url"]
        for row in table.to_pylist()
        for message in row["messages"]
        for content in message["content"]
        if content["type"] == "image_url"
    }
    for path in paths:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"Unsafe image path: {path}")
    args.output.mkdir(parents=True)
    remaining = set(paths)
    for archive in sorted((args.data_root / "image_zip").glob("*.zip")):
        with zipfile.ZipFile(archive) as zipped:
            matches = remaining.intersection(zipped.namelist())
            for path in sorted(matches):
                destination = args.output / "images" / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(path) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            remaining -= matches
    if remaining:
        raise FileNotFoundError(f"Images absent from archives: {sorted(remaining)[:10]}")
    import pyarrow as pa

    pq.write_table(pa.Table.from_batches([table]), args.output / "samples.parquet")
    print(f"Prepared {table.num_rows} samples and {len(paths)} images in {args.output}")


if __name__ == "__main__":
    main()
