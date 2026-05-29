#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Converts a set of CSV-based file lists into a sharded WebDataset.

This script is designed to handle the specific multi-dataset, weighted
sampling setup for VJEPA. It creates a separate subdirectory of shards
for each input CSV, preserving the dataset boundaries needed for
weighted sampling and ablation studies.

Each sample in the .tar files will contain:
- 'video.mp4': The raw video file bytes
- 'label.txt': The label associated with the video
- '__key__': A unique identifier (e.g., '00000001')

It also creates a `metadata.json` in each directory with the total
number of samples for that dataset.
"""

import argparse
import os
import random
import sys
from pathlib import Path
import pandas as pd
import webdataset as wds
from tqdm import tqdm
import json


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert CSV file lists to sharded WebDataset"
    )
    parser.add_argument(
        "--csvs",
        type=str,
        nargs="+",
        required=True,
        help="List of input CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to write WebDataset shards.",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Number of samples per .tar shard file.",
    )
    parser.add_argument(
        "--shard_prefix",
        type=str,
        default=None,
        help="(Optional) A general prefix for all shard names. "
        "If not provided, the CSV filename will be used.",
    )
    return parser.parse_args()


def load_filepaths_from_csv(csv_file):
    """
    Loads file paths and labels from a CSV.
    Assumes CSV has no header and is space-delimited, with
    path in the first column and label in the second.
    """
    base_dir = os.path.dirname(csv_file)

    try:
        # Replicating the logic from your VideoDataset class
        try:
            # Try with space delimiter first
            df = pd.read_csv(csv_file, header=None, delimiter=" ")
        except pd.errors.ParserError:
            # Fallback to '::' delimiter
            print(f"Space delimiter failed, trying '::'...")
            df = pd.read_csv(csv_file, header=None, delimiter="::")
        
        # Paths are in column 0, labels in column 1
        path_col = 0
        label_col = 1
        
    except Exception as e:
        print(f"Error reading CSV {csv_file}: {e}")
        return [], []

    # Check if paths are absolute. If not, assume they are relative
    # to the CSV file's directory.
    def make_absolute(path):
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(base_dir, path))
        return path

    try:
        paths = df[path_col].apply(make_absolute).tolist()
        labels = df[label_col].astype(str).tolist() # Read labels as strings
    except Exception as e:
        print(f"Error processing columns for {csv_file}: {e}")
        return [], []

    return paths, labels


def write_webdataset(
    samples, shard_pattern, total_samples, shard_prefix, samples_per_shard
):
    """
    Writes a list of (path, label, key) samples to a sharded WebDataset.
    """
    print(f"  Writing {total_samples} samples...")
    
    # Use ShardWriter to handle automatic sharding
    try:
        with wds.ShardWriter(
            shard_pattern, 
            maxcount=samples_per_shard, 
            encoder=False
        ) as sink:
            
            for key, video_path, label_str in tqdm(
                samples,
                desc=f"Writing {shard_prefix}",
                total=total_samples,
                unit="samples"
            ):
                if not os.path.exists(video_path):
                    print(f"Warning: File not found, skipping: {video_path}")
                    continue

                try:
                    # Read the raw video file
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    # Create the sample
                    sample = {
                        "__key__": key,
                        "video.mp4": video_bytes,
                        "label.txt": label_str.encode('utf-8'),  # Encode string to bytes
                    }

                    # Write the sample to the tar archive
                    sink.write(sample)

                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    
    except Exception as e:
        print(f"An error occurred during writing: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = get_args()
    print(f"Starting WebDataset conversion for {len(args.csvs)} datasets...")
    print(f"Base output directory: {args.output_dir}\n")

    total_samples_all_datasets = 0

    for i, csv_file in enumerate(args.csvs):
        print(f"Processing dataset [{i+1}/{len(args.csvs)}]: {Path(csv_file).stem}")

        # 1. Load file paths and labels
        paths, labels = load_filepaths_from_csv(csv_file)
        if not paths:
            print(f"  No valid paths found in {csv_file}, skipping.")
            print("-" * 30)
            continue
        
        print(f"  Loaded {len(paths)} paths and labels from {csv_file}")

        # 2. Create a unique key for each sample and shuffle
        # We must pair key/path/label *before* shuffling
        total_samples = len(paths)
        total_samples_all_datasets += total_samples
        samples_to_write = [
            (f"{j:08d}", paths[j], labels[j]) for j in range(total_samples)
        ]
        
        print(f"  Shuffling {total_samples} file paths...")
        random.shuffle(samples_to_write)
        print("  Shuffling complete.")

        # 3. Set up output directory and shard pattern
        shard_prefix = args.shard_prefix if args.shard_prefix else Path(csv_file).stem
        shard_prefix = shard_prefix.replace("path_", "") # Clean up prefix
        
        dataset_output_dir = os.path.join(args.output_dir, shard_prefix)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        shard_pattern = os.path.join(
            dataset_output_dir, f"{shard_prefix}-%06d.tar"
        )
        print(f"  Writing shards to: {dataset_output_dir}")

        # 4. Write to WebDataset
        write_webdataset(
            samples_to_write,
            shard_pattern,
            total_samples,
            shard_prefix,
            args.samples_per_shard,
        )
        
        # 5. --- NEW: Write metadata file ---
        metadata = {
            "name": shard_prefix,
            "sample_count": total_samples,
            "shard_count": (total_samples + args.samples_per_shard - 1) // args.samples_per_shard
        }
        metadata_path = os.path.join(dataset_output_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Wrote metadata to {metadata_path}")
        except Exception as e:
            print(f"  Error writing metadata: {e}")
        # --- END NEW ---

        print(f"Finished processing {shard_prefix}.")
        print("-" * 30)

    print(f"\nAll datasets converted. Total samples: {total_samples_all_datasets}")


if __name__ == "__main__":
    main()