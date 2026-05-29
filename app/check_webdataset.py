#!/usr/bin/env python3
# inspect_webdataset.py
"""
Quick script to inspect WebDataset tar files and show their key patterns.
"""

import argparse
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path


def inspect_tar_file(tar_path, num_samples=5):
    """
    Inspect a tar file and show the structure of samples.
    
    Args:
        tar_path: Path to the .tar file
        num_samples: Number of samples to inspect
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {tar_path}")
    print(f"{'='*80}\n")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Get all members
            members = tar.getmembers()
            print(f"Total files in tar: {len(members)}\n")
            
            # Group files by their base name (key)
            samples = defaultdict(list)
            for member in members:
                if member.isfile():
                    # Extract base name and extension
                    name = member.name
                    # Handle both "00001.mp4" and "path/00001.mp4" formats
                    basename = os.path.basename(name)
                    
                    # Split on first dot to get key and extension
                    parts = basename.split('.', 1)
                    if len(parts) == 2:
                        key, ext = parts
                        samples[key].append(ext)
                    else:
                        # No extension
                        samples[basename].append('(no extension)')
            
            # Show sample structure
            print(f"Number of samples (unique keys): {len(samples)}\n")
            
            # Show first N samples
            print(f"First {num_samples} samples:\n")
            for i, (key, extensions) in enumerate(sorted(samples.items())[:num_samples]):
                print(f"  Sample {i+1}:")
                print(f"    __key__: '{key}'")
                print(f"    Extensions: {sorted(extensions)}")
                
                # Show actual file sizes
                for ext in sorted(extensions):
                    filename = f"{key}.{ext}"
                    # Try to find the member (might have path prefix)
                    for member in members:
                        if member.name.endswith(filename):
                            size_kb = member.size / 1024
                            print(f"      {ext}: {size_kb:.2f} KB")
                            
                            # If it's a text file, show content
                            if ext in ['txt', 'cls', 'label', 'json']:
                                try:
                                    f = tar.extractfile(member)
                                    if f:
                                        content = f.read().decode('utf-8').strip()
                                        print(f"        Content: '{content}'")
                                except Exception as e:
                                    print(f"        (Could not read: {e})")
                            break
                print()
            
            # Show extension statistics
            print("\nExtension statistics:")
            ext_counts = defaultdict(int)
            for extensions in samples.values():
                for ext in extensions:
                    ext_counts[ext] += 1
            
            for ext, count in sorted(ext_counts.items()):
                print(f"  .{ext}: {count} files")
            
            # Check for consistency
            print("\nConsistency check:")
            extension_sets = defaultdict(int)
            for extensions in samples.values():
                ext_set = tuple(sorted(extensions))
                extension_sets[ext_set] += 1
            
            if len(extension_sets) == 1:
                print("  ✓ All samples have the same file structure")
            else:
                print("  ✗ Samples have different file structures:")
                for ext_set, count in sorted(extension_sets.items(), key=lambda x: -x[1]):
                    print(f"    {count} samples with: {list(ext_set)}")
                    
    except Exception as e:
        print(f"Error inspecting tar file: {e}")
        import traceback
        traceback.print_exc()


def inspect_dataset_directory(data_path, num_samples=5):
    """
    Inspect all tar files in a dataset directory.
    
    Args:
        data_path: Path to dataset directory containing .tar files
        num_samples: Number of samples to inspect per tar file
    """
    data_path = Path(data_path)
    
    # Check for metadata
    meta_path = data_path / "metadata.json"
    if meta_path.exists():
        print(f"\n{'='*80}")
        print(f"Metadata: {meta_path}")
        print(f"{'='*80}\n")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            print(json.dumps(meta, indent=2))
    else:
        print(f"\nWarning: No metadata.json found in {data_path}")
    
    # Find all tar files
    tar_files = sorted(data_path.glob("*.tar"))
    
    if not tar_files:
        print(f"\nNo .tar files found in {data_path}")
        return
    
    print(f"\nFound {len(tar_files)} tar file(s)")
    
    # Inspect first tar file in detail
    if tar_files:
        inspect_tar_file(tar_files[0], num_samples=num_samples)
    
    # Show summary of other tar files
    if len(tar_files) > 1:
        print(f"\n{'='*80}")
        print(f"Other tar files in directory:")
        print(f"{'='*80}\n")
        for tar_file in tar_files[1:]:
            size_mb = tar_file.stat().st_size / (1024 * 1024)
            print(f"  {tar_file.name}: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect WebDataset tar files to understand their structure"
    )
    parser.add_argument(
        "path",
        help="Path to either a .tar file or a directory containing .tar files"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=5,
        help="Number of samples to inspect (default: 5)"
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file() and path.suffix == '.tar':
        # Inspect single tar file
        inspect_tar_file(path, num_samples=args.num_samples)
    elif path.is_dir():
        # Inspect directory
        inspect_dataset_directory(path, num_samples=args.num_samples)
    else:
        print(f"Error: Path must be either a .tar file or a directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
