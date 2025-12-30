#!/usr/bin/env python3
"""
Pre-warm Multi-SWE Docker images by pulling them before training.

This script downloads all Docker images from the Multi-SWE dataset to local storage,
eliminating pull time during training.

Usage:
    # Pull all images (may take hours for full dataset)
    python warm_images.py

    # Pull first 100 images only
    python warm_images.py --limit 100

    # Pull with higher concurrency
    python warm_images.py --concurrent 16
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from typing import Set

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def check_image_exists(image: str) -> bool:
    """Check if image already exists locally."""
    try:
        result = await asyncio.create_subprocess_exec(
            "docker", "inspect", image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await result.wait()
        return result.returncode == 0
    except Exception:
        return False


async def pull_image(image: str, semaphore: asyncio.Semaphore) -> tuple[str, bool, str]:
    """
    Pull a single Docker image.
    
    Returns:
        Tuple of (image_name, success, message)
    """
    async with semaphore:
        # Check if already exists
        if await check_image_exists(image):
            logger.debug(f"Image already exists: {image}")
            return (image, True, "already exists")
        
        logger.info(f"Pulling: {image}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info(f"Pulled: {image}")
                return (image, True, "pulled")
            else:
                error_msg = stderr.decode().strip()
                logger.warning(f"Failed to pull {image}: {error_msg}")
                return (image, False, error_msg)
        except Exception as e:
            logger.error(f"Error pulling {image}: {e}")
            return (image, False, str(e))


def get_all_images(dataset_name: str, split: str = "train", limit: int | None = None) -> Set[str]:
    """
    Extract all unique Docker image names from the dataset.
    
    Returns:
        Set of full image names (e.g., "mswebench/pandas-dev_m_pandas:pr-12345")
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    images = set()
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        image = f"mswebench/{row['org']}_m_{row['repo']}:pr-{row['number']}".lower()
        images.add(image)
    
    logger.info(f"Found {len(images)} unique images")
    return images


async def warm_all_images(
    dataset_name: str = "PrimeIntellect/Multi-SWE-RL",
    split: str = "train",
    max_concurrent: int = 8,
    limit: int | None = None,
) -> dict:
    """
    Pre-pull all Multi-SWE Docker images.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        max_concurrent: Maximum concurrent pulls
        limit: Maximum number of images to pull (None for all)
    
    Returns:
        Dictionary with statistics about the warming process
    """
    images = get_all_images(dataset_name, split, limit)
    
    logger.info(f"Starting to warm {len(images)} images with {max_concurrent} concurrent pulls")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [pull_image(img, semaphore) for img in images]
    results = await asyncio.gather(*tasks)
    
    # Collect statistics
    stats = {
        "total": len(results),
        "success": 0,
        "already_existed": 0,
        "failed": 0,
        "failed_images": [],
    }
    
    for image, success, message in results:
        if success:
            stats["success"] += 1
            if message == "already exists":
                stats["already_existed"] += 1
        else:
            stats["failed"] += 1
            stats["failed_images"].append((image, message))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-warm Multi-SWE Docker images for faster training"
    )
    parser.add_argument(
        "--dataset",
        default="PrimeIntellect/Multi-SWE-RL",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="Maximum concurrent image pulls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to pull (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Multi-SWE image warming")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Concurrent pulls: {args.concurrent}")
    if args.limit:
        logger.info(f"Limit: {args.limit} images")
    
    stats = asyncio.run(
        warm_all_images(
            dataset_name=args.dataset,
            split=args.split,
            max_concurrent=args.concurrent,
            limit=args.limit,
        )
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("WARMING COMPLETE")
    print("=" * 60)
    print(f"Total images:     {stats['total']}")
    print(f"Successfully pulled: {stats['success'] - stats['already_existed']}")
    print(f"Already existed:  {stats['already_existed']}")
    print(f"Failed:           {stats['failed']}")
    
    if stats["failed_images"]:
        print("\nFailed images:")
        for image, error in stats["failed_images"][:10]:
            print(f"  - {image}: {error}")
        if len(stats["failed_images"]) > 10:
            print(f"  ... and {len(stats['failed_images']) - 10} more")
    
    print("=" * 60)
    
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

