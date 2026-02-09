"""Dataset utilities for discovering and loading shards."""

from pathlib import Path


def discover_shards(dataset_dir: Path, split: str) -> list[tuple[str, str]]:
    """Discover all shards for a given split across all tasks.

    Args:
        dataset_dir: Root directory of NL dataset.
        split: Split name (train, tuning, held_out).

    Returns:
        List of (task_name, shard_name) tuples.
    """
    shards = []
    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue

        split_dir = task_dir / split
        if not split_dir.exists():
            continue

        for shard_file in split_dir.glob("*.parquet"):
            shards.append((task_dir.name, shard_file.stem))

    return sorted(shards)
