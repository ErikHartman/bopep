import time
import hashlib
from typing import Any, Dict, Optional


_structure_cache: Dict[str, Dict[str, Any]] = {}


def _get_cache_key(filepath: str) -> str:
    """Generate content-based cache key using file hash."""
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


def _is_cache_valid(filepath: str) -> bool:
    """Check if cached data is still valid based on file content hash."""
    cache_key = _get_cache_key(filepath)
    return cache_key in _structure_cache


def get_from_cache(filepath: str, data_type: str) -> Any:
    """Get data from cache if valid, otherwise return None."""
    cache_key = _get_cache_key(filepath)
    if cache_key not in _structure_cache:
        return None

    cache_entry = _structure_cache[cache_key]
    cache_entry["last_accessed"] = time.time()

    # Handle nested data access (e.g., 'coordinates.A_CA')
    if "." in data_type:
        main_type, sub_type = data_type.split(".", 1)
        if main_type in cache_entry and isinstance(cache_entry[main_type], dict):
            return cache_entry[main_type].get(sub_type)
        return None

    return cache_entry.get(data_type)


def store_in_cache(filepath: str, data_type: str, data: Any) -> None:
    """Store data in cache with content-based key."""
    cache_key = _get_cache_key(filepath)

    if cache_key not in _structure_cache:
        _structure_cache[cache_key] = {
            "filename": filepath,
            "last_accessed": time.time(),
        }

    cache_entry = _structure_cache[cache_key]

    # Handle nested data storage (e.g., 'coordinates.A_CA')
    if "." in data_type:
        main_type, sub_type = data_type.split(".", 1)
        if main_type not in cache_entry:
            cache_entry[main_type] = {}
        cache_entry[main_type][sub_type] = data
    else:
        cache_entry[data_type] = data

    cache_entry["last_accessed"] = time.time()


def clear_structure_cache(filepath: Optional[str] = None) -> None:
    """Clear cache for a specific file or all files if filepath is None."""
    if filepath is None:
        _structure_cache.clear()
    else:
        cache_key = _get_cache_key(filepath)
        _structure_cache.pop(cache_key, None)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    total_entries = len(_structure_cache)
    total_size = 0
    data_types: Dict[str, int] = {}
    unique_files = set()

    for cache_entry in _structure_cache.values():
        # Track unique files by their original filepath
        if "filepath" in cache_entry:
            unique_files.add(cache_entry["filepath"])

        for key, value in cache_entry.items():
            if key not in ["cache_key", "filepath", "last_accessed"]:
                data_types[key] = data_types.get(key, 0) + 1
                # Rough size estimation
                if hasattr(value, "__len__"):
                    total_size += len(str(value))

    return {
        "total_entries": total_entries,
        "unique_files_cached": len(unique_files),
        "estimated_size_chars": total_size,
        "data_types": data_types,
    }


def get_cache_info() -> Dict[str, Any]:
    """Get detailed cache information for debugging."""
    cache_info = {}

    for cache_key, cache_entry in _structure_cache.items():
        filepath = cache_entry.get("filepath", "unknown")
        last_accessed = cache_entry.get("last_accessed", 0)

        stored_types = [
            key
            for key in cache_entry.keys()
            if key not in ["cache_key", "filepath", "last_accessed"]
        ]

        cache_info[cache_key] = {
            "filepath": filepath,
            "last_accessed": time.ctime(last_accessed) if last_accessed else "never",
            "stored_data_types": stored_types,
            "cache_key_short": (
                cache_key[:8] + "..." if len(cache_key) > 8 else cache_key
            ),
        }

    return cache_info
