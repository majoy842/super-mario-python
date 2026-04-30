from __future__ import annotations


def detect_mode(comment_count: int) -> str:
    if comment_count == 1:
        return "single"
    if 2 <= comment_count <= 9:
        return "small_batch"
    if comment_count >= 10:
        return "full_batch"
    return "empty"

